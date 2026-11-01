import json
import hashlib
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


class DocumentStorage:
    """企业文档加载、向量化和 FAISS 检索"""

    def __init__(
        self,
        embeddings: Embeddings,
        index_path: str = "faiss_index",
        chunk_size: int = 800,
        chunk_overlap: int = 120
    ):
        self.embeddings = embeddings
        self.index_path = Path(index_path)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

        self.vector_store: Optional[FAISS] = None

    def load_documents(self, directory: str) -> list[Document]:
        """
        加载目录下的 txt、md、pdf、json 文件。
        """
        directory = Path(directory)#type:ignore

        if not directory.exists():#type:ignore
            raise FileNotFoundError(
                f"文档目录不存在: {directory}"
            )

        documents = []

        # 1. 加载 TXT
        txt_loader = DirectoryLoader(
            str(directory),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True
        )
        documents.extend(txt_loader.load())

        # 2. 加载 Markdown
        md_loader = DirectoryLoader(
            str(directory),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True
        )
        documents.extend(md_loader.load())

        # 3. 加载 PDF
        pdf_loader = DirectoryLoader(
            str(directory),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents.extend(pdf_loader.load())

        # 4. 加载 JSON
        for json_file in directory.rglob("*.json"):
            documents.extend(
                self._load_json_file(json_file)
            )

        # 5. 统一补充元数据
        for index, doc in enumerate(documents):
            if "source_id" not in doc.metadata:
                source = str(doc.metadata.get("source", "unknown"))

                source_id = hashlib.sha1(
                    source.encode("utf-8")
                ).hexdigest()[:12]

                doc.metadata["source_id"] = source_id

            doc.metadata.setdefault(
                "file_type",
                Path(doc.metadata.get("source", "")).suffix.lstrip(".")
            )

        print(f"共加载 {len(documents)} 个文档单元")

        return documents

    def _load_json_file(self, file_path: Path) -> list[Document]:
        """
        加载单个 JSON 文件，并转换为 LangChain Document。

        支持：
        1. JSON 对象
        2. JSON 数组
        3. 嵌套 JSON 结构
        """
        documents = []

        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        # 根据相对路径生成稳定的 source_id
        source_id = hashlib.sha1(
            str(file_path.resolve()).encode("utf-8")
        ).hexdigest()[:12]

        # 如果顶层是数组，则每个元素单独作为一个 Document
        if isinstance(data, list):
            for index, item in enumerate(data):
                content = json.dumps(
                    item,
                    ensure_ascii=False,
                    indent=2
                )

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path),
                            "source_id": source_id,
                            "file_type": "json",
                            "record_index": index
                        }
                    )
                )

        else:
            # 顶层是对象或其他 JSON 数据类型
            content = json.dumps(
                data,
                ensure_ascii=False,
                indent=2
            )

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "source_id": source_id,
                        "file_type": "json"
                    }
                )
            )

        return documents

    def split_documents(
        self,
        documents: list[Document]
    ) -> list[Document]:
        """切分文档，并为每个文本块生成唯一 ID"""

        chunks = self.text_splitter.split_documents(
            documents
        )

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = (
                f"chunk_{index + 1:06d}"
            )

        return chunks

    def build_vector_store(
        self,
        documents: list[Document]
    ) -> FAISS:
        """生成 Embedding 并创建 FAISS 索引"""

        if not documents:
            raise ValueError("没有可用于向量化的文档")

        chunks = self.split_documents(documents)

        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        return self.vector_store

    def save(self, path: Optional[str] = None):
        """保存 FAISS 索引"""

        if self.vector_store is None:
            raise ValueError("FAISS 向量库尚未初始化")

        save_path = Path(path) if path else self.index_path

        save_path.mkdir(
            parents=True,
            exist_ok=True
        )

        self.vector_store.save_local(str(save_path))

    def load(self, path: Optional[str] = None) -> FAISS:
        """从本地加载 FAISS 索引"""

        load_path = Path(path) if path else self.index_path

        if not load_path.exists():
            raise FileNotFoundError(
                f"FAISS 索引不存在：{load_path}"
            )

        # 只加载可信来源、由自己生成的索引文件。
        self.vector_store = FAISS.load_local(
            str(load_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        return self.vector_store

    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> list[Document]:
        """相似性搜索"""

        if self.vector_store is None:
            raise ValueError("请先构建或加载 FAISS 索引")

        return self.vector_store.similarity_search(
            query,
            k=k
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ):
        """带距离分数的相似性搜索"""

        if self.vector_store is None:
            raise ValueError("请先构建或加载 FAISS 索引")

        return self.vector_store.similarity_search_with_score(
            query,
            k=k
        )

    def as_retriever(self, k: int = 4):
        """将 FAISS 转换为 LangChain Retriever"""

        if self.vector_store is None:
            raise ValueError("请先构建或加载 FAISS 索引")

        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )


