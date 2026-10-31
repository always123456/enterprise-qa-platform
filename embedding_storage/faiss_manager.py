# FAISS 向量存储管理器， 负责：
# 1. 接收 Document 对象；
# 2. Document 转换为向量；
# 3. 创建 FAISS 索引；
# 4. 保存和加载索引；
# 5. 添加和删除文档；
# 6. 提供相似性检索接口；
# 7. 提供 LangChain Retriever 接口

import os
import uuid
from typing import Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_core.vectorstores import VectorStoreRetriever

from embedding_storage.embedding_model import EmbeddingModelManagerWithHuggingFace


class FAISSManager:
    """FAISS 向量数据库管理器"""
    def __init__(
            self,
            embedding_model_manager: EmbeddingModelManagerWithHuggingFace,
            index_path: str = "vector_db/faiss_index",
        ) -> None:
        self.embedding_model_manager = embedding_model_manager
        self.embeddings = self.embedding_model_manager.get_embeddings()

        self.index_path = Path(index_path)
        self.vector_store: Optional[FAISS]= None


    # --------------------------------------------------
    # 1. 创建向量 + 索引
    # --------------------------------------------------
    def build_index(
            self,
            documents: list[Document],
            ids: list[str] | None = None
    ) -> FAISS:
        """根据 Document 创建 FAISS 索引"""

        # 1. 检查文档
        if not documents:
            raise ValueError("documents 不能为空")

        documents = [
            doc for doc in documents
            if doc.page_content.strip()
        ]

        if not documents:
            raise ValueError("没有有效的文档内容")


        # 2. 为每个文档块生成唯一 ID
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        if len(ids) != len(documents):
            raise ValueError("ids 数量必须与 documents 数量一致")

        if (len(set(ids))) != len(ids):
            raise ValueError("文档 ID 不能重复")


        # 3. 创建 FAISS 索引
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
            ids=ids,
        )

        return self.vector_store


    # --------------------------------------------------
    # 2. 保存向量索引
    # --------------------------------------------------
    def save_index(self) -> None:
        """将 FAISS 索引持久化保存到本地"""
        if self.vector_store is None:
            raise RuntimeError("当前没有可保存到本地的 FAISS 索引")

        self.index_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.vector_store.save_local(
            folder_path=str(self.index_path),
        )

        print(f"FAISS 索引已保存到本地：{self.index_path}")


    # --------------------------------------------------
    # 3. 加载向量索引
    # --------------------------------------------------
    def load_index(self) -> FAISS:
        """从本地加载 FAISS 索引"""
        if not self.index_path.exists():
            raise FileNotFoundError(f"索引目录不存在：{self.index_path}")

        self.vector_store = FAISS.load_local(
            folder_path=str(self.index_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

        print("FAISS 索引加载成功")

        return self.vector_store


    # --------------------------------------------------
    # 4. 添加新文档
    # --------------------------------------------------
    def add_documents(
            self,
            documents: list[Document],
            ids: list[str] | None = None,
    ) -> list[str]:
        """
        向现有索引中添加新文档
        :param documents: 待添加的 Documents 对象列表
        :param ids: 文档 ID 列表，默认为 None，调用方传入或我自动生成的 uuid4
        :return: 成功添加的 文档 ID 列表
        """
        if self.vector_store is None:
            raise RuntimeError("请先创建或加载 FAISS 索引")

        documents = [
            doc for doc in documents
            if doc.page_content.strip()
        ]

        if not documents:
            return []

        if ids is None: # 调用方不传入，则自动生成 uuid4
            ids = [str(uuid.uuid4()) for _ in documents]

        if len(ids) != len(documents):
            raise ValueError("ids 数量必须与 documents 数量一致")

        if len(set(ids)) != len(ids):
            raise ValueError("文档 ID 必须唯一")

        result = self.vector_store.add_documents(
            documents=documents,
            ids=ids,
        )   # FAISS.add_documents() 会返回新添加的文档的 ID 列表

        return result


    # --------------------------------------------------
    # 5. 删除文档
    # --------------------------------------------------
    def delete_documents(self, ids: list[str]) -> (bool | None):
        """根据文档 ID 删除向量"""
        if self.vector_store is None:
            raise RuntimeError("请先创建或加载 FAISS 索引")

        return self.vector_store.delete(ids=ids)


    # --------------------------------------------------
    # 6. 相似性检索
    # --------------------------------------------------
    def similarity_search(
            self,
            query: str,
            k: int = 4,
            filter: dict | None = None,
    ) -> list[Document]:
        """根据查询文本 query，检索最相似的文档"""

        if self.vector_store is None:
            raise RuntimeError("请先创建或加载 FAISS 索引")

        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )


    # --------------------------------------------------
    # 7. 带距离分数的相似性检索
    # --------------------------------------------------
    def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            filter: dict | None = None,
    ):
        """返回文档及其向量距离分数"""
        if self.vector_store is None:
            raise RuntimeError("请先创建或加载 FAISS 索引")

        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )


    # --------------------------------------------------
    # 8. 获取 Retriever
    # --------------------------------------------------
    def as_retriever(
            self,
            search_type: str = "similarity",
            search_kwargs: dict | None = None,
    ) -> VectorStoreRetriever:
        """转换为 LangChain Retriever"""

        if self.vector_store is None:
            raise RuntimeError("请先创建或加载 FAISS 索引")

        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs or {"k": 4},
        )