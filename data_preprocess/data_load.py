import os
import re
import logging
from typing import List, Optional, Callable

from langchain_core.documents import Document       # LangChain 不认 .txt/.json/.md 等等，只认 Document
from langchain_community.document_loaders import (
    TextLoader,
    JSONLoader,
    UnstructuredMarkdownLoader,
)

logger = logging.getLogger(__name__)


# ============================================================
# 1. 清洗管道（与格式解耦，可独立测试和扩展）
# ============================================================
def remove_control_chars(text: str) -> str:
    """移除不可打印控制字符（保留换行、制表符）。"""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

def normalize_whitespace(text: str) -> str:
    """规范空白：合并连续空格/制表符，压缩连续空行。"""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def remove_boilerplate(text: str) -> str:
    """
    移除常见 boilerplate 模式。
    企业场景可按需扩展：页眉页脚、版权声明、导航栏文本等。
    """
    patterns = [
        r"^\s*Copyright\s+©.*$",
        r"^\s*版权所有.*$",
        r"^\s*Confidential.*$",
        r"^\s*Page\s+\d+\s*(of\s+\d+)?\s*$",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.MULTILINE | re.IGNORECASE)
    return text

def filter_empty_docs(docs: List[Document], min_length: int = 10) -> List[Document]:
    """过滤清洗后内容过短的文档块。"""
    kept = []
    for doc in docs:
        content = doc.page_content.strip()
        if len(content) >= min_length:
            kept.append(doc)
        else:
            logger.debug(f"过滤空/过短文档: metadata={doc.metadata}")
    return kept

class CleanPipeline:
    """可组合的清洗管道"""
    def __init__(self, steps: Optional[List[Callable[[str], str]]] = None) -> None:
        self.steps = steps or [
            remove_control_chars,
            normalize_whitespace,
            remove_boilerplate,
        ]   # 清洗函数组合

    def run(self, docs: list[Document]) -> List[Document]:
        for doc in docs:
            text = doc.page_content
            for step in self.steps:
                text = step(text)
            doc.page_content = text
        return filter_empty_docs(docs=docs)


# ============================================================
# 2. 多格式加载器
# ============================================================
class DataLoaderModule:
    """
    多格式文档加载器：

    支持格式：.txt / .md / .json / .jsonl / .markdown
    """

    SUPPORTED_EXTS = {".txt", ".json", ".md", ".markdown", ".jsonl"} # 集合

    def __init__(
            self,
            #directory_path,
            clean_pipeline: Optional[CleanPipeline] = None,
            encoding: str = 'utf-8',
            json_jq_schema: str = '.',
            json_content_key: Optional[str] = None, # Optional[str]: 可以是 str，也可以是 None
            #json_lines: bool = False,
            markdown_mode: str = "single", # "single" | "elements"
            chunk_size=500,
            chunk_overlap=50,
        ) -> None:
        """
        初始化数据加载模块
        :param clean_pipeline: 自定义的数据清洗器
        :param encoding:
        :param json_jq_schema:  JSONLoader 的参数
        :param json_content_key: JSONLoader 的参数
        :param markdown_mode: UnstructuredMarkdownLoader 的参数
        :param chunk_size: 文档分割的每个块大小
        :param chunk_overlap: 文档块之间的重叠大小
        """
        self.encoding = encoding
        self.json_jq_schema = json_jq_schema
        self.json_content_key = json_content_key
        #self.json_lines = json_lines
        self.markdown_mode = markdown_mode
        self.cleaner = clean_pipeline or CleanPipeline()


    # ------------- 统一接口 ---------------
    def load(self, file_path: str) -> List[Document]:
        """
        文档加载，并清洗，分割成若干个 Document 文档块
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTS:
            raise ValueError(
                f"不支持的文档格式：{ext}，支持：{self.SUPPORTED_EXTS}"
            )
        logger.info(f"加载文件{file_path} (ext={ext})")

        if ext == '.txt':
            docs: list[Document] = self._load_txt(file_path)
        elif ext in {"json", "jsonl"}:
            docs: list[Document] = self._load_json(file_path, ext == "jsonl")
        else: # .md/.markdown
            docs: list[Document] = self._load_markdown(file_path)

        if not docs:
            logger.warning(f"文件加载为空：{file_path}")

        docs = self._enrich_metadata(docs, file_path)   # 元数据（Metadata）指的是描述数据的数据
        docs = self.cleaner.run(docs)
        logger.info(f"加载并清洗完成：{file_path} -> {len(docs)} 个文档块")

        return docs


    # ------------------------ 批量加载(目录) 对外接口----------------------------
    def load_batch(
            self,
            directory: str,
            recursive: bool = True,
            glob_pattern: Optional[str] = None,
    ) -> List[Document]:
        """
        批量加载目录下的支持格式的文件，加载策略：
        - recursive=True 时，递归扫描（目录下的目录也会被扫描）；
        - glob_pattern 可覆盖默认匹配
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"不是目录：{directory}")

        all_docs: List[Document] = []

        if recursive:
            walker = os.walk(directory)
            for root, _, files in walker:
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in self.SUPPORTED_EXTS:
                        path = os.path.join(root, f)
                        try:
                            all_docs.extend(self.load(path))
                        except Exception as e:
                            logger.error(f"加载失败{path}：{e}", exc_info=True)
        else:
            for f in os.listdir(directory):
                path = os.path.join(directory, f)
                if os.path.isfile(path) and os.path.splitext(f)[1].lower() in self.SUPPORTED_EXTS:
                    try:
                        all_docs.extend(self.load(path))
                    except Exception as e:
                        logger.error(f"加载失败{path}：{e}", exc_info=True)

        logger.info(f"批量加载完成：{directory} -> 共 {len(all_docs)} 个文档块")
        return all_docs


    # ---------------- 各个格式实现 -------------------
    def _load_txt(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path, encoding=self.encoding)
        return loader.load()

    def _load_json(self, file_path: str, is_jsonl: bool) -> List[Document]:
        """
        JSON 加载策略
        - jq_schema: 默认加载策略为 "."，即整体加载；
        - content_key: 若指定 json_content_key，则从 jq 结果中提取指定字段作为 Document 的 page_content
        - text_content: 决定 content_key 是否生效，text_content 为 False 时，content_key 生效
        """
        loader = JSONLoader(
            file_path=file_path, 
            jq_schema=self.json_jq_schema,                      
            content_key=self.json_content_key,
            text_content=self.json_content_key is None,
            json_lines=is_jsonl,
        )
        """
        JSONLoader:
        :param jq_schema: 使用 jq 语法编写的查询字符串，从复杂的 JSON 结构中提取目标数据 (str, jq 属于处理 JSON 的命令行工具)
        :param content_key: 指定 JSON 对象中的哪一个键(key)的值作为 Document 的 正文内容 page_content
        :param text_content: 是否将 jq 提取出来的内容直接当做纯文本（正文）处理
        :param json_lines: 传入的文件是否是 .jsonl 格式
        """
        return loader.load()

    def _load_markdown(self, file_path: str) -> List[Document]:
        """
        Markdown 加载：
        - "single":  整篇为一个 Document (适合短文档)
        - "elements": 按语义元素(标题、段落等)拆分为多个 Document，保留
        """
        # mode = None if self.markdown_mode == 'single' else "elements"
        loader = UnstructuredMarkdownLoader(
            file_path=file_path,
            mode=self.markdown_mode,
        )
        return loader.load()


    # ------------------ 元数据增强 -----------------------:
    def _enrich_metadata(self, docs: list[Document], file_path: str) -> List[Document]:
        """
        为每个 Document 对象的 metadata 属性(文档打的附属属性或标签) 添加多个元数据。
        可以根据业务要求进行更改， 这里补充企业级加载常用的通用元数据 :  source/filename/file_type
        :param docs: 待处理的 Document 列表
        :param file_path: 所有 Document 的来源文件地址
        """
        abs_path = os.path.abspath(file_path)
        file_name = os.path.basename(file_path)
        for doc in docs:
            doc.metadata.setdefault("source", abs_path)
            doc.metadata.setdefault("filename", file_name)
            doc.metadata.setdefault("file_type", os.path.splitext(file_name)[1].lstrip("."))
        return docs


if __name__ == "__main__":
    loader = DataLoaderModule()
    docs = loader.load_batch("./dataset")

    for doc in docs:
        print(doc.page_content)