# 将检索模块传来的 list[Documents] 格式化进入 Prompt 上下文：

from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from qa_template import EnterpriseQAPrompt   
from enterprise_parser import EnterpriseAnswerParser

def format_documents(documents):
    """
    将 Document 列表转换为 Prompt 上下文
    """

    context_parts = []

    for i, doc in enumerate(documents):

        metadata = doc.metadata or {}

        # 优先使用 Chunk 唯一编号
        source_id = (
            metadata.get("chunk_id")
            or metadata.get("document_id")
            or metadata.get("source_id")
            or f"doc_{i + 1}"
        )

        source = metadata.get("source", "未知来源")
        page = metadata.get("page", "未知")

        context_parts.append(
            f"""
[资料编号：{source_id}]
[来源：{source}]
[页码：{page}]

{doc.page_content}
"""
        )

    return "\n\n".join(context_parts)


class PromptManager:
    """企业提示词管理器"""
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

        # 1. 初始化 Prompt
        self.prompt = EnterpriseQAPrompt().get_prompt()

        # 2. 初始化输出解析器：
        self.parser = EnterpriseAnswerParser()

        # 3. 构建企业问答链
        self.qa_chain = self._build_qa_chain()


    # --------------------------------------------------
    # 1. 格式化文档
    # --------------------------------------------------
    @staticmethod
    def _prepare_documents(documents: list[Document]) -> dict:

        context_parts = []
        source_texts = {}

        for i, doc in enumerate(documents):

            metadata = doc.metadata or {}

            source_id = (
                metadata.get("chunk_id")
                or metadata.get("document_id")
                or metadata.get("source_id")
                or f"doc_{i + 1}"
            )

            if source_id in source_texts:
                raise ValueError(
                    f"本次检索存在重复 source_id：{source_id}"
                )

            source = metadata.get("source", "未知来源")
            page = metadata.get("page", "未知")

            source_texts[source_id] = doc.page_content

            context_parts.append(
                f"""
[资料编号：{source_id}]
[来源：{source}]
[页码：{page}]

{doc.page_content}
"""
            )

        return {
            "context": "\n\n".join(context_parts),
            "source_texts": source_texts
        }


    # --------------------------------------------------
    # 2. 准备模型输入
    # --------------------------------------------------
    def _prepare_input(self, inputs: dict) -> dict:
        """
        :param inputs: 包含'question'和'documents'的字典，分别为 用户输入的问题 以及 FAISS 检索到的文档
        :return: 包含'question'、'context' 和 'source_texts' 的字典（以便提供给模型输入），其中 后两个来自 'documents'
        """

        question = inputs.get("question")
        documents = inputs.get("documents")

        if not question:
            raise ValueError("用户问题不能为空")

        if documents is None:
            documents = []

        prepared = self._prepare_documents(documents)

        return {
            "question": question,               # 用户问题
            "context": prepared["context"],     # 文档内容
            "source_texts": prepared["source_texts"]
        }


    # --------------------------------------------------
    # 3. 校验引用来源
    # --------------------------------------------------
    @staticmethod
    def _validate_result(inputs):

        answer_by_model_and_parsed = inputs["parsed"]
        source_texts = inputs["source_texts"]       
        # source_texts 是 FAISS 检索得到的，一定是真实的，因此，只需要将模型给我的引用来源反馈和 source_texts 作比较，
        # 就可以校验引用来源的真实性
        # source_texts 就是从 FAISS 检索结果中提取出来的原始文档内容，而 parsed 是模型根据这些内容生成的回答。
        # 后续验证阶段，就是将模型的回答与真实检索内容进行比较
        # https://chatgpt.com/c/6ae58483-8e68-83e8-b0ee-a9d8288e43bf

        for citation in answer_by_model_and_parsed.citations:

            source_id = citation.source_id

            # 检查引用编号是否存在
            if source_id not in source_texts:
                raise OutputParserException(
                    f"引用了不存在的资料编号：{source_id}"
                )

            # 检查引用内容是否来自对应文档
            original_text = source_texts[source_id]

            # 进行简单的空白字符归一化
            normalize = lambda s: " ".join(s.split())

            if normalize(citation.quote) not in normalize(original_text):
                raise OutputParserException(
                    f"引用内容与原始资料不匹配：{source_id}"
                )

        # 返回标准化答案
        return answer_by_model_and_parsed


    # --------------------------------------------------
    # 4. 构建企业问答 LCEL 链
    # --------------------------------------------------
    def _build_qa_chain(self):

        # Prompt -> Model -> Parser
        generation_chain = (
            self.prompt
            | self.model
            | self.parser
        )

        # 同时保留本次检索的原始资料
        parallel_chain = RunnableParallel(
            parsed=generation_chain,                # 需要 question 和 context，并给出模型的输出 parsed 
            source_texts=RunnableLambda(            # 只取 字典中 source_texts 的键值对
                lambda x: x["source_texts"]
            )
        )
        # 两个执行分支：
        # parsed:    Prompt -> Model -> Parser      输出：EnterpriseAnswer 答案模式
        # source_texts:        提取原始文档          输出：文档编号与原文的映射字典
        # parallel_chain 后续的 self._validate_result() 不仅需要模型生成的答案 parserd，还需要原始文档信息 source_texts

        # 完整执行链
        chain = (
            RunnableLambda(self._prepare_input)
            | parallel_chain
            | RunnableLambda(self._validate_result)     # 接收上游的模型输出 parsed 和故意保留的 source_texts
        )

        return chain


    # --------------------------------------------------
    # 5. 对外统一调用接口
    # --------------------------------------------------
    def answer(
        self,
        question: str,
        documents: list
    ):
        return self.qa_chain.invoke({
            "question": question,
            "documents": documents
        })


    # --------------------------------------------------
    # 6. 获取原始 Prompt
    # --------------------------------------------------
    def get_qa_prompt(self):
        return self.prompt


    # --------------------------------------------------
    # 7. 获取原始 Parser
    # --------------------------------------------------
    def get_parser(self):
        return self.parser

