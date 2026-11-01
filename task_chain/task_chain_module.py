# 接收用户问题，使用模型分析问题的主题    ->    根据主题调用 FAISS Retriever   ->     问题、主题和检索结果交给模型，生成结构化回答

# 整个任务链的数据不断累积： 
# 第一阶段
# {
#     "question": "员工如何申请年假？",
#     "topic": "员工年假申请条件和流程"
# }

# 第二阶段
# {
#     "question": "...",
#     "topic": "...",
#     "documents": [...],
#     "context": "...",
#     "source_texts": {...}
# }

# 第三阶段
# {
#     "question": "...",
#     "topic": "...",
#     "documents": [...],
#     "context": "...",
#     "source_texts": {...},
#     "answer": EnterpriseAnswer(...)
# }

import sys
import os

# 获取当前文件所在目录（memory），然后获取其上一级目录（项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 将项目根目录加入到系统搜索路径中
if project_root not in sys.path:
    sys.path.append(project_root)

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough
)

from common.model import DeepSeekChatModel
from prompt_engineering.enterprise_parser import (
    EnterpriseAnswerParser
)
from prompt_engineering.answer_schema import (
    EnterpriseAnswer
)
from task_chain.debug_callback import DebugCallbackHandler


class TaskChainModule:
    """企业级问答任务链模块"""

    def __init__(
        self,
        model=None,
        retriever=None,
        parser=None,
        callback_handler=None
    ):
        """
        Args:
            model: LangChain ChatModel
            retriever: FAISS Retriever
            parser: 企业问答输出解析器
            callback_handler: 调试回调处理器
        """

        self.model = model or DeepSeekChatModel()

        if retriever is None:
            raise ValueError("必须提供知识库 Retriever")

        self.retriever = retriever

        self.parser = parser or EnterpriseAnswerParser()

        self.callback_handler = (
            callback_handler or DebugCallbackHandler()
        )

        # 创建任务链
        self.task_chain = self.create_task_chain()

    def create_task_chain(self):
        """构建三个连续子任务组成的任务链"""

        # =====================================
        # 任务一：分析用户问题，确定查询主题
        # =====================================

        analysis_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
你是企业知识库查询分析助手。

你的任务是分析用户问题，提取最适合用于知识库检索的查询主题。

要求：
1. 准确识别用户真正想解决的问题。
2. 提取核心业务主题。
3. 保留重要的业务条件和限制。
4. 删除无关表达。
5. 只输出查询主题，不要输出其他解释。
"""
            ),
            (
                "human",
                "请分析以下问题：\n{question}"
            )
        ])

        analysis_chain = (
            analysis_prompt
            | self.model
            | StrOutputParser()
        )

        # =====================================
        # 任务二：根据主题检索知识库
        # =====================================

        def retrieve_documents(inputs, config=None):
            """
            根据任务一产生的主题检索知识库。

            返回原始文档和格式化上下文。
            """

            topic = inputs["topic"]

            # 通过 Retriever 检索
            documents = self.retriever.invoke(
                topic,
                config=config
            )

            source_texts = {}
            context_parts = []

            for index, doc in enumerate(documents):

                # 获取文档 ID
                source_id = doc.metadata.get(
                    "source_id",
                    f"source_{index + 1}"
                )

                content = doc.page_content

                source_texts[source_id] = content

                context_parts.append(
                    f"[来源ID：{source_id}]\n{content}"
                )

            context = "\n\n".join(context_parts)

            return {
                **inputs,
                "documents": documents,
                "context": context,
                "source_texts": source_texts,
                "retrieval_query": topic
            }

        retrieval_step = RunnableLambda(
            retrieve_documents
        )

        # =====================================
        # 任务三：生成结构化企业回答
        # =====================================

        answer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
你是企业内部知识库问答助手。

请严格根据提供的知识库内容回答用户问题。

回答规则：
1. 只能使用知识库内容支持的事实。
2. 不得编造制度、流程、时间、数字或业务规则。
3. 如果知识库没有足够信息，明确说明信息不足。
4. 如果知识库中的内容存在冲突，标记为 conflict。
5. 每条重要事实尽可能提供对应的文档引用。
6. 引用的 source_id 必须来自给定的知识库内容。
7. quote 必须是对应文档中的真实原文。
8. 不要把模型推测当作企业事实。

请按照以下格式输出：

{format_instructions}
"""
            ),
            (
                "human",
                """
用户问题：
{question}

问题分析得到的查询主题：
{topic}

知识库检索结果：
{context}

请结合以上内容生成最终回答。
"""
            )
        ]).partial(
            format_instructions=(
                self.parser.get_format_instructions()
            )
        )

        generation_chain = (
            answer_prompt
            | self.model
            | self.parser
        )

        # =====================================
        # 组合三个子任务
        # =====================================

        task_chain = (
            # 保留 question，并添加 topic
            RunnablePassthrough.assign(
                topic=analysis_chain
            )

            # 使用 topic 检索，并添加检索结果
            | retrieval_step

            # 保留之前的所有字段，并添加结构化回答
            | RunnablePassthrough.assign(
                answer=generation_chain
            )

            # 最终结果校验
            | RunnableLambda(
                self._validate_result
            )
        )

        return task_chain

    def _validate_result(self, inputs: dict) -> EnterpriseAnswer:
        """
        校验模型回答中的引用是否真实存在。

        注意：
        这里进行的是引用真实性校验，
        并不等同于完整的语义事实核验。
        """

        answer = inputs["answer"]
        source_texts = inputs["source_texts"]

        for citation in answer.citations:

            source_id = citation.source_id
            quote = citation.quote

            # 检查文档 ID
            if source_id not in source_texts:
                raise ValueError(
                    f"引用了不存在的文档 ID：{source_id}"
                )

            # 检查引用原文
            source_text = source_texts[source_id]

            if quote not in source_text:
                raise ValueError(
                    f"引用内容与原文不匹配：{quote}"
                )

        return answer

    def execute_task_chain(
        self,
        question: str
    ) -> EnterpriseAnswer:
        """
        执行任务链并处理异常。

        成功：返回结构化回答。
        失败：返回信息不足状态的结构化结果。
        """

        if not question or not question.strip():
            return EnterpriseAnswer(
                status="insufficient_information",
                answer="用户问题不能为空。",
                citations=[],
                confidence=0.0,
                follow_up_questions=[
                    "请提供需要查询的具体问题。"
                ]
            )

        try:
            # 执行整个任务链
            result = self.task_chain.invoke(
                {"question": question},
                config={
                    "callbacks": [
                        self.callback_handler
                    ],
                    "run_name": "enterprise_task_chain",
                    "tags": [
                        "enterprise_qa",
                        "task_chain"
                    ],
                    "metadata": {
                        "module": "TaskChainModule"
                    }
                }
            )

            return result

        except Exception as e:

            # 记录任务链执行异常
            self.callback_handler.logger.exception(
                "[TASK CHAIN FAILED] question=%s error=%s",
                question,
                str(e)
            )

            # 返回统一的结构化失败结果
            return EnterpriseAnswer(
                status="insufficient_information",
                answer=(
                    "抱歉，系统在处理该问题时发生异常，"
                    "暂时无法生成可靠回答。"
                ),
                citations=[],
                confidence=0.0,
                follow_up_questions=[
                    "请稍后重试，或联系系统管理员。"
                ]
            )


