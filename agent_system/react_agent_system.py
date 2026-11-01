
from langchain.agents import create_agent
from langchain.tools import tool

from langchain_core.documents import Document

from prompt_engineering.enterprise_parser import (
    EnterpriseAnswerParser
)

from agent_system.document_storage import DocumentStorage


class ReActAgentSystem:
    """基于企业知识库的 ReAct Agent"""

    def __init__(
        self,
        model,
        document_storage: DocumentStorage,
        parser=None,
        top_k: int = 4
    ):
        self.model = model
        self.document_storage = document_storage
        self.parser = parser or EnterpriseAnswerParser()
        self.top_k = top_k

        # 创建知识库检索工具
        self.search_tool = self._create_search_tool()

        # 创建 ReAct Agent
        self.agent = self.create_agent()

    def _create_search_tool(self):
        """创建企业知识库检索工具"""

        storage = self.document_storage
        top_k = self.top_k

        @tool
        def search_enterprise_documents(query: str) -> str:
            """
            根据查询内容搜索企业内部知识库。

            当用户询问公司制度、员工福利、业务流程、
            产品资料、内部规定等企业专属信息时，
            必须调用此工具。

            Args:
                query: 需要在企业知识库中检索的问题或主题。

            Returns:
                检索到的文档 ID 和原始内容。
            """

            documents = storage.similarity_search(
                query=query,
                k=top_k
            )

            if not documents:
                return "知识库中没有检索到相关文档。"

            results = []

            for index, doc in enumerate(documents):
                source_id = doc.metadata.get(
                    "source_id",
                    f"unknown_{index}"
                )

                source = doc.metadata.get(
                    "source",
                    "unknown"
                )

                results.append(
                    f"""
文档编号：{source_id}
文档来源：{source}
文档内容：
{doc.page_content}
"""
                )

            return "\n\n".join(results)

        return search_enterprise_documents

    def create_agent(self):
        """创建 ReAct Agent"""

        system_prompt = """
你是企业内部知识库问答 Agent。

你的任务是根据企业知识库回答用户的问题。

工作规则：

1. 对于涉及企业内部制度、流程、员工福利、
   产品资料、公司规定等问题，必须调用
   search_enterprise_documents 工具。

2. 如果第一次检索的信息不足，可以调整关键词再次检索。

3. 只能依据检索到的企业文档回答企业内部事实。

4. 不允许编造企业制度、时间、金额和业务规则。

5. 如果知识库没有相关信息，应明确说明信息不足。

6. 如果不同文档之间存在冲突，应指出冲突，
   不得自行选择一个结果作为事实。

7. 回答中需要给出引用，引用 ID 必须来自工具返回的文档。

8. 最终严格输出 JSON，不要输出 Markdown 代码块。

JSON 格式：

{
    "status": "answered | insufficient_information | conflict",
    "answer": "string",
    "citations": [
        {
            "source_id": "string",
            "quote": "string"
        }
    ],
    "confidence": 0.95,
    "follow_up_questions": ["string"]
}
"""

        return create_agent(
            model=self.model,
            tools=[self.search_tool],
            system_prompt=system_prompt,
            name="enterprise_react_agent"
        )

    def _extract_final_content(self, result: dict) -> str:
        """从 Agent 执行结果中提取最终 AI 消息"""

        messages = result.get("messages", [])

        for message in reversed(messages):
            if getattr(message, "type", None) == "ai":
                if message.content:
                    if isinstance(message.content, str):
                        return message.content

                    # 兼容多模态或分块 content
                    return str(message.content)

        raise ValueError("ReAct Agent 没有返回最终答案")

    def invoke(
        self,
        question: str,
        config=None
    ):
        """执行企业知识问答"""

        if not question or not question.strip():
            raise ValueError("问题不能为空")

        result = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            },
            config=config
        )

        content = self._extract_final_content(result)

        # 复用之前实现的企业问答解析器
        return self.parser.parse(content)
