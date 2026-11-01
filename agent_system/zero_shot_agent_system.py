
from langchain.agents import create_agent

from prompt_engineering.enterprise_parser import (
    EnterpriseAnswerParser
)


class ZeroShotAgentSystem:
    """不依赖企业知识库的 Zero-shot Agent"""

    def __init__(
        self,
        model,
        parser=None
    ):
        self.model = model
        self.parser = parser or EnterpriseAnswerParser()

        self.agent = self.create_agent()

    def create_agent(self):
        """创建不使用外部工具的 Agent"""

        system_prompt = """
你是一个通用知识问答助手。

你的任务是回答用户提出的常识性、技术性和一般知识问题。

工作规则：

1. 根据已有知识直接回答。
2. 不要访问或假设存在企业内部知识库。
3. 对于不确定的信息，应明确说明不确定性。
4. 不得伪造事实、数据或参考来源。
5. 回答应准确、清晰，并具有适当的解释。
6. citations 必须为空数组，因为本 Agent 不使用企业文档。
7. 如果问题不完整，可以提出进一步的问题。
8. 最终严格输出 JSON，不要输出 Markdown 代码块。

JSON 格式：

{
    "status": "answered | insufficient_information | conflict",
    "answer": "string",
    "citations": [],
    "confidence": 0.90,
    "follow_up_questions": ["string"]
}
"""

        # 不提供工具，直接使用模型回答
        return create_agent(
            model=self.model,
            tools=[],
            system_prompt=system_prompt,
            name="zero_shot_agent"
        )

    def _extract_final_content(self, result: dict) -> str:
        """提取最终模型回答"""

        messages = result.get("messages", [])

        for message in reversed(messages):
            if getattr(message, "type", None) == "ai":
                if message.content:
                    if isinstance(message.content, str):
                        return message.content

                    return str(message.content)

        raise ValueError("Zero-shot Agent 没有返回答案")

    def invoke(
        self,
        question: str,
        config=None
    ):
        """执行常识性问题回答"""

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

        return self.parser.parse(content)
