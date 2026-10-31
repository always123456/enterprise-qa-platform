from langchain_core.prompts import ChatPromptTemplate

from qa_prompt import (ROLE_PROMPT, QA_TASK_PROMPT, QA_CONSTRAINT_PROMPT, QA_OUTPUT_PROMPT)


# 组合 Prompt 模块，形成模版:
class EnterpriseQAPrompt:
    """企业问答 Prompt 模版"""

    def __init__(self) -> None:
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
{role}

{task}

{constraints}

{output_format}
"""
            ),
            (
                "human",
                """
以下是企业知识库检索到的参考资料：

{context}

用户问题：
{question}

请根据上述资料完成回答。
"""
            )
        ]).partial(
            role=ROLE_PROMPT,
            task=QA_TASK_PROMPT,
            constraints=QA_CONSTRAINT_PROMPT,
            output_format=QA_OUTPUT_PROMPT,
        )

    def get_prompt(self) -> ChatPromptTemplate:
        return self.prompt