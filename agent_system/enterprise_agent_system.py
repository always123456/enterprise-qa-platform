
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from agent_system.react_agent_system import ReActAgentSystem
from agent_system.zero_shot_agent_system import ZeroShotAgentSystem
from agent_system.document_storage import DocumentStorage


class EnterpriseAgentSystem:
    """企业级多 Agent 路由系统"""

    def __init__(
        self,
        model: BaseChatModel,
        document_storage: DocumentStorage,
        parser=None,
        top_k: int = 4
    ):
        self.model = model
        self.document_storage = document_storage

        # 创建企业知识库 Agent
        self.react_agent = ReActAgentSystem(
            model=model,
            document_storage=document_storage,
            parser=parser,
            top_k=top_k
        )

        # 创建通用知识 Agent
        self.zero_shot_agent = ZeroShotAgentSystem(
            model=model,
            parser=parser
        )

        # 创建问题分类链
        self.router_chain = self._create_router_chain()

    def _create_router_chain(self):
        """创建问题分类链"""

        router_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
你是企业问答系统的问题分类器。

你的任务是判断用户问题属于哪一种类型：

enterprise：
涉及企业内部专属信息，例如：
- 公司制度和管理规定
- 员工福利、考勤、年假、薪酬
- 企业业务流程
- 内部产品、项目和技术文档
- 企业内部数据和信息

general：
涉及通用知识，例如：
- 编程语言和计算机技术
- 数学、物理和自然科学
- 通用学习问题
- 不依赖特定企业内部资料的常识问题

判断规则：
1. 只要问题需要依赖企业内部资料，就选择 enterprise。
2. 如果问题明确属于通用知识，选择 general。
3. 如果无法确定，默认选择 enterprise，
   避免把企业专属问题交给通用模型猜测。

只允许输出以下两个单词之一：
enterprise
general

不要输出其他内容。
"""
            ),
            (
                "human",
                "用户问题：\n{question}"
            )
        ])

        return (
            router_prompt
            | self.model
            | StrOutputParser()
        )

    def classify_question(
        self,
        question: str,
        config=None
    ) -> str:
        """判断问题类型"""

        if not question or not question.strip():
            raise ValueError("用户问题不能为空")

        result = self.router_chain.invoke(
            {
                "question": question
            },
            config=config
        )

        result = result.strip().lower()

        # 清理可能出现的 Markdown 标记
        result = result.replace("`", "").strip()

        if "enterprise" in result:
            return "enterprise"

        if "general" in result:
            return "general"

        # 分类结果不符合要求时，保守地选择企业知识 Agent
        return "enterprise"

    def invoke(
        self,
        question: str,
        config=None
    ):
        """自动路由并执行对应 Agent"""

        route = self.classify_question(
            question,
            config=config
        )

        if route == "enterprise":
            return self.react_agent.invoke(
                question,
                config=config
            )

        return self.zero_shot_agent.invoke(
            question,
            config=config
        )

    def invoke_with_route(
        self,
        question: str,
        config=None
    ) -> dict:
        """执行任务并同时返回路由信息"""

        route = self.classify_question(
            question,
            config=config
        )

        if route == "enterprise":
            answer = self.react_agent.invoke(
                question,
                config=config
            )
        else:
            answer = self.zero_shot_agent.invoke(
                question,
                config=config
            )

        return {
            "route": route,
            "answer": answer
        }
