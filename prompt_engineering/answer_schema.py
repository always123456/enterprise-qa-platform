# 定义输出数据的结构：
# 企业问答系统的输出不仅仅是一段文本，后续可能需要：
#   - 前端展示答案。
#   - 展示答案来源。
#   - 判断是否回答成功。
#   - 判断是否需要人工介入。
#   - 根据追问建议继续对话。
#   - 记录答案质量，进行任务评估。

# 因此要对模型的输出作一定的处理。

from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class Citation(BaseModel):
    """答案引用来源"""
    model_config = ConfigDict(extra='forbid')

    source_id: str = Field(
        description="参考资料的唯一编号",
    )

    quote: str = Field(
        description="支持答案的原文片段",
    )


# 对应于 qa_prompt.py 中的 QA_OUTPUT_PROMPT 提示词
class EnterpriseAnswer(BaseModel):
    """企业问答标准输出结构类"""
    """可能会有疑惑🤔：qa_prompt.py 中已经有了输出模版提示词，为什么这里还要创建一个类"""
    """注意：输出模版提示词只是告诉大模型，它应该输出什么结构的信息，比如 status/answer/citations/...等等"""
    """而这些信息我们需要保存下来，以便后续使用（例如：前端展示给用户），因此创建这样一个类来保存"""
    """再一个，使用 Pydantic 来定义类的各种属性，起到校验数据的作用！！！💡"""
    model_config = ConfigDict(extra='forbid')

    # 下面使用了 Pydantic 的几个重要特性：
    # Literal/Field/...等等
    # 字段不符合，会被 Pydantic 拒绝
    status: Literal[
        "answerd",
        "insufficient_information",
        "conflict",
    ] = Field(
        description="回答状态",
    )

    answer: str = Field(
        min_length=1,
        description="最终回答内容"
    )

    citations: list[Citation] = Field(
        default_factory=list,
        description="答案引用来源",
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="模型自评的回答可信度😎",
    )

    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="建议用户进一步询问的问题🤔",
    )