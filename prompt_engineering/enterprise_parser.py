# 自定义输出解析器：
# 企业项目往往有自己的业务要求。
#例如：
# 1. 必须包含指定字段。
# 2. 字段类型必须正确。
# 3. 引用的资料必须真实存在。
# 4. JSON 格式错误时，需要给出明确异常；
# 5. 必须能够与 LangChain LCEL 组合等等。

import json
import os
import re

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException

from answer_schema import EnterpriseAnswer
from qa_prompt import QA_OUTPUT_PROMPT


class EnterpriseAnswerParser(BaseOutputParser):
    """企业问答自定义输出解析器"""
    """提炼模型回答的消息  ->  自定义的回答标准模式 EnterpriseAnswer"""
    """BaseOutputParser 是统一的 Runnable 接口，我们的自定义解析器可以直接参与 LCEL😎"""
    def parse(self, text: str) -> EnterpriseAnswer:
        """
        将模型返回的消息解析为 EnterpriseAnswer
        """
        if not text or text.strip():
            raise OutputParserException("模型返回内容为空")

        # 1. 去除 Markdown Json 代码块
        text = text.strip()

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"\s*```$",
            "",
            text
        )

        # 2. 提取 JSON 内容
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or start >= end:
            raise OutputParserException(
                "模型返回内容中没有有效的 JSON 对象"
            )

        json_text = text[start:end + 1]

        # 3. JSON 格式解析
        try:
            data = json.loads(json_text)

        except json.JSONDecodeError as e:
            raise OutputParserException(
                f"JSON 格式解析失败：{e}"
            ) from e

        # 4. 检查根节点类型
        if not isinstance(data, dict):
            raise OutputParserException(
                "JSON 根节点必须是对象"
            )

        # 5. 使用 Pydantic 校验数据(就是看能不能把 data 塞进 answer_schema.py 中定义的 EnterpriseAnswer 类)
        try:
            result: EnterpriseAnswer = EnterpriseAnswer.model_validate(data)

        except Exception as e:
            raise OutputParserException(
                f"企业问答结果字段校验失败：{e}"
            ) from e

        return result

    def get_format_instructions(self) -> str:
        """获取输出格式约束"""
        """这是对外的接口，你不能让用户直接看到 QA_OUTPUT_PROMPT"""
        return QA_OUTPUT_PROMPT
        #return super().get_format_instructions()

    @property
    def _type(self) -> str:
        return "enterprise_answer_parser"