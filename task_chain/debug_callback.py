import logging
import json
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler



import logging
import json
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class DebugCallbackHandler(BaseCallbackHandler):
    """任务链调试与监控回调处理器"""

    def __init__(
        self,
        log_file: str = "logs/task_chain.log",
        log_payload: bool = False,
        max_preview_length: int = 500
    ):
        super().__init__()

        self.log_payload = log_payload
        self.max_preview_length = max_preview_length

        # 创建日志目录
        log_path = Path(log_file)
        log_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # 创建独立 Logger
        self.logger = logging.getLogger(
            f"task_chain_debug_{id(self)}"
        )

        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # 写入文件
        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8"
        )

        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s | "
                "%(levelname)s | "
                "%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _preview(self, data: Any) -> str:
        """安全地生成日志预览"""

        if not self.log_payload:
            return "<payload logging disabled>"

        try:
            content = json.dumps(
                data,
                ensure_ascii=False,
                default=str
            )
        except Exception:
            content = str(data)

        if len(content) > self.max_preview_length:
            content = (
                content[:self.max_preview_length]
                + "...(truncated)"
            )

        return content

    def on_chain_start(
        self,
        serialized,
        inputs,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs
    ):
        name = (
            serialized.get("name", "Runnable")
            if serialized else "Runnable"
        )

        self.logger.info(
            "[CHAIN START] name=%s run_id=%s input=%s",
            name,
            run_id,
            self._preview(inputs)
        )

    def on_chain_end(
        self,
        outputs,
        *,
        run_id,
        parent_run_id=None,
        **kwargs
    ):
        self.logger.info(
            "[CHAIN END] run_id=%s output=%s",
            run_id,
            self._preview(outputs)
        )

    def on_chain_error(
        self,
        error,
        *,
        run_id,
        parent_run_id=None,
        **kwargs
    ):
        self.logger.error(
            "[CHAIN ERROR] run_id=%s error=%s",
            run_id,
            error,
            exc_info=(
                type(error),
                error,
                error.__traceback__
            )
        )

    def on_llm_start(
        self,
        serialized,
        prompts,
        *,
        run_id,
        parent_run_id=None,
        **kwargs
    ):
        self.logger.info(
            "[LLM START] run_id=%s prompt_count=%d",
            run_id,
            len(prompts)
        )

    def on_chat_model_start(
        self,
        serialized,
        messages,
        *,
        run_id,
        parent_run_id=None,
        **kwargs
    ):
        self.logger.info(
            "[CHAT MODEL START] run_id=%s message_count=%d",
            run_id,
            len(messages)
        )

    def on_llm_end(
        self,
        response,
        *,
        run_id,
        parent_run_id=None,
        **kwargs
    ):
        self.logger.info(
            "[LLM END] run_id=%s",
            run_id
        )

    def on_llm_error(
        self,
        error,
        *,
        run_id,
        parent_run_id=None,
        **kwargs
    ):
        self.logger.error(
            "[LLM ERROR] run_id=%s error=%s",
            run_id,
            error,
            exc_info=(
                type(error),
                error,
                error.__traceback__
            )
        )

    def on_retriever_start(
        self,
        serialized,
        query,
        *,
        run_id,
        parent_run_id=None,
        **kwargs
    ):
        self.logger.info(
            "[RETRIEVER START] run_id=%s query=%s",
            run_id,
            self._preview(query)
        )

    def on_retriever_end(
        self,
        documents,
        *,
        run_id,
        parent_run_id=None,
        **kwargs
    ):
        self.logger.info(
            "[RETRIEVER END] run_id=%s document_count=%d",
            run_id,
            len(documents)
        )

    def on_retriever_error(
        self,
        error,
        *,
        run_id,
        parent_run_id=None,
        **kwargs
    ):
        self.logger.error(
            "[RETRIEVER ERROR] run_id=%s error=%s",
            run_id,
            error,
            exc_info=(
                type(error),
                error,
                error.__traceback__
            )
        )

    def close(self):
        """关闭日志处理器"""

        for handler in self.logger.handlers[:]:
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)
