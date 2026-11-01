# 自定义模型：
import builtins
from collections.abc import Callable
import os
import hashlib
import json
from typing import Any, Callable, List, Optional, Iterator, Sequence

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from openai import OpenAI
from pydantic import PrivateAttr

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.prompt_values import PromptValue

from common.func import generate_text_with_chat, generate_text_with_responses
from common.cache import InMemoryCache, FileCache, SimpleRedisCache

# ！！！：
# 在 LangChain 的基类底层使用了 Pydantic(v2 版本)，而 Pydantic 对 “类中声明的字段” 有严格管控：
# 1. 只有带类型注解的字段才会被 Pydantic 注册为 “模型字段”；
# 2. PrivateAttr 是绕过管控 1 的官方通道：它告诉 Pydantic "这个属性是内部使用的，不参与‘模型字段’体系"；
# 3. Pydantic 的管控有明确边界： 只管 “声明了的字段”，不管 “私有属性” 的类型安全


class CustomLangChainModel(BaseChatModel):
    """
    一个同时兼容 OpenAI 的 Responses API 和 Chat Completions API 的 LangChain ChatModel
    """

    # 注意：只有带类型注解的类属性才会成为 Pydantic 字段，
    # 否则通过构造参数传值会被静默忽略
    model_name: str = "deepseek-flash"
    use_responses_api: bool = True
    temperature: float = 0.7
    base_url: str = "https://api.deepseek.com"   # 修复：原为 api_deepseek.com（下划线应为点）
    max_tokens: int = 4896
    top_p: float = 0.5
    frequency_penalty: float = 0.0

    # OpenAI 客户端是运行时资源，不是模型配置字段：
    # 用 PrivateAttr 声明，避免 Pydantic 将其认为是模型字段，把实例上的赋值当成"未声明字段"而报错
    # （原代码 self.client = OpenAI(...) 在 Pydantic v2 下会抛
    #  ValueError: "CustomLangChainModel" object has no field "client"）
    _client: Optional[OpenAI] = PrivateAttr(default=None)   # 声明私有属性，允许在实例上修改它的值

    @property # Python 的内置装饰器，把一个方法伪装成属性来访问，直接写 对象.方法名 就能得到方法的返回值，无需添加()
    def client(self) -> OpenAI:
        """惰性创建 OpenAI 客户端（首次真正调用 API 时才创建）"""
        if self._client is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("环境变量 DEEPSEEK_API_KEY 未设置")
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    def _generate(self,
                  messages: List[BaseMessage],
                  stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any,
                  ) -> ChatResult:
        # 2. 将 LangChain 的 messages 转换为 OpenAI 格式
        system_content = None
        user_prompt_parts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_content = msg.content
            elif isinstance(msg, HumanMessage):
                user_prompt_parts.append(msg.content)

        user_prompt = user_prompt_parts[-1] if user_prompt_parts else ""

        # 3. 根据配置调用不同的 API（把 model_name 等字段真正传给 API，否则字段形同虚设）
        if self.use_responses_api:
            # 使用 Responses API
            content = generate_text_with_responses(
                client=self.client,
                instructions=system_content,
                input=user_prompt,
                model_name=self.model_name,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                top_p = self.top_p,
            )
        else:
            # 使用 Chat Completions API
            content = generate_text_with_chat(
                client=self.client,
                prompt=user_prompt,
                system_prompt=system_content, # type:ignore
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
            )

        # 兜底：API 层出错时函数会返回 None，避免 AIMessage(content=None) 导致下游解析器报错
        content = content or ""

        # 4. 包装成 LangChain 的 ChatResult
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """返回模型类型标识，用于日志记录"""
        return "deepseek-flash"


class DeepSeekChatModel(BaseChatModel):
    """自定义 DeepSeek Chat 模型"""
    """新调整： 普通调用 + 流式调用"""
    """新调整： 添加工具绑定方法，使模型可调用工具"""
    """三个阶段： 工具绑定到模型上(bind_tools)  ->  模型结合输入信息和参数，决定是否生成工具调用请求  ->  执行工具"""

    model_name: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.5
    frequency_penalty: float = 0.0

    # 是否启用 API 流式请求：
    streaming: bool = False

    _client: Optional[OpenAI] = PrivateAttr(default=None)

    # https://chatgpt.com/c/6ae3fac9-33f8-83ee-9af0-19682b009158
    def bind_tools(
            self, 
            tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool], 
            *, 
            tool_choice: str | None = None,
            **kwargs: Any,
        ) -> Runnable[PromptValue | str | Sequence[BaseMessage | List[str] | tuple[str, str | List[str | dict[str, Any]]] | str | dict[str, Any]], AIMessage]:
        """将不同形式的 Python 工具统一转换成 OpenAI 兼容的 Tool Schema，然后绑定到大语言模型上，返回一个支持工具调用的 RunnableBinding 对象
        :param tools: 支持四种工具输入形式：    
                    1. dict[str, Any]:  已经是 OpenAI 工具 schema 的字典；
                    2. type： 带 docstring 的 Python 类；
                    3. Callable[..., Any]： 普通函数；
                    4. BaseTool： LangChain 的 BaseTool 实例
        :param tool_choice: 控制模型如何选择工具， "auto", "required", "none", 具体工具名等
        :param kwargs: 允许用户传入其他模型参数
        """

        # 工具格式统一转换： 各种来源的工具统一转换成 OpenAI Tools 的 JSON Schema 格式
        formatted_tools = [
            convert_to_openai_tool(tool)
            for tool in tools
        ]

        # LangChain 生态中常用 "any" 表示 “必须调用某个工具”，
        # 而 OpenAI/部分兼容 API 要求写成 "required"
        if tool_choice == "any":
            tool_choice = "required"

        # 组织绑定参数： 字典解包语法， **kwargs： 允许用户传入其他模型参数
        bind_kwargs = {
            **kwargs,
            "tools": formatted_tools,   # tools 放在后面，后面的值会覆盖前面的值
        }

        if tool_choice is not None: # 如果用户指定了工具调用策略，也将其添加到绑定参数中
            bind_kwargs['tool_choice'] = tool_choice

        # bind()：一种参数绑定机制，将一些固定参数绑定到 Runnable 上，并返回一个 RunnableBinding 对象；
        # 如果某次调用传入了新的参数或参数值，会和已经绑定的参数进行合并
        return self.bind(**bind_kwargs)
        
        #return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)

    @property   
    def client(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    def _convert_message_to_dict(self, message: BaseMessage):
        """将 LangChain 消息转换为 API 所需的格式"""
        message_dict = {"content": message.content or ""}

        if isinstance(message, HumanMessage):
            message_dict['role'] = 'user'
        elif isinstance(message, AIMessage):
            message_dict['role'] = 'assistant'

            # 将之前的模型发起的工具调用请求再次传递给 API
            if message.tool_calls:
                message_dict['tool_calls'] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call['name'],
                            "arguments": json.dumps(call['args'], ensure_ascii=False,)
                        },
                    }
                    for call in message.tool_calls
                ]
        elif isinstance(message, SystemMessage):
            message_dict['role'] = 'system'
        elif isinstance(message, ToolMessage):
            message_dict['role'] = "tool"
            message_dict['tool_call_id'] = message.tool_call_id
            message_dict['content'] = message.content if isinstance(message.content, str) else json.dumps(message.content, ensure_ascii=False)
        else:
            message_dict['role'] = message.type

        return message_dict

    def _build_cache_key(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> str:
        api_messages = []

        for msg in messages:
            msg_dict = self._convert_message_to_dict(msg)
            api_messages.append(msg_dict)

        cache_key_data = {
            "messages": api_messages,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "stop": stop,
            "kwargs": kwargs,
        }

        serialized = json.dumps(cache_key_data, sort_keys=True, ensure_ascii=False, default=str,)

        return hashlib.md5(serialized.encode('utf-8')).hexdigest()


    def _generate(self,
                  messages: List[BaseMessage],
                  stop: Optional[List[str]] = None,
                  run_manager: Optional[Any] = None,
                  **kwargs: Any,
                ) -> ChatResult:
        """核心方法：普通调用，并一次性返回 ChatResult"""
        # 1. BaseMessage -> API 需要的 messages 的 dict 格式
        api_messages = []
        for msg in messages:
            msg_dict = self._convert_message_to_dict(msg)
            api_messages.append(msg_dict)

        request_params = {
            "model": self.model_name,
            "messages": api_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "stop": stop,
            "stream": False,
        }

        # 接收 bind_tools() 绑定的工具参数
        for key in ("tools", "tool_choice"):
            if key in kwargs:
                request_params[key] = kwargs[key]

        response = self.client.chat.completions.create(
            **request_params
        )

        choice = response.choices[0]
        api_message = choice.message

        content = api_message.content or ""
        parsed_tool_calls = []
        invalid_tool_calls = []

        for call in (api_message.tool_calls or []):
            name = call.function.name
            arguments = call.function.arguments
            call_id = call.id

            try:
                args = json.loads(arguments)

                if not isinstance(args, dict):
                    raise ValueError("工具参数必须是 JSON 对象")

                parsed_tool_calls.append({
                    "name": name,
                    "args": args,
                    "id": call_id,
                    "type": "tool_call",
                })

            except (json.JSONDecodeError, ValueError) as exc:
                invalid_tool_calls.append({
                    "name": name,
                    "args": arguments,
                    "id": call_id,
                    "error": str(exc),
                    "type": "invalid_tool_call",
                })

        ai_message = AIMessage(
            content=content,
            tool_calls=parsed_tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            response_metadata={
                "finish_reason": choice.finish_reason,
                "model_name": response.model,
            },
            usage_metadata={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else None,
        )

        generation = ChatGeneration(message=ai_message)

        return ChatResult(
            generations=[generation],
            llm_output={
                "model_name": response.model,
            },
        )

        

        # # 2. 调用 DeepSeek Chat API
        # # streaming = True 时，内部使用流式 API
        # # 但此方法仍只一次性返回完整的 ChatResult：
        # if self.streaming:
        #     stream = self.client.chat.completions.create(
        #         model=self.model_name,
        #         messages=api_messages,
        #         temperature=self.temperature,
        #         max_tokens=self.max_tokens,
        #         top_p=self.top_p,
        #         frequency_penalty=self.frequency_penalty,
        #         stop=stop,
        #         stream=True,
        #     )

        #     content_parts = []

        #     for chunk in stream:
        #         if not chunk.choices:
        #             continue

        #         token = chunk.choices[0].delta.content

        #         if token:
        #             content_parts.append(token)

        #     content = "".join(content_parts)

        # else:
        #     response = self.client.chat.completions.create(
        #         model=self.model_name,
        #         messages=api_messages,
        #         temperature=self.temperature,
        #         max_tokens=self.max_tokens,
        #         top_p=self.top_p,
        #         frequency_penalty=self.frequency_penalty,
        #         stop=stop,
        #         stream=False,
        #     )
        #     content = response.choices[0].message.content

        # # 3. 结果封装成 LangChain 的 ChatResult 形式
        # #content = response.choices[0].message.content
        # message_as_aimessage = AIMessage(content=content)
        # generation = ChatGeneration(message=message_as_aimessage)
        # return ChatResult(generations=[generation])


    def _stream(
            self,   
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[Any] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式调用： 逐块返回 ChatGenerationChunk"""

        # 关闭流式模式，退化为一次性返回结果
        if not self.streaming:
            result = self._generate(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs
            )

            yield ChatGenerationChunk(message=AIMessageChunk(content=result.generations[0].message.content))

            return 

        api_messages = []
        for msg in messages:
            msg_dict = self._convert_message_to_dict(msg)
            api_messages.append(msg_dict)

        # 一个生成器：
        response_stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=api_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            stop=stop,
            stream=True,    # 开启 API 流式模式，分批返回生成结果，而不是等完整答案生成后才返回
        )

        for chunk in response_stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            token = choice.delta.content

            if not token:
                continue

            generation_chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=token)
            )

            # 支持 LangChain 的回调机制：
            if run_manager is not None:
                run_manager.on_llm_new_token(token, chunk=generation_chunk)

            # 将 API 返回的数据，通过 yield 将每块结果交给 LangChain
            yield generation_chunk
        

    @property
    def _llm_type(self) -> str:
        """返回模型类型标识，用于日志记录"""
        return "deepseek-flash"


class DeepSeekChatModelWithMemoryCache(DeepSeekChatModel):
    """带有 Memory Cache 的 DeepSeek Chat Model，继承自 DeepSeekChatModel 类"""
    """新调整： 支持 普通调用 和 流式调用"""

    _cache: InMemoryCache = PrivateAttr(default_factory=InMemoryCache)


    # def _get_cached_content(
    #         self, 
    #         cache_key: str,
    # ) -> Optional[str]:
    #     """查询缓存，未命中返回 None"""

    #     cached_generations = self._cache.

    def _generate(self,
                  messages: List[BaseMessage],
                  stop: Optional[List[str]] = None,
                  run_manager: Optional[Any] = None,
                  **kwargs: Any,
                ) -> ChatResult:
        """ 普通调用，先查缓存，再决定是否请求 API"""

        # 1. BaseMessage -> API 需要的 messages 的 dict 格式
        # api_messages = []
        # for msg in messages:
        #     role = "user" if msg.type == "human" else msg.type
        #     api_messages.append({"role": role, "content": msg.content})

        # 2. 构造缓存 key (基于输入消息和关键参数):
        # cache_key_data: dict ，包含所有输入和参数；
        # json.dumps(cache_key_data, sort_keys=True) :  dict -> JSON 格式的纯文本字符串， sorted_keys=True 能够保证即使
        #                                               字典里的键定义顺序不同，最终生成的字符串也是相同的；
        # .encode('utf-8') : JSON 格式的纯文本字符串  ->   UTF-8 编码的字节流；
        # hashlib.md5() : 字节流 -> 128 位的二进制数值；
        # .hexdigest() : 128 位的二进制数值  ->  32个字符长的纯文本字符串
        # cache_key_data = {
        #     "messages": api_messages,
        #     "model": self.model_name, 
        #     "temperature": self.temperature,
        #     "max_tokens": self.max_tokens,
        #     "top_p": self.top_p,
        #     "frequency_penalty": self.frequency_penalty,
        # }
        # cache_key = hashlib.md5(json.dumps(cache_key_data, sort_keys=True).encode('utf-8')).hexdigest()

        cache_key = self._build_cache_key(messages=messages, stop=stop, kwargs=kwargs)

        # 3. 从缓存中获取
        cached_content = self._cache.get(cache_key)
        if cached_content:
            # 缓冲命中，组装并返回 LangChain 格式的结果
            print("✅️ [Memory 缓存命中] 从缓存中获取结果：")
            message_as_aimmessage = AIMessage(content=cached_content)
            generation = ChatGeneration(message=message_as_aimmessage)
            return ChatResult(generations=[generation])                 # 组装和提取，两个刚好相反的操作

        # 4. 缓存未命中，调用父类的 _generate() 方法请求 API
        print("❌️ [Memory 缓存未命中] 请求 DeepSeek API: ")
        result = super()._generate(messages=messages, stop=stop, run_manager=run_manager, **kwargs)

        # 5. 提取结果，并写入缓存中：
        content_to_cache = result.generations[0].message.content        # 组装和提取，是两个刚好相反的操作
        self._cache.set(cache_key, content_to_cache)

        # 6. 返回结果
        return result

    def _stream(self, 
                messages: List[BaseMessage], 
                stop: List[str] | None = None, 
                run_manager: Any | None = None, 
                **kwargs: Any
                ) -> Iterator[ChatGenerationChunk]:
        """流式调用，缓存命中则输出缓存，未命中则边输出边缓存"""

        # api_messages = []
        # for msg in messages:
        #     msg_dict = self._convert_message_to_dict(msg)
        #     api_messages.append(msg_dict)

        cache_key = self._build_cache_key(messages=messages, stop=stop, kwargs=kwargs)

        # 从缓存中获取：
        cached_content = self._cache.get(cache_key)
        if cached_content:
            # 缓存命中
            print("✅️ [Memory 缓存命中] 从缓存中获取结果：")

            # 缓存保存的是完整的文本内容，这里一次性返回完整内容
            # LangChain 仍会通过 stream() 的迭代接口交付结果
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=cached_content,
                )
            )

            return 

        # 缓存未命中，
        print("❌️ [Memory 缓存未命中] 请求 DeepSeek API: ")

        content_parts = []

        for chunk in super()._stream(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            kwargs=kwargs,
        ):
            # 存储单个块的内容，后续合并一起送入缓存：
            content = chunk.message.content
            content_parts.append(content)

            # 立即向调用方转发当前分块
            yield chunk

        # 只有流正常结束并被调用方完全消费后，才执行到这里：
        self._cache.set(cache_key, "".join(content_parts))



# https://chatgpt.com/c/6ae03ffa-81fc-83ec-875f-93e82c84938b
class DeepSeekChatModelWithFileCache(DeepSeekChatModel):
    """带有 File Cache 的 DeepSeek Chat Model ，继承自 DeepSeekChatModel """
    """新调整：支持普通调用 和 流式调用"""

    # 实例化 FileCache，指定保存路径为 cache/cache.json
    _cache: FileCache = PrivateAttr(default_factory=lambda: FileCache("cache/cache.json"))
        
    def _generate(self,
                  messages: List[BaseMessage],
                  stop: Optional[List[str]] = None,
                  run_manager: Optional[Any] = None,
                  **kwargs: Any,
                ) -> ChatResult:
        # 1. BaseMessage -> API 指定格式
        # api_messages = []
        # for msg in messages:
        #     msg_dict = self._convert_message_to_dict(msg)
        #     api_messages.append(msg_dict)

        # 2. 构造缓存 key (应当包含输入信息 + 必要的模型参数)
        # cache_key_data = {
        #     "messages": api_messages,
        #     "model": self.model_name,
        #     "temperature": self.temperature,
        #     "max_tokens": self.max_tokens,
        #     "top_p": self.top_p,
        #     "frequency_penalty": self.frequency_penalty
        # }
        # cache_key = hashlib.md5(json.dumps(cache_key_data, sort_keys=True).encode('utf-8')).hexdigest()
        cache_key = self._build_cache_key(messages, stop, kwargs=kwargs)

        # 3. 查缓存：
        cached_content = self._cache.get(cache_key)
        if cached_content:
            # 缓存命中
            print(f"✅️ [File 缓存命中] 直接读取本地文件: {self._cache.cache_file}")
            message_as_aimmessage = AIMessage(content=cached_content)
            generation = ChatGeneration(message=message_as_aimmessage)
            return ChatResult(generations=[generation])

        # 缓存未命中，发送 API
        print(" ❌️ [File 缓存未命中] 正在请求 DeepSeek API")
        result = super()._generate(messages=messages, stop=stop, run_manager=run_manager, **kwargs)
        content = result.generations[0].message.content

        # 4. 写入缓存，自动持久化到 Json 文件：
        self._cache.set(cache_key, content)

        # 5. 返回结果
        return result

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式调用： 命中缓存则返回缓存，未命中则边输出边写缓存"""

        cache_key = self._build_cache_key(messages, stop=stop, kwargs=kwargs)

        # 1. 查询文件缓存：
        cached_content = self._cache.get(cache_key)

        if cached_content is not None:
            # 缓存命中：
            print(f"✅️ [File 缓存命中] 直接读取本地文件: {self._cache.cache_file}")
            # 缓存中是完整文本，这里一次性返回
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=cached_content
                )
            )

            return

        # 2. 缓存未命中，调用父类的流式接口
        print(" ❌️ [File 缓存未命中] 正在请求 DeepSeek API")

        content_parts = []

        for chunk in super()._stream(messages=messages, stop=stop, run_manager=run_manager, kwargs=kwargs):
            content = chunk.message.content

            if isinstance(content, str):
                content_parts.append(content)

            # 不等待完整回答，先将分块交给调用方
            yield chunk

        # 3. 正常完成整个流式过程后，完整结果存入缓存
        self._cache.set(key=cache_key, value="".join(content_parts),)
        



class DeepSeekChatModelWithRedisCache(DeepSeekChatModel):
    """带有 Redis Cache 的 DeepSeek Chat Model，继承自 DeepSeekChatModel """
    """新调整： 普通调用 + 流式调用"""

    # 实例化 RedisCache
    _cache: SimpleRedisCache = PrivateAttr(default_factory=lambda: SimpleRedisCache(host='localhost', port=6379, db=0))

    def _generate(self,
                  messages: List[BaseMessage],
                  stop: Optional[List[str]] = None,
                  run_manager: Optional[Any] = None,
                  **kwargs: Any,
                ) -> ChatResult:
        """普通调用"""
        # # 1. BaseMessage  ->  API 指定格式
        # api_message = []
        # for msg in messages:
        #     role = "user" if msg.type == 'human' else 'system'
        #     content = msg.content
        #     api_message.append({'role' : role, 'content' : content})

        # # 2. 构造缓存 key
        # cached_key_data = {
        #     "messages" : api_message,  # json.dumps() 只认识 dict/list/string/int 等基本类型，因此不要把 messages 直接赋值给 cached_key_data
        #     "model" : self.model_name,
        #     "temperature" : self.temperature,
        #     "max_tokens" : self.max_tokens,
        #     "top_p" : self.top_p,
        #     "frequency_penalty" : self.frequency_penalty,
        # }
        cached_key = self._build_cache_key(messages=messages, stop=stop, kwargs=kwargs)

        # 3. 查看 Redis 缓存
        content_cached = self._cache.get(cached_key)
        if content_cached:
            # 缓存命中
            print("✅️ [Redis 缓存命中] 从 Redis 数据库中读取缓存：")
            message_as_aimmessage = AIMessage(content=content_cached)
            generation = ChatGeneration(message=message_as_aimmessage)
            return ChatResult(generations=[generation]) # generations 复数形式，也在提醒你记得将 generation 添加至 list

        # 缓存未命中，发送 API 请求：
        print("❌️ [Redis 缓存未命中] 发送 DeepSeek API 请求：")
        result = super()._generate(messages=messages, stop=stop, run_manager=run_manager, **kwargs)

        # 4. 生成的结果存入 Redis 缓存 :
        content = result.generations[0].message.content
        self._cache.set(key=cached_key, value=content)

        # 5. 返回结果
        return result

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式调用： 命中缓存则返回缓存，未命中则边输出边写缓存"""

        cache_key = self._build_cache_key(messages, stop=stop, kwargs=kwargs)

        # 1. 查询文件缓存：
        cached_content = self._cache.get(cache_key)

        if cached_content is not None:
            # 缓存命中：
            print("✅️ [Redis 缓存命中] 从 Redis 数据库中读取缓存：")
            # 缓存中是完整文本，这里一次性返回
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=cached_content
                )
            )

            return

        # 2. 缓存未命中，调用父类的流式接口
        print("❌️ [Redis 缓存未命中] 发送 DeepSeek API 请求：")

        content_parts = []

        for chunk in super()._stream(messages=messages, stop=stop, run_manager=run_manager, kwargs=kwargs):
            content = chunk.message.content

            if isinstance(content, str):
                content_parts.append(content)

            # 不等待完整回答，先将分块交给调用方
            yield chunk

        # 3. 正常完成整个流式过程后，完整结果存入缓存
        self._cache.set(key=cache_key, value="".join(content_parts),)