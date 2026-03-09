"""
ZhipuAI OpenAI 兼容适配器

将 ZhipuAI SDK 的调用方式适配为 OpenAI 兼容的接口，使其可以与 LangChain 无缝集成。
"""

from typing import (
    Optional,
    Any,
    List,
    Dict,
    Iterator,
    Type,
    Union,
    Callable,
    Sequence,
    ClassVar,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    AIMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel
import os
import json
import re


class ZhipuAIAdapter(BaseChatModel):
    """
    ZhipuAI 适配器类

    将 ZhipuAI SDK 的调用方式适配为 LangChain 兼容的接口。
    使用方式与 OpenAI 的 ChatOpenAI 完全一致。

    支持超时后自动切换到备用模型（fallback models）。
    """

    model: str = "chatglm3-6b-1001"
    temperature: float = 0.7
    api_key: Optional[str] = None
    zhipu_client: Optional[Any] = None
    timeout: int = 120  # Default timeout in seconds
    max_retries: int = 3  # Maximum number of retries for timeout errors
    retry_delay: float = 2.0  # Delay between retries in seconds
    fallback_models: Optional[List[str]] = None  # 备用模型列表，超时后依次切换
    auto_fallback: bool = True  # 是否启用自动备用模型切换
    progress_callback: Optional[Callable] = None  # 进度回调函数，用于报告思考过程
    enable_thinking_prompt: bool = True  # 是否自动注入思考引导指令
    compact_thinking_mode: bool = True  # 是否使用紧凑模式的思考指令

    # 默认的备用模型映射（主模型 -> 备用模型）
    DEFAULT_FALLBACK_MAP: ClassVar[Dict[str, List[str]]] = {
        "glm-4.5": ["glm-4.5-air", "glm-4-plus"],
        "glm-4.5-air": ["glm-4-plus"],
        "glm-4-plus": ["glm-4"],
        "glm-4": ["glm-4-flash"],
    }

    def __init__(
        self,
        model: str = "chatglm3-6b-1001",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        fallback_models: Optional[List[str]] = None,
        auto_fallback: bool = True,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ):
        """
        初始化 ZhipuAI 适配器

        Args:
            model: 模型名称，默认为 "chatglm3-6b-1001"
            temperature: 温度参数，默认为 0.7
            api_key: API 密钥，如果不提供则从环境变量 ZHIPU_API_KEY 读取
            timeout: 请求超时时间（秒），默认 120
            max_retries: 超时错误的最大重试次数，默认 3
            retry_delay: 重试之间的延迟时间（秒），默认 2.0
            fallback_models: 备用模型列表，超时后依次切换。如果为 None 且 auto_fallback=True，
                           则根据 DEFAULT_FALLBACK_MAP 自动选择备用模型
            auto_fallback: 是否启用自动备用模型切换，默认 True
            progress_callback: 进度回调函数，用于报告LLM思考过程
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)

        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.auto_fallback = auto_fallback
        self.progress_callback = progress_callback

        # 设置备用模型列表
        if fallback_models is not None:
            self.fallback_models = fallback_models
        elif auto_fallback and model in self.DEFAULT_FALLBACK_MAP:
            self.fallback_models = self.DEFAULT_FALLBACK_MAP[model]
        else:
            self.fallback_models = []

        # 获取 API Key
        if api_key is None:
            api_key = os.getenv("ZHIPU_API_KEY")

        if not api_key:
            raise ValueError("ZHIPU_API_KEY 未设置，请设置环境变量或传入 api_key 参数")

        self.api_key = api_key

        # 初始化 ZhipuAI 客户端
        try:
            from zhipuai import ZhipuAI

            self.zhipu_client = ZhipuAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("请安装 zhipuai 库: pip install zhipuai")
        except Exception as e:
            raise RuntimeError(f"初始化 ZhipuAI 客户端失败: {e}")

    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型标识"""
        return "zhipu"

    def _convert_messages_to_zhipu_format(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, str]]:
        """
        将 LangChain 的 Message 对象转换为 ZhipuAI 格式

        Args:
            messages: LangChain 的 Message 列表

        Returns:
            ZhipuAI 格式的消息列表
        """
        zhipu_messages = []

        for message in messages:
            # 获取消息内容，确保是字符串类型
            if isinstance(message, SystemMessage):
                content = message.content
                if content is None:
                    content = ""
                elif not isinstance(content, str):
                    content = str(content)
                # 跳过空的 system 消息
                if not content.strip():
                    continue
                zhipu_messages.append({"role": "system", "content": content})
            elif isinstance(message, HumanMessage):
                content = message.content
                if content is None:
                    content = ""
                elif not isinstance(content, str):
                    content = str(content)
                # 跳过空的 user 消息
                if not content.strip():
                    continue
                zhipu_messages.append({"role": "user", "content": content})
            elif isinstance(message, AIMessage):
                content = message.content
                if content is None:
                    content = ""
                elif not isinstance(content, str):
                    content = str(content)
                # 跳过空的 assistant 消息
                if not content.strip():
                    continue
                zhipu_messages.append({"role": "assistant", "content": content})
            else:
                # 对于其他类型的消息，尝试获取内容
                content = getattr(message, "content", None)
                if content is None:
                    content = str(message)
                elif not isinstance(content, str):
                    content = str(content)
                # 跳过空消息
                if not content.strip():
                    continue
                zhipu_messages.append({"role": "user", "content": content})

        # 确保至少有一条消息
        if not zhipu_messages:
            raise ValueError("消息列表为空或所有消息内容为空")

        # 确保最后一条消息是 user 消息（ZhipuAI 要求）
        if zhipu_messages and zhipu_messages[-1]["role"] not in ["user", "assistant"]:
            raise ValueError("最后一条消息必须是 user 或 assistant 消息")

        return zhipu_messages

    def _is_timeout_error(self, error: Exception) -> bool:
        """
        判断是否为超时错误

        Args:
            error: 异常对象

        Returns:
            是否为超时错误
        """
        error_type = type(error).__name__
        error_msg = str(error).lower()

        # 检查常见的超时错误类型
        timeout_types = [
            "timeout",
            "timedout",
            "apitimeouterror",
            "timeouterror",
            "readtimeout",
            "connecttimeout",
        ]
        timeout_keywords = ["timeout", "timed out", "time out", "请求超时"]

        # 检查错误类型名
        if any(tt in error_type.lower() for tt in timeout_types):
            return True

        # 检查错误消息
        if any(tk in error_msg for tk in timeout_keywords):
            return True

        return False

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        生成回复（核心方法，带超时重试和备用模型切换机制）

        Args:
            messages: 消息列表
            stop: 停止词列表
            run_manager: 回调管理器
            **kwargs: 其他参数（包括 tools, tool_choice 等）

        Returns:
            ChatResult 对象
        """
        import time
        import platform
        import signal

        # 🔥 报告LLM调用前的思考过程
        if self.progress_callback:
            try:
                # 提取用户消息
                user_msg = ""
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        user_msg = str(msg.content)[:200]
                        break

                self.progress_callback(
                    event_type="llm_thinking",
                    message=f"🤔 开始思考: {user_msg}",
                    details={
                        "model": self.model,
                        "message_count": len(messages),
                        "has_tools": "tools" in kwargs,
                    },
                )
            except Exception as e:
                print(f"[ZhipuAIAdapter] Error reporting pre-LLM thinking: {e}")
            except Exception as e:
                print(f"[ZhipuAIAdapter] Error reporting pre-LLM thinking: {e}")

        # 转换消息格式
        try:
            zhipu_messages = self._convert_messages_to_zhipu_format(messages)
        except Exception as e:
            raise RuntimeError(f"消息格式转换失败: {e}")

        # 验证消息格式
        if not zhipu_messages:
            raise ValueError("转换后的消息列表为空")

        # 准备请求参数
        request_params = {
            "messages": zhipu_messages,
            "temperature": self.temperature,
        }

        # 添加其他参数
        if stop:
            request_params["stop"] = stop

        request_params.update(kwargs)

        # 超时处理函数
        def timeout_handler(signum, frame):
            raise TimeoutError(f"ZhipuAI API 请求超时 ({self.timeout}秒)")

        # 构建要尝试的模型列表：主模型 + 备用模型
        models_to_try = [self.model]
        if self.fallback_models:
            models_to_try.extend(self.fallback_models)

        last_error = None

        # 遍历所有模型（主模型 + 备用模型）
        for model_idx, current_model in enumerate(models_to_try):
            is_fallback = model_idx > 0

            if is_fallback:
                print(f"🔄 主模型超时，切换到备用模型: {current_model}")

            # 对每个模型进行重试
            for attempt in range(self.max_retries):
                try:
                    # 设置当前模型
                    request_params["model"] = current_model

                    # 调用 ZhipuAI SDK
                    # Windows 和非主线程不支持 signal.alarm，直接调用（依赖 SDK 的超时）
                    import threading

                    use_signal_timeout = (
                        platform.system() != "Windows"
                        and threading.current_thread() is threading.main_thread()
                    )

                    if use_signal_timeout:
                        # 设置超时
                        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(self.timeout)

                    response = self.zhipu_client.chat.completions.create(
                        **request_params
                    )

                    if use_signal_timeout:
                        # 取消超时
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)

                    # 提取回复内容
                    if hasattr(response, "choices") and len(response.choices) > 0:
                        choice = response.choices[0]
                        message = choice.message
                        content = message.content or ""

                        # 处理工具调用
                        tool_calls = []
                        additional_kwargs = {}

                        # 检查是否有工具调用
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            for tc in message.tool_calls:
                                tool_calls.append(
                                    {
                                        "name": tc.function.name
                                        if hasattr(tc.function, "name")
                                        else tc.get("function", {}).get("name"),
                                        "args": json.loads(tc.function.arguments)
                                        if hasattr(tc.function, "arguments")
                                        else json.loads(
                                            tc.get("function", {}).get(
                                                "arguments", "{}"
                                            )
                                        ),
                                        "id": tc.id
                                        if hasattr(tc, "id")
                                        else tc.get("id"),
                                    }
                                )
                            additional_kwargs["tool_calls"] = [
                                {
                                    "id": tc.id if hasattr(tc, "id") else tc.get("id"),
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name
                                        if hasattr(tc.function, "name")
                                        else tc.get("function", {}).get("name"),
                                        "arguments": tc.function.arguments
                                        if hasattr(tc.function, "arguments")
                                        else tc.get("function", {}).get(
                                            "arguments", "{}"
                                        ),
                                    },
                                }
                                for tc in message.tool_calls
                            ]
                        # 检查 legacy function_call
                        elif (
                            hasattr(message, "function_call") and message.function_call
                        ):
                            additional_kwargs["function_call"] = {
                                "name": message.function_call.name
                                if hasattr(message.function_call, "name")
                                else message.function_call.get("name"),
                                "arguments": message.function_call.arguments
                                if hasattr(message.function_call, "arguments")
                                else message.function_call.get("arguments", "{}"),
                            }

                        # 创建 AIMessage
                        ai_message = AIMessage(
                            content=content,
                            tool_calls=tool_calls,
                            additional_kwargs=additional_kwargs,
                        )

                        # 报告LLM响应和工具调用
                        if self.progress_callback:
                            try:
                                # 如果有工具调用，报告工具调用
                                if tool_calls:
                                    for tc in tool_calls:
                                        tool_name = tc.get("name", "unknown")
                                        self.progress_callback(
                                            event_type="llm_thinking",
                                            message=f"[Tool Call] Decided to call: {tool_name}",
                                            details={
                                                "tool_name": tool_name,
                                                "tool_args": tc.get("args", {}),
                                            },
                                        )
                                # 否则报告响应内容
                                elif content:
                                    content_preview = content[:200]
                                    self.progress_callback(
                                        event_type="llm_thinking",
                                        message=f"[LLM Response] {content_preview}",
                                        details={
                                            "response_length": len(content),
                                            "model": current_model,
                                        },
                                    )
                            except Exception as e:
                                print(
                                    f"[ZhipuAIAdapter] Error reporting post-LLM thinking: {e}"
                                )

                    elif hasattr(response, "data") and hasattr(
                        response.data, "choices"
                    ):
                        content = response.data.choices[0].message.content
                        ai_message = AIMessage(content=content or "")
                    else:
                        # 尝试直接获取 content
                        content = getattr(response, "content", str(response))
                        ai_message = AIMessage(content=content)

                    # 如果是备用模型成功，记录信息
                    if is_fallback:
                        print(f"✅ 备用模型 {current_model} 调用成功")

                    # 创建 ChatGeneration 对象
                    generation = ChatGeneration(
                        message=ai_message,
                        generation_info={
                            "model": current_model,
                            "temperature": self.temperature,
                            "is_fallback": is_fallback,
                        },
                    )

                    # 创建 ChatResult 对象
                    return ChatResult(generations=[generation])

                except Exception as e:
                    last_error = e
                    error_type = type(e).__name__
                    error_msg = str(e)

                    # 判断是否为超时错误
                    is_timeout = self._is_timeout_error(e)

                    if is_timeout:
                        # 如果是超时错误且还有重试次数，则进行重试
                        if attempt < self.max_retries - 1:
                            retry_num = attempt + 2  # 显示为第几次尝试（从2开始）
                            # 指数退避：延迟时间随重试次数增加
                            delay = self.retry_delay * (2**attempt)
                            print(
                                f"⚠️ ZhipuAI 超时 ({current_model})，正在重试 ({retry_num}/{self.max_retries})，等待 {delay:.1f} 秒..."
                            )
                            time.sleep(delay)
                            continue
                        else:
                            # 当前模型重试次数用尽，打印提示并尝试下一个模型
                            print(f"⚠️ 模型 {current_model} 超时重试次数用尽")
                            # 如果还有备用模型，继续尝试下一个
                            if model_idx < len(models_to_try) - 1:
                                print(f"   → 准备切换到下一个备用模型...")
                                break  # 跳出重试循环，进入下一个模型
                            else:
                                # 没有更多备用模型，退出
                                break
                    else:
                        # 非超时错误，直接跳出重试（对于严重错误不重试）
                        break

            # 检查是否成功（非超时错误也会跳出外层循环）
            if last_error is None or not self._is_timeout_error(last_error):
                break

        # 所有模型和重试都失败，提供详细错误信息
        error_type = type(last_error).__name__
        error_msg = str(last_error)

        # 详细诊断错误原因
        print(f"❌ ZhipuAI 调用失败详情:")
        print(f"   - 错误类型: {error_type}")
        print(f"   - 错误消息: {error_msg}")
        print(f"   - 尝试的模型: {models_to_try}")
        print(f"   - 消息数量: {len(zhipu_messages)}")

        if "messages" in error_msg.lower() or "参数非法" in error_msg:
            # 打印消息格式以便调试
            print(f"   - 可能是消息格式问题")
            for i, msg in enumerate(zhipu_messages):
                content_preview = str(msg.get("content", ""))[:100]
                print(
                    f"     消息 {i + 1}: role={msg.get('role')}, content={content_preview}..."
                )
        elif self._is_timeout_error(last_error):
            print(f"   - 网络超时或API响应超时（已尝试所有备用模型）")
        elif "api" in error_msg.lower() or "key" in error_msg.lower():
            print(f"   - 可能是API密钥或权限问题")
        elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
            print(f"   - 可能是请求频率限制")

        raise RuntimeError(f"ZhipuAI 调用失败 ({error_type}): {last_error}")

    def _report_thinking_chunk(
        self, content: str, phase: str, accumulated_length: int = 0
    ):
        """报告思维链片段（内部辅助方法）"""
        if not self.progress_callback:
            return

        try:
            # 只报告有意义的内容片段
            if len(content.strip()) > 20:
                self.progress_callback(
                    event_type="llm_streaming",
                    message=f"💭 {content[:150]}",
                    details={
                        "phase": phase,
                        "accumulated_length": accumulated_length,
                    },
                )
        except Exception as e:
            print(f"[ZhipuAIAdapter] Error reporting thinking chunk: {e}")

    def invoke_with_streaming_thinking(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> AIMessage:
        """
        带有流式思维链报告的 invoke 方法

        这个方法在调用 LLM 时会实时报告思维过程，即使是底层使用非流式 API。
        它会将完整的响应分成多个思维片段进行报告，让前端能够展示思维链。

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            AIMessage 对象
        """
        import time

        # 报告准备开始思考
        if self.progress_callback:
            try:
                user_msg = ""
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        user_msg = str(msg.content)[:200]
                        break

                self.progress_callback(
                    event_type="llm_thinking",
                    message=f"🤔 开始思考: {user_msg[:100]}",
                    details={
                        "model": self.model,
                        "phase": "thinking_start",
                        "message_count": len(messages),
                    },
                )
            except Exception as e:
                print(f"[ZhipuAIAdapter] Error reporting thinking start: {e}")

        # 调用底层的 generate 方法
        start_time = time.time()
        result = self._generate(messages, **kwargs)

        # 提取响应内容
        if result.generations:
            ai_message = result.generations[0].message
            content = ai_message.content if hasattr(ai_message, "content") else ""

            # 如果有内容，模拟流式报告思维过程
            if content and self.progress_callback:
                try:
                    # 将内容分成思维片段（按句子或段落）
                    thinking_chunks = self._split_into_thinking_chunks(content)

                    for i, chunk in enumerate(thinking_chunks):
                        elapsed = time.time() - start_time
                        self.progress_callback(
                            event_type="llm_streaming",
                            message=f"💭 {chunk}",
                            details={
                                "phase": "thinking_progress",
                                "chunk_number": i + 1,
                                "total_chunks": len(thinking_chunks),
                                "elapsed_seconds": round(elapsed, 2),
                            },
                        )
                        time.sleep(0.05)  # 模拟流式输出的延迟

                    # 报告思考完成
                    self.progress_callback(
                        event_type="llm_thinking",
                        message=f"✅ 思考完成 (共 {len(thinking_chunks)} 个思维片段)",
                        details={
                            "phase": "thinking_complete",
                            "total_length": len(content),
                            "total_chunks": len(thinking_chunks),
                            "total_time": round(time.time() - start_time, 2),
                        },
                    )
                except Exception as e:
                    print(f"[ZhipuAIAdapter] Error reporting thinking process: {e}")

            return ai_message

        # 如果没有生成结果，返回空消息
        return AIMessage(content="")

    def _split_into_thinking_chunks(self, content: str) -> List[str]:
        """
        将内容分割成思维链片段

        Args:
            content: 完整内容

        Returns:
            思维片段列表
        """
        chunks = []

        # 按段落分割
        paragraphs = content.split("\n\n")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果段落太长，按句子分割
            if len(para) > 200:
                sentences = re.split(r"[。！？.!?]", para)
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) > 10:  # 忽略太短的片段
                        chunks.append(sent)
            else:
                chunks.append(para)

        # 确保至少有一个片段
        if not chunks and content.strip():
            chunks = [content.strip()]

        return chunks

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        流式生成回复（支持实时思考链报告）

        Args:
            messages: 消息列表
            stop: 停止词列表
            run_manager: 回调管理器
            **kwargs: 其他参数

        Yields:
            ChatGenerationChunk 对象
        """
        import time

        # 报告准备调用 LLM
        if self.progress_callback:
            try:
                user_msg = ""
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        user_msg = str(msg.content)[:200]
                        break

                self.progress_callback(
                    event_type="llm_thinking",
                    message=f"🤔 开始思考...",
                    details={
                        "model": self.model,
                        "phase": "streaming_start",
                        "user_message": user_msg[:100],
                    },
                )
            except Exception as e:
                print(f"[ZhipuAIAdapter] Error reporting streaming start: {e}")

        # 转换消息格式
        zhipu_messages = self._convert_messages_to_zhipu_format(messages)

        # 准备请求参数
        request_params = {
            "model": self.model,
            "messages": zhipu_messages,
            "temperature": self.temperature,
            "stream": True,  # 启用流式输出
        }

        if stop:
            request_params["stop"] = stop

        request_params.update(kwargs)

        accumulated_content = ""
        chunk_count = 0

        try:
            # 调用 ZhipuAI SDK（流式）
            response = self.zhipu_client.chat.completions.create(**request_params)

            # 处理流式响应
            if hasattr(response, "__iter__"):
                for chunk in response:
                    if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            chunk_content = delta.content
                            accumulated_content += chunk_content
                            chunk_count += 1

                            # 每 5 个 chunk 或每 50 字符报告一次思维过程
                            if (
                                self.progress_callback
                                and chunk_count % 5 == 0
                                and len(accumulated_content) % 50 < 10
                            ):
                                try:
                                    # 截取最近的思维片段
                                    recent_thinking = accumulated_content[-100:]
                                    self.progress_callback(
                                        event_type="llm_streaming",
                                        message=f"💭 {recent_thinking}",
                                        details={
                                            "chunk_count": chunk_count,
                                            "total_length": len(accumulated_content),
                                            "phase": "streaming_progress",
                                        },
                                    )
                                except Exception as e:
                                    print(
                                        f"[ZhipuAIAdapter] Error reporting streaming progress: {e}"
                                    )

                            yield ChatGenerationChunk(
                                message=AIMessageChunk(content=chunk_content),
                                generation_info={"chunk": True},
                            )

                # 报告流式输出完成
                if self.progress_callback and accumulated_content:
                    try:
                        self.progress_callback(
                            event_type="llm_thinking",
                            message=f"✅ 思考完成: {accumulated_content[:200]}...",
                            details={
                                "total_length": len(accumulated_content),
                                "chunk_count": chunk_count,
                                "phase": "streaming_complete",
                            },
                        )
                    except Exception as e:
                        print(
                            f"[ZhipuAIAdapter] Error reporting streaming complete: {e}"
                        )
            else:
                # 如果不是流式响应，回退到普通生成
                result = self._generate(messages, stop, run_manager, **kwargs)
                # 将 ChatGeneration 转换为 ChatGenerationChunk
                if result.generations:
                    gen = result.generations[0]
                    # 将 AIMessage 转换为 AIMessageChunk
                    msg = gen.message
                    if isinstance(msg, AIMessage):
                        chunk_msg = AIMessageChunk(
                            content=msg.content,
                            additional_kwargs=msg.additional_kwargs,
                        )
                    else:
                        chunk_msg = AIMessageChunk(content=str(msg.content))

                    yield ChatGenerationChunk(
                        message=chunk_msg,
                        generation_info=gen.generation_info or {},
                    )

        except Exception as e:
            raise RuntimeError(f"ZhipuAI 流式调用失败: {e}")

    def with_structured_output(
        self, schema: Union[Type[BaseModel], Dict[str, Any], Type], **kwargs: Any
    ) -> Any:
        """
        创建支持结构化输出的 LLM 包装器

        Args:
            schema: Pydantic 模型类、字典或类型
            **kwargs: 其他参数

        Returns:
            支持结构化输出的 LLM 包装器
        """
        from langchain_core.runnables import RunnableLambda

        # 创建输出解析器
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # Pydantic 模型
            output_parser = PydanticOutputParser(pydantic_object=schema)
            schema_json = schema.model_json_schema()
        elif isinstance(schema, dict):
            # 字典格式的 schema
            output_parser = JsonOutputParser()
            schema_json = schema
        else:
            # 其他类型，使用 JsonOutputParser
            output_parser = JsonOutputParser()
            schema_json = {}

        # 创建包装器函数
        def structured_invoke(messages: List[BaseMessage]) -> Any:
            """结构化调用的包装函数"""
            # 如果消息是字符串，转换为 HumanMessage
            if isinstance(messages, str):
                messages = [HumanMessage(content=messages)]
            elif not isinstance(messages, list):
                messages = [messages]

            # 获取 JSON schema 字符串
            schema_str = (
                json.dumps(schema_json, ensure_ascii=False, indent=2)
                if schema_json
                else ""
            )

            # 修改最后一条消息，添加 JSON 格式要求
            modified_messages = messages.copy()
            if modified_messages:
                last_msg = modified_messages[-1]
                if isinstance(last_msg, HumanMessage):
                    # 在提示词末尾添加 JSON 格式要求
                    original_content = last_msg.content
                    json_instruction = f"""

请严格按照以下 JSON 格式返回结果，不要包含任何其他文本或说明：

{schema_str}

重要：只返回 JSON 对象，不要包含 markdown 代码块标记（如 ```json）或其他文本。"""
                    modified_messages[-1] = HumanMessage(
                        content=original_content + json_instruction
                    )

            # 调用原始 LLM
            response = self.invoke(modified_messages)
            response_content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # 解析 JSON
            try:
                # 尝试直接解析
                parsed = output_parser.parse(response_content)
                return parsed
            except Exception as e:
                # 如果解析失败，尝试提取 JSON
                try:
                    # 尝试提取 JSON 代码块
                    json_match = re.search(
                        r"```(?:json)?\s*(\{.*?\})\s*```", response_content, re.DOTALL
                    )
                    if json_match:
                        parsed = output_parser.parse(json_match.group(1))
                        return parsed

                    # 尝试提取第一个 JSON 对象
                    json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
                    if json_match:
                        parsed = output_parser.parse(json_match.group(0))
                        return parsed

                    # 如果都失败了，尝试直接解析整个响应
                    parsed = json.loads(response_content)
                    if isinstance(schema, type) and issubclass(schema, BaseModel):
                        return schema(**parsed)
                    return parsed
                except Exception as e2:
                    raise ValueError(
                        f"无法解析结构化输出: {e2}. 原始响应: {response_content[:200]}"
                    )

        # 创建可调用对象（兼容 LangChain 接口）
        class StructuredOutputWrapper:
            """结构化输出包装器"""

            def __init__(self, invoke_func, base_llm):
                self._invoke = invoke_func
                self._base_llm = base_llm
                # 保持原始 LLM 的属性
                self.model = base_llm.model
                self.temperature = base_llm.temperature

            def invoke(self, messages: List[BaseMessage], **kwargs) -> Any:
                return self._invoke(messages)

            def __call__(self, messages: List[BaseMessage], **kwargs) -> Any:
                return self._invoke(messages)

            def generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
                """生成方法（兼容 LangChain）"""
                result = self._invoke(messages)
                # 将结果包装为 ChatResult
                if isinstance(result, BaseModel):
                    content = result.model_dump_json()
                elif isinstance(result, dict):
                    content = json.dumps(result, ensure_ascii=False)
                else:
                    content = str(result)

                generation = ChatGeneration(
                    message=AIMessage(content=content),
                    generation_info={"structured": True},
                )
                return ChatResult(generations=[generation])

        return StructuredOutputWrapper(structured_invoke, self)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[str, Dict[str, Any], bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """
        将工具绑定到模型

        类似于 OpenAI 的 bind_tools 方法，将工具定义绑定到模型，
        使模型能够决定何时以及如何调用这些工具。

        Args:
            tools: 工具列表，支持多种格式：
                - Dict: OpenAI 工具格式的字典
                - Type: Pydantic BaseModel 类
                - Callable: Python 函数（需要有类型注解）
                - BaseTool: LangChain 工具实例
            tool_choice: 工具选择策略：
                - "auto": 自动选择（默认）
                - "none": 不使用工具
                - "required" 或 "any": 必须使用工具
                - 特定工具名称: 强制使用该工具
                - Dict: OpenAI 格式的 tool_choice
            **kwargs: 其他传递给 bind 的参数

        Returns:
            绑定工具后的 Runnable

        Examples:
            >>> from pydantic import BaseModel, Field
            >>> from agent.utils.zhipu_adapter import ZhipuAIAdapter
            >>>
            >>> class GetWeather(BaseModel):
            ...     '''获取指定城市的天气'''
            ...     city: str = Field(description="城市名称")
            >>>
            >>> llm = ZhipuAIAdapter(model="glm-4")
            >>> llm_with_tools = llm.bind_tools([GetWeather])
            >>> response = llm_with_tools.invoke("北京今天天气怎么样？")
            >>> print(response.tool_calls)
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        # 将所有工具转换为 OpenAI 格式
        formatted_tools = []
        for tool in tools:
            try:
                formatted_tools.append(convert_to_openai_tool(tool))
            except Exception as e:
                raise ValueError(f"无法转换工具 {tool}: {e}")

        # 提取工具名称
        tool_names = []
        for tool in formatted_tools:
            if "function" in tool:
                tool_names.append(tool["function"]["name"])
            elif "name" in tool:
                tool_names.append(tool["name"])

        # 处理 tool_choice 参数
        if tool_choice:
            if isinstance(tool_choice, str):
                # 如果是工具名称，转换为正确格式
                if tool_choice in tool_names:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                elif tool_choice in ("required", "any"):
                    # 'any' 不是原生支持的，转换为 'required'
                    tool_choice = "required"
                # "auto" 和 "none" 直接使用
            elif isinstance(tool_choice, bool):
                tool_choice = "required" if tool_choice else None
            elif isinstance(tool_choice, dict):
                # 已经是正确格式，直接使用
                pass
            else:
                raise ValueError(
                    f"无法识别的 tool_choice 类型: {type(tool_choice)}. "
                    "期望 str, bool 或 dict"
                )

            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        # 使用父类的 bind 方法绑定工具
        # 这会创建一个 RunnableBinding，在调用时自动注入 tools 参数
        return super().bind(tools=formatted_tools, **kwargs)


def create_zhipu_chat_model(
    model: str = "chatglm3-6b-1001",
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> ZhipuAIAdapter:
    """
    创建 ZhipuAI 聊天模型实例（兼容 OpenAI 接口）

    Args:
        model: 模型名称，默认为 "chatglm3-6b-1001"
        temperature: 温度参数，默认为 0.7
        api_key: API 密钥，如果不提供则从环境变量 ZHIPU_API_KEY 读取

    Returns:
        ZhipuAIAdapter 实例，可以像 ChatOpenAI 一样使用

    Examples:
        >>> from agent.utils.zhipu_adapter import create_zhipu_chat_model
        >>> from langchain_core.messages import HumanMessage, SystemMessage
        >>>
        >>> llm = create_zhipu_chat_model(model="chatglm3-6b-1001", temperature=0.7)
        >>>
        >>> messages = [
        ...     SystemMessage(content="你是一个AI助手。"),
        ...     HumanMessage(content="你好，请介绍一下自己。")
        ... ]
        >>>
        >>> response = llm.invoke(messages)
        >>> print(response.content)
    """
    return ZhipuAIAdapter(model=model, temperature=temperature, api_key=api_key)
