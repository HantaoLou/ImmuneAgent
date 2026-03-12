"""
SSE Callback Handler for LangChain

使用 LangChain 标准的 CallbackManager 机制来实时推送 LLM 思考过程
"""

from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class SSECallbackHandler(BaseCallbackHandler):
    """
    自定义回调处理器，通过 SSE 实时推送 LLM 思考过程

    使用方式：
    ```python
    from langchain_community.chat_models import ChatZhipuAI
    from agent.utils.sse_callback_handler import SSECallbackHandler
    from langchain_core.callbacks.manager import CallbackManager

    # 创建带 SSE 回调的 LLM
    llm = ChatZhipuAI(
        model="glm-4",
        streaming=True,
        callbacks=[SSECallbackHandler(progress_callback)]
    )
    ```
    """

    def __init__(self, progress_callback: Optional[callable] = None):
        self.progress_callback = progress_callback
        self.accumulated_tokens = []
        self.current_llm_start = None
        self.last_sent_position = 0  # 记录上次发送的位置

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """LLM 开始时调用"""
        import time

        self.current_llm_start = time.time()
        self.accumulated_tokens = []
        self.last_sent_position = 0

        if self.progress_callback:
            try:
                # 提取用户问题
                user_msg = prompts[0][:100] if prompts else ""

                self.progress_callback(
                    event_type="llm_thinking",
                    message=f"[THINK] 开始思考...",
                    details={
                        "phase": "thinking_start",
                        "model": kwargs.get("invocation_params", {}).get(
                            "model", "unknown"
                        ),
                        "prompt_count": len(prompts),
                        "user_msg": user_msg,
                    },
                )
            except Exception as e:
                print(f"[SSECallbackHandler] Error in on_llm_start: {e}")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """每个新 token 生成时调用"""
        self.accumulated_tokens.append(token)

        if self.progress_callback:
            try:
                current_position = len(self.accumulated_tokens)

                # 每 20 个 token 或每 0.5 秒发送一次新内容
                if current_position - self.last_sent_position >= 20:
                    # 只发送新产生的 tokens
                    new_tokens = self.accumulated_tokens[
                        self.last_sent_position : current_position
                    ]
                    new_text = "".join(new_tokens)

                    self.progress_callback(
                        event_type="llm_streaming",
                        message=new_text,
                        details={
                            "phase": "thinking_progress",
                            "total_tokens": current_position,
                            "is_incremental": True,  # 标记这是增量内容
                        },
                    )
                    self.last_sent_position = current_position

            except Exception as e:
                print(f"[SSECallbackHandler] Error in on_llm_new_token: {e}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 结束时调用"""
        import time

        if self.progress_callback:
            try:
                elapsed = (
                    time.time() - self.current_llm_start
                    if self.current_llm_start
                    else 0
                )

                # 发送剩余未发送的 tokens
                if self.last_sent_position < len(self.accumulated_tokens):
                    remaining_tokens = self.accumulated_tokens[
                        self.last_sent_position :
                    ]
                    remaining_text = "".join(remaining_tokens)
                    if remaining_text:
                        self.progress_callback(
                            event_type="llm_streaming",
                            message=remaining_text,
                            details={
                                "phase": "thinking_progress",
                                "total_tokens": len(self.accumulated_tokens),
                                "is_incremental": True,
                            },
                        )

                full_text = "".join(self.accumulated_tokens)

                self.progress_callback(
                    event_type="llm_thinking",
                    message=f"[SUCCESS] 思考完成",
                    details={
                        "phase": "thinking_complete",
                        "total_tokens": len(self.accumulated_tokens),
                        "elapsed_seconds": round(elapsed, 2),
                        "total_length": len(full_text),
                        "full_content": full_text,  # 发送完整内容供前端展示
                    },
                )

                # 清理
                self.accumulated_tokens = []
                self.current_llm_start = None
                self.last_sent_position = 0

            except Exception as e:
                print(f"[SSECallbackHandler] Error in on_llm_end: {e}")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """LLM 出错时调用"""
        if self.progress_callback:
            try:
                self.progress_callback(
                    event_type="error",
                    message=f"[LLM Error] {str(error)[:200]}",
                    details={
                        "phase": "error",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )
            except Exception as e:
                print(f"[SSECallbackHandler] Error in on_llm_error: {e}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """链开始时调用"""
        if self.progress_callback:
            try:
                self.progress_callback(
                    event_type="chain_start",
                    message=f"[Chain Start] {serialized.get('name', 'unknown')}",
                    details={"phase": "chain_start", "inputs": str(inputs)[:100]},
                )
            except Exception as e:
                print(f"[SSECallbackHandler] Error in on_chain_start: {e}")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """工具开始时调用"""
        if self.progress_callback:
            try:
                tool_name = serialized.get("name", "unknown")

                self.progress_callback(
                    event_type="tool_call",
                    message=f"[Tool Call] {tool_name}",
                    details={
                        "phase": "tool_start",
                        "tool_name": tool_name,
                        "input": input_str[:100],
                    },
                )
            except Exception as e:
                print(f"[SSECallbackHandler] Error in on_tool_start: {e}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """工具结束时调用"""
        if self.progress_callback:
            try:
                self.progress_callback(
                    event_type="tool_result",
                    message=f"[Tool Result] {output[:100]}...",
                    details={
                        "phase": "tool_complete",
                        "output_length": len(output),
                    },
                )
            except Exception as e:
                print(f"[SSECallbackHandler] Error in on_tool_end: {e}")


def create_llm_with_sse(
    model: str = "glm-4-flash",
    temperature: float = 0.7,
    progress_callback: Optional[callable] = None,
    streaming: bool = True,
    **kwargs,
):
    """
    创建带 SSE 回调的 LLM 实例

    Args:
        model: 模型名称
        temperature: 温度参数
        progress_callback: SSE 进度回调函数
        streaming: 是否启用流式输出
        **kwargs: 其他参数

    Returns:
        配置了 SSE 回调的 LLM 实例
    """
    import os

    # 优先使用 ZhipuAIAdapter（有完整的 streaming callback 支持）
    ChatZhipuAI = None
    use_adapter = False

    try:
        from utils.zhipu_adapter import ZhipuAIAdapter

        ChatZhipuAI = ZhipuAIAdapter
        use_adapter = True
    except ImportError:
        try:
            from agent.utils.zhipu_adapter import ZhipuAIAdapter

            ChatZhipuAI = ZhipuAIAdapter
            use_adapter = True
        except ImportError:
            pass

    # 如果 ZhipuAIAdapter 不可用，尝试使用 langchain_community
    if ChatZhipuAI is None:
        try:
            from langchain_community.chat_models import (
                ChatZhipuAI as LangChainChatZhipuAI,
            )

            ChatZhipuAI = LangChainChatZhipuAI
        except ImportError:
            raise ImportError(
                "Neither ZhipuAIAdapter nor langchain_community.chat_models.ChatZhipuAI is available"
            )

    # 获取 API key
    zhipu_api_key = os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
    if zhipu_api_key and not os.getenv("ZHIPUAI_API_KEY"):
        os.environ["ZHIPUAI_API_KEY"] = zhipu_api_key

    if use_adapter:
        # ZhipuAIAdapter 直接支持 progress_callback
        llm = ChatZhipuAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            progress_callback=progress_callback,
            **kwargs,
        )
    else:
        # langchain_community.ChatZhipuAI 使用 callbacks
        sse_handler = SSECallbackHandler(progress_callback=progress_callback)
        llm = ChatZhipuAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            callbacks=[sse_handler],
            **kwargs,
        )

    return llm


def attach_sse_to_llm(llm: Any, progress_callback: callable) -> Any:
    """
    为已有的 LLM 实例附加 SSE 回调

    Args:
        llm: 已有的 LLM 实例
        progress_callback: SSE 进度回调函数

    Returns:
        配置了 SSE 回调的 LLM 实例
    """
    # 创建 SSE 回调处理器
    sse_handler = SSECallbackHandler(progress_callback=progress_callback)

    # 获取现有的 callbacks（如果有）
    existing_callbacks = getattr(llm, "callbacks", [])

    # 添加新的回调处理器
    if existing_callbacks:
        llm.callbacks = existing_callbacks + [sse_handler]
    else:
        llm.callbacks = [sse_handler]

    # 确保流式输出启用
    if hasattr(llm, "streaming"):
        llm.streaming = True

    return llm
