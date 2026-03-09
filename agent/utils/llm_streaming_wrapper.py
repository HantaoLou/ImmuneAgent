"""
LLM Streaming Wrapper - 自动为所有LLM调用添加thinking报告

这个wrapper会拦截LLM的invoke调用，自动报告thinking过程。
不需要修改任何现有代码，只需要在创建LLM时使用这个wrapper。
"""

from typing import Any, List, Optional, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import time


class StreamingLLMWrapper:
    """
    LLM包装器，自动报告thinking过程

    使用方式：
    ```python
    llm = create_reasoning_llm()
    wrapped_llm = StreamingLLMWrapper(llm, progress_callback)
    response = wrapped_llm.invoke(messages)  # 自动报告thinking
    ```
    """

    def __init__(self, llm: Any, progress_callback: Optional[Callable] = None):
        self.llm = llm
        self.progress_callback = progress_callback

    def invoke(self, messages: List[BaseMessage], **kwargs) -> Any:
        """拦截invoke调用，添加thinking报告"""

        if not self.progress_callback:
            # 没有callback，直接调用
            return self.llm.invoke(messages, **kwargs)

        # 🔥 报告开始思考
        try:
            user_msg = ""
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    user_msg = str(msg.content)[:100]
                    break

            self.progress_callback(
                event_type="llm_thinking",
                message=f"🤔 开始思考: {user_msg}",
                details={
                    "phase": "thinking_start",
                    "model": getattr(self.llm, "model", "unknown"),
                },
            )
        except Exception as e:
            print(f"[StreamingLLMWrapper] Error reporting start: {e}")

        # 🔥 调用LLM
        start_time = time.time()
        try:
            response = self.llm.invoke(messages, **kwargs)
        except Exception as e:
            # 报告错误
            if self.progress_callback:
                self.progress_callback(
                    event_type="error",
                    message=f"❌ LLM调用失败: {str(e)[:100]}",
                    details={"error": str(e)},
                )
            raise

        elapsed = time.time() - start_time

        # 🔥 报告thinking内容
        try:
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            if content:
                # 分段报告thinking过程
                chunks = self._split_content(content)

                for i, chunk in enumerate(chunks):
                    self.progress_callback(
                        event_type="llm_streaming",
                        message=f"💭 {chunk}",
                        details={
                            "phase": "thinking_progress",
                            "chunk_number": i + 1,
                            "total_chunks": len(chunks),
                            "elapsed_seconds": round(elapsed, 2),
                        },
                    )
                    time.sleep(0.03)  # 模拟流式效果

                # 报告完成
                self.progress_callback(
                    event_type="llm_thinking",
                    message=f"✅ 思考完成 (共 {len(chunks)} 个思维片段, 耗时 {elapsed:.1f}s)",
                    details={
                        "phase": "thinking_complete",
                        "total_length": len(content),
                        "total_chunks": len(chunks),
                        "elapsed_seconds": round(elapsed, 2),
                    },
                )
        except Exception as e:
            print(f"[StreamingLLMWrapper] Error reporting content: {e}")

        return response

    def _split_content(self, content: str, max_chunk_size: int = 200) -> List[str]:
        """将内容分割成思维片段"""
        if len(content) <= max_chunk_size:
            return [content]

        # 按句子分割
        import re

        sentences = re.split(r"[。！？\n]", content)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if not sentence.strip():
                continue

            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [content[:max_chunk_size]]

    def __getattr__(self, name: str) -> Any:
        """代理所有其他属性到原始LLM"""
        return getattr(self.llm, name)


def wrap_llm_with_streaming(
    llm: Any, progress_callback: Optional[Callable] = None
) -> Any:
    """
    为LLM添加流式thinking报告

    Args:
        llm: 原始LLM实例
        progress_callback: 进度回调函数

    Returns:
        包装后的LLM实例
    """
    if not progress_callback:
        return llm

    return StreamingLLMWrapper(llm, progress_callback)
