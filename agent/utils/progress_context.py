"""
Progress Callback Context Manager

提供全局的progress_callback上下文，让所有LLM调用都能自动获取callback
"""

from typing import Optional, Callable
from contextvars import ContextVar

# 使用ContextVar存储当前线程的progress_callback
_current_progress_callback: ContextVar[Optional[Callable]] = ContextVar(
    "progress_callback", default=None
)


def set_progress_callback(callback: Optional[Callable]) -> None:
    """
    设置当前上下文的progress_callback

    Args:
        callback: progress回调函数
    """
    _current_progress_callback.set(callback)


def get_progress_callback() -> Optional[Callable]:
    """
    获取当前上下文的progress_callback

    Returns:
        当前的progress_callback，如果未设置则返回None
    """
    return _current_progress_callback.get()


class ProgressCallbackContext:
    """
    Progress Callback 上下文管理器

    使用方式：
    ```python
    with ProgressCallbackContext(state.progress_callback):
        llm = create_reasoning_llm()
        response = llm.invoke(messages)
    ```
    """

    def __init__(self, callback: Optional[Callable]):
        self.callback = callback
        self.token = None

    def __enter__(self):
        self.token = _current_progress_callback.set(self.callback)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            _current_progress_callback.reset(self.token)
        return False
