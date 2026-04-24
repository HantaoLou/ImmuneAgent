"""
Progress Callback Context Manager

Provides a global progress_callback context so all LLM calls can automatically access the callback
"""

from typing import Optional, Callable
from contextvars import ContextVar

_current_progress_callback: ContextVar[Optional[Callable]] = ContextVar(
    "progress_callback", default=None
)


def set_progress_callback(callback: Optional[Callable]) -> None:
    """
    Set the progress_callback for the current context

    Args:
        callback: Progress callback function
    """
    _current_progress_callback.set(callback)


def get_progress_callback() -> Optional[Callable]:
    """
    Get the progress_callback for the current context

    Returns:
        Current progress_callback, or None if not set
    """
    return _current_progress_callback.get()


class ProgressCallbackContext:
    """
    Progress Callback Context Manager

    Usage:
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
