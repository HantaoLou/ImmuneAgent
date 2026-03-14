"""
Sandbox Handlers - 用于将沙盒执行日志推送到 ProgressTracker
"""

from dataclasses import dataclass
from typing import Callable, Optional, Any


@dataclass
class ExecutionHandlers:
    """执行处理器集合"""

    on_stdout: Optional[Callable[[str], None]] = None
    on_stderr: Optional[Callable[[str], None]] = None
    on_init: Optional[Callable[[], None]] = None
    on_execution_complete: Optional[Callable[[Any], None]] = None
    on_error: Optional[Callable[[str], None]] = None


def create_sandbox_handlers(
    progress_callback: Callable, session_id: str, node_name: str = "sandbox"
) -> ExecutionHandlers:
    """
    创建沙盒执行处理器

    Args:
        progress_callback: 进度回调函数
        session_id: 会话 ID
        node_name: 节点名称

    Returns:
        ExecutionHandlers 对象
    """

    def on_stdout(text: str):
        if text and text.strip():
            progress_callback(
                event_type="sandbox_stdout",
                message=text.strip(),
                node_name=node_name,
                details={"source": "stdout"},
            )

    def on_stderr(text: str):
        if text and text.strip():
            progress_callback(
                event_type="sandbox_stderr",
                message=text.strip(),
                node_name=node_name,
                details={"source": "stderr"},
            )

    def on_init():
        progress_callback(
            event_type="sandbox_init",
            message=f"Sandbox initialized: {session_id}",
            node_name=node_name,
            details={"session_id": session_id},
        )

    def on_execution_complete(result: Any):
        progress_callback(
            event_type="sandbox_complete",
            message="Sandbox execution completed",
            node_name=node_name,
            details={"result_type": type(result).__name__},
        )

    def on_error(error_msg: str):
        progress_callback(
            event_type="sandbox_error",
            message=error_msg,
            node_name=node_name,
            details={"error": error_msg},
        )

    return ExecutionHandlers(
        on_stdout=on_stdout,
        on_stderr=on_stderr,
        on_init=on_init,
        on_execution_complete=on_execution_complete,
        on_error=on_error,
    )
