"""
进度跟踪器 - 用于收集和推送agent执行进度
"""

import asyncio
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

from session_storage import get_session_storage


class ProgressEventType(str, Enum):
    """进度事件类型"""

    NODE_START = "node_start"
    NODE_PROGRESS = "node_progress"
    NODE_COMPLETE = "node_complete"
    TASK_START = "task_start"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    CODE_GENERATION = "code_generation"
    CODE_EXECUTION = "code_execution"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    INFO = "info"

    LLM_THINKING = "llm_thinking"
    LLM_THINKING_START = "llm_thinking_start"
    LLM_THINKING_END = "llm_thinking_end"
    LLM_REASONING = "llm_reasoning"
    LLM_STREAMING = "llm_streaming"
    LLM_RESPONSE = "llm_response"
    TOOL_RESULT = "tool_result"
    SUBGRAPH_STEP = "subgraph_step"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    ANALYSIS_PROGRESS = "analysis_progress"
    FILE_CONTENT = "file_content"
    CONSOLE_OUTPUT = "console_output"
    SANDBOX_EXEC = "sandbox_exec"
    SANDBOX_STDOUT = "sandbox_stdout"
    SANDBOX_STDERR = "sandbox_stderr"
    SANDBOX_INIT = "sandbox_init"
    SANDBOX_COMPLETE = "sandbox_complete"
    SANDBOX_ERROR = "sandbox_error"
    OPENCODE_INIT = "opencode_init"
    OPENCODE_STDOUT = "opencode_stdout"
    OPENCODE_STDERR = "opencode_stderr"
    OPENCODE_RESULT = "opencode_result"
    OPENCODE_ERROR = "opencode_error"
    OPENCODE_COMPLETE = "opencode_complete"
    HITL_REQUEST = "hitl_request"
    HITL_CONFIRMED = "hitl_confirmed"
    HITL_REJECTED = "hitl_rejected"


class ProgressEvent(BaseModel):
    """进度事件"""

    session_id: Optional[str] = Field(
        default=None, description="会话ID，用于多会话隔离"
    )
    event_type: ProgressEventType = Field(description="事件类型")
    node_name: Optional[str] = Field(default=None, description="节点名称")
    task_id: Optional[str] = Field(default=None, description="任务ID")
    message: str = Field(description="进度消息")
    details: Dict[str, Any] = Field(default_factory=dict, description="详细信息")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="时间戳"
    )
    progress_percent: Optional[int] = Field(
        default=None, ge=0, le=100, description="进度百分比"
    )


_global_trackers: Dict[str, "ProgressTracker"] = {}

# Global registry of progress callbacks (keyed by session_id)
_global_callbacks: Dict[str, Callable] = {}


class ProgressTracker:
    """进度跟踪器"""

    def __init__(self, session_id: Optional[str] = None):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.session_id: Optional[str] = session_id
        self._active = True
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        if session_id:
            _global_trackers[session_id] = self

    def set_session_id(self, session_id: str):
        """设置会话ID"""
        self.session_id = session_id
        _global_trackers[session_id] = self

    async def push_event(self, event: ProgressEvent):
        """
        推送进度事件（强制验证 session_id）

        Args:
            event: 进度事件
        """
        if not self._active:
            return

        if event.session_id and event.session_id != self.session_id:
            print(
                f"[ProgressTracker] WARNING: Session mismatch: {event.session_id} != {self.session_id}"
            )
            return

        if not event.session_id:
            event.session_id = self.session_id

        await self.queue.put(event)

        if self.session_id:
            try:
                storage = get_session_storage()
                storage.save_message(self.session_id, event.model_dump())
            except Exception as e:
                print(f"[ProgressTracker] Failed to save message to storage: {e}")

    async def emit(self, event: ProgressEvent):
        """发送进度事件"""
        await self.push_event(event)

    def emit_sync(self, event: ProgressEvent):
        """同步发送进度事件（用于同步代码中)"""
        if not self._active:
            return

        success = False

        llm_thinking_events = {
            ProgressEventType.LLM_THINKING,
            ProgressEventType.LLM_THINKING_START,
            ProgressEventType.LLM_THINKING_END,
            ProgressEventType.LLM_REASONING,
            ProgressEventType.LLM_RESPONSE,
            ProgressEventType.LLM_STREAMING,
        }

        is_llm_event = event.event_type in llm_thinking_events

        if self._loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.queue.put(event), self._loop
                )
                future.result(timeout=2.0)
                success = True
            except Exception:
                try:
                    self.queue.put_nowait(event)
                    success = True
                except Exception:
                    pass
        else:
            try:
                self.queue.put_nowait(event)
                success = True
            except asyncio.QueueFull:
                pass
            except Exception:
                pass

        if success:
            if self.session_id:
                try:
                    storage = get_session_storage()
                    storage.save_message(self.session_id, event.model_dump())
                except Exception:
                    pass

    async def get_event(self, timeout: float = 0.1) -> Optional[ProgressEvent]:
        """获取下一个进度事件"""
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def stop(self):
        """停止跟踪器"""
        self._active = False

    def create_callback(self) -> Callable:
        """创建一个进度回调函数，可传递给agent"""

        def progress_callback(
            event_type: str,
            message: str,
            node_name: Optional[str] = None,
            task_id: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None,
            progress_percent: Optional[int] = None,
        ):
            event = ProgressEvent(
                event_type=ProgressEventType(event_type),
                node_name=node_name,
                task_id=task_id,
                message=message,
                details=details or {},
                progress_percent=progress_percent,
            )
            self.emit_sync(event)

        return progress_callback


def create_progress_callback_with_session(session_id: str) -> Optional[Callable]:
    """
    根据session_id创建进度回调函数

    Args:
        session_id: 会话ID

    Returns:
        进度回调函数，如果找不到对应的tracker则返回None
    """
    tracker = get_progress_tracker(session_id)
    if tracker:
        return tracker.create_callback()
    return None


def create_progress_tracker(session_id: str) -> ProgressTracker:
    """
    创建进度跟踪器（强制绑定 session_id 和主事件循环）

    Args:
        session_id: 会话ID

    Returns:
        绑定了 session_id 和主事件循环的 ProgressTracker 实例
    """
    if session_id in _global_trackers:
        return _global_trackers[session_id]

    tracker = ProgressTracker(session_id=session_id)

    try:
        loop = asyncio.get_running_loop()
        if loop is not None:
            tracker._loop = loop
    except RuntimeError:
        pass

    return tracker


def get_progress_tracker(session_id: str) -> Optional[ProgressTracker]:
    """获取进度跟踪器"""
    return _global_trackers.get(session_id)


def remove_progress_tracker(session_id: str):
    """移除进度跟踪器"""
    if session_id in _global_trackers:
        tracker = _global_trackers.pop(session_id)


def set_progress_callback(session_id: str, callback: Callable):
    """
    设置进度回调函数（保存到全局registry）

    Args:
        session_id: 会话ID
        callback: 进度回调函数
    """
    if callback:
        _global_callbacks[session_id] = callback
        print(f"[ProgressTracker] Set callback for session: {session_id}")
        print(
            f"[ProgressTracker] Total callbacks in registry: {len(_global_callbacks)}"
        )
    else:
        print(
            f"[ProgressTracker] WARNING: Trying to set None callback for session: {session_id}"
        )


def get_progress_callback(session_id: str) -> Optional[Callable]:
    """
    获取进度回调函数（从全局registry）

    Args:
        session_id: 会话ID

    Returns:
        进度回调函数，如果不存在则返回None
    """
    callback = _global_callbacks.get(session_id)
    print(f"[ProgressTracker] get_progress_callback called for session: {session_id}")
    print(f"[ProgressTracker] Callback found: {callback is not None}")
    print(f"[ProgressTracker] Total callbacks in registry: {len(_global_callbacks)}")
    print(f"[ProgressTracker] Available sessions: {list(_global_callbacks.keys())}")
    return callback


def remove_progress_callback(session_id: str):
    """
    移除进度回调函数

    Args:
        session_id: 会话ID
    """
    if session_id in _global_callbacks:
        _global_callbacks.pop(session_id)
        print(f"[ProgressTracker] Removed callback for session: {session_id}")
