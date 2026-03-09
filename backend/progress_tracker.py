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
    LLM_REASONING = "llm_reasoning"
    LLM_STREAMING = "llm_streaming"
    TOOL_RESULT = "tool_result"
    SUBGRAPH_STEP = "subgraph_step"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    ANALYSIS_PROGRESS = "analysis_progress"
    FILE_CONTENT = "file_content"
    CONSOLE_OUTPUT = "console_output"


class ProgressEvent(BaseModel):
    """进度事件"""

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


class ProgressTracker:
    """进度跟踪器"""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.session_id: Optional[str] = None
        self._active = True
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_session_id(self, session_id: str):
        """设置会话ID"""
        self.session_id = session_id

    async def emit(self, event: ProgressEvent):
        """发送进度事件"""
        if self._active:
            await self.queue.put(event)

    def emit_sync(self, event: ProgressEvent):
        """同步发送进度事件（用于同步代码中）"""
        if not self._active:
            return

        success = False

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.queue.put(event))
            success = True
        except RuntimeError:
            try:
                self.queue.put_nowait(event)
                success = True
            except asyncio.QueueFull:
                pass
            except RuntimeError:
                # 没有运行中的循环，尝试获取主事件循环
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_closed() and loop.is_running():
                        # 使用run_coroutine_threadsafe
                        future = asyncio.run_coroutine_threadsafe(
                            self.queue.put(event), loop
                        )
                        future.result(timeout=1.0)  # 等待完成
                        success = True
                        print(
                            f"[ProgressTracker] Event emitted (threadsafe): {event.event_type}"
                        )
                    else:
                        print(f"[ProgressTracker] Loop not running, direct put")
                        # 直接放入队列
                        self.queue.put_nowait(event)
                        success = True
                except Exception as e:
                    print(f"[ProgressTracker] Failed to emit via event loop: {e}")
                    # 最后的尝试：直接放入队列
                    try:
                        self.queue.put_nowait(event)
                        success = True
                        print(
                            f"[ProgressTracker] Event emitted (direct): {event.event_type}"
                        )
                    except Exception as e2:
                        print(f"[ProgressTracker] Direct put also failed: {e2}")
        except Exception as e:
            print(f"[ProgressTracker] Unexpected error in emit_sync: {e}")

        if success:
            # 保存到持久化存储
            if self.session_id:
                try:
                    storage = get_session_storage()
                    storage.save_message(self.session_id, event.model_dump())
                except Exception as e:
                    print(f"[ProgressTracker] Failed to save message to storage: {e}")
        else:
            print(
                f"[ProgressTracker] WARNING: Failed to emit event: {event.event_type} - {event.message[:50]}"
            )

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


# 全局进度跟踪器管理
_progress_trackers: Dict[str, ProgressTracker] = {}


def create_progress_tracker(session_id: str) -> ProgressTracker:
    """创建进度跟踪器"""
    tracker = ProgressTracker()
    tracker.set_session_id(session_id)
    _progress_trackers[session_id] = tracker
    return tracker


def get_progress_tracker(session_id: str) -> Optional[ProgressTracker]:
    """获取进度跟踪器"""
    return _progress_trackers.get(session_id)


def remove_progress_tracker(session_id: str):
    """移除进度跟踪器"""
    if session_id in _progress_trackers:
        tracker = _progress_trackers.pop(session_id)
        tracker.stop()
