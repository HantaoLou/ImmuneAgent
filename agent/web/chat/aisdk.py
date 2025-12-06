import json
import secrets
from collections import deque
from enum import Enum
from typing import Optional

from sse_starlette.event import ServerSentEvent


def sse_json(data_obj: dict) -> ServerSentEvent:
    return ServerSentEvent(data=json.dumps(data_obj))


class StreamEventType(Enum):
    # start and finish a message
    START = "start"
    FINISH = "finish"

    # start, append and end text
    TEXT_START = "text-start"
    TEXT_DELTA = "text-delta"
    TEXT_END = "text-end"

    FILE = "data-file"
    PLAN_SUMMARY = "plan-summary"
    PLAN_CONFIRM_REQUEST = "plan-confirm-request"
    EXECUTION_PROGRESS = "execution-progress"
    EXECUTION_ERROR = "execution-error"
    
    # action request for human-in-the-loop
    ACTION_REQUEST = "action-request"


class AiSdkMixin:
    def __init__(self) -> None:
        self.sse_buffer = deque()
        self.message_id = None
        self.text_id = None

    def _new_text_id(self):
        self.text_id = secrets.token_hex(16)
        return self.text_id

    def _new_message_id(self):
        self.message_id = secrets.token_hex(16)
        return self.message_id

    def start_message(self):
        self._new_message_id()
        self.sse_buffer.append(
            sse_json(
                {
                    "messageId": self.message_id,
                    "type": StreamEventType.START.value,
                }
            )
        )

    def finish_message(self):
        if self.is_message_finished():
            return

        # include messageId for reconciliation with AI SDK UI protocol
        self.sse_buffer.append(
            sse_json(
                {
                    "type": StreamEventType.FINISH.value,
                    "messageId": self.message_id,
                }
            )
        )
        self.message_id = None

    def is_message_finished(self):
        return self.message_id is None

    def start_text(self):
        self._new_text_id()

        self.sse_buffer.append(
            sse_json(
                {
                    "type": StreamEventType.TEXT_START.value,
                    "id": self.text_id,
                }
            )
        )

    def append_text(self, text: str):
        self.sse_buffer.append(
            sse_json(
                {
                    "type": StreamEventType.TEXT_DELTA.value,
                    "id": self.text_id,
                    "delta": text,
                }
            )
        )

    def end_text(self):
        if self.is_message_finished():
            return

        self.sse_buffer.append(
            sse_json(
                {
                    "type": StreamEventType.TEXT_END.value,
                    "id": self.text_id,
                }
            )
        )

    def terminate_stream(self):
        self.sse_buffer.append(ServerSentEvent(data="[DONE]"))

    def send_err(self, e: str):
        self.sse_buffer.append(
            sse_json(
                {
                    "type": "error",
                    "errorText": e,
                }
            )
        )

    def get_next_part(self) -> Optional[ServerSentEvent]:
        if self.sse_buffer:
            return self.sse_buffer.popleft()
        return None

    def send_file(self, media_type: str, url: str, filename: str, size: int):
        self.sse_buffer.append(
            sse_json(
                {
                    "type": StreamEventType.FILE.value,
                    "data": {
                        "url": url,
                        "filename": filename,
                        "size": size,
                        "mediaType": media_type,
                    },
                }
            )
        )

    def send_action_request(self, action_data: dict):
        """发送action请求到前端，使用单独的ACTION_REQUEST事件类型"""
        # 使用单独的action-request事件类型，避免嵌套在text-delta中
        # 这样可以提高性能，不需要解析嵌套的JSON
        import logging
        logger = logging.getLogger(__name__)
        sse_event = sse_json(
            {
                "type": StreamEventType.ACTION_REQUEST.value,
                "actionData": action_data,
            }
        )
        self.sse_buffer.append(sse_event)
        logger.info(f"Action request added to SSE buffer: type={StreamEventType.ACTION_REQUEST.value}, action_id={action_data.get('action_id')}, buffer_size={len(self.sse_buffer)}")

    def send_plan_summary(self, plan_data: dict):
        """发送结构化计划摘要到前端"""
        self.sse_buffer.append(
            sse_json(
                {
                    "type": StreamEventType.PLAN_SUMMARY.value,
                    "plan": plan_data,
                }
            )
        )

    def send_plan_confirmation_request(self, plan_data: dict):
        """发送计划确认请求事件，提示前端进行确认或修改"""
        self.sse_buffer.append(
            sse_json(
                {
                    "type": StreamEventType.PLAN_CONFIRM_REQUEST.value,
                    "plan": plan_data,
                }
            )
        )

    def send_execution_progress(self, progress_data: dict):
        """发送计划执行进度事件"""
        self.sse_buffer.append(
            sse_json(
                {
                    "type": StreamEventType.EXECUTION_PROGRESS.value,
                    "progress": progress_data,
                }
            )
        )

    def send_execution_error(self, error_data: dict):
        """发送执行错误事件，并提供可能的干预选项"""
        self.sse_buffer.append(
            sse_json(
                {
                    "type": StreamEventType.EXECUTION_ERROR.value,
                    "error": error_data,
                }
            )
        )