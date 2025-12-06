import asyncio
import json
import os
import traceback
from abc import abstractmethod
from collections import deque
from datetime import datetime
from enum import Enum
from sre_parse import State
from threading import Lock
from typing import Optional, Tuple
from uuid import UUID, uuid4

import yaml
from fastapi import APIRouter, Depends
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import StateGraph
from langgraph.types import Interrupt, interrupt
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette import EventSourceResponse
from sse_starlette.event import ServerSentEvent
from starlette.types import Message

from common.runner import GraphRunner
from web.chat.aisdk import AiSdkMixin
from web.chat.utils import state_to_markdown
from web.db.db import get_db
from web.session.service import (
    SessionArtifactCreate,
    SessionArtifactService,
    SessionService,
)
from web.session.usecases import Usecase, Usecases

router = APIRouter(prefix="/api/chat")


class ChatStreamer:
    def __init__(self):
        pass

    def __aiter__(self):
        return self

    @abstractmethod
    async def __anext__(self):
        pass

    @abstractmethod
    async def on_close(self, message: Message):
        pass


class ItemType(Enum):
    MESSAGE = "message"
    INTERRUPT = "interrupt"
    END = "end"
    ERROR = "error"
    FILE = "file"
    ACTION_REQUEST = "action_request"


class StreamItem(BaseModel):
    item_type: ItemType
    message: Optional[str] = None
    file: Optional[dict] = None
    action_request: Optional[dict] = None


class MessageType(Enum):
    # Supported incoming UI message part types (from @ai-sdk/react UIMessage parts)
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    TOOL_CALL = "tool-call"
    TOOL_RESULT = "tool-result"


class MessagePart(BaseModel):
    type: str
    text: str


class Message(BaseModel):
    id: str
    parts: list[MessagePart]


class StreamItemResponse(BaseModel):
    parts: list[MessagePart]


# Centralize SSE data stream event type strings used by AI SDK UI protocol
class StreamEventType(Enum):
    # start and finish a message
    START = "start"
    FINISH = "finish"

    # start, append and end text
    TEXT_START = "text-start"
    TEXT_DELTA = "text-delta"
    TEXT_END = "text-end"


class ChatRequest(BaseModel):
    session_id: UUID
    messages: list[Message]

    def get_text_message(self):
        ret = ""
        msg = self.messages[-1]
        for part in msg.parts:
            # Only aggregate textual input for the user message payload
            if part.type == MessageType.TEXT.value:
                ret += part.text
        return ret


# Helper to create SSE events with JSON payloads
def sse_json(data_obj: dict) -> ServerSentEvent:
    return ServerSentEvent(data=json.dumps(data_obj))


class GraphRunnerChatStreamer(ChatStreamer, AiSdkMixin):
    def __init__(
        self,
        db: AsyncSession,
        session_id: UUID,
        graph_runner: GraphRunner,
        initial_state: StateGraph,
        rc: RunnableConfig,
        uc: Usecase,
    ):
        super().__init__()
        # Initialize AI SDK mixin state and buffers
        AiSdkMixin.__init__(self)
        self.session_id = session_id
        self.initial_state = initial_state
        self.rc = rc
        self.db = db
        self.uc = uc
        self.event_queue = asyncio.Queue()
        self.message = ""

        def _handle_message(msg: AIMessage):
            self.event_queue.put_nowait(
                StreamItem(
                    item_type=ItemType.MESSAGE,
                    message=msg.content,
                )
            )
            self.message += msg.content

        self.graph_runner = graph_runner.with_message_handler(_handle_message)
        self.running = False
        self.should_close = False
        self.pending_action_response = None
        self.response_waiters: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Future]] = {}
        self.cached_responses: dict[str, dict] = {}
        self.waiting_for_action_response = False
        
        # 将SSE流处理器添加到config中，供TaskExecutor使用
        if "configurable" not in self.rc:
            self.rc["configurable"] = {}
        self.rc["configurable"]["sse_streamer"] = self
        # 使用标准UUID字符串（包含连字符），与前端/回传保持一致
        self.rc["configurable"]["session_id"] = str(self.session_id)

    async def run(self):
        self.start_message()
        self.start_text()
        try:
            if self.running:
                return

            self.running = True
            ret = await self.graph_runner.run(self.initial_state, self.rc)
            if ret is not None:
                self.send_interrupt(ret)
            else:
                await self._save_state()
                self.event_queue.put_nowait(
                    StreamItem(
                        item_type=ItemType.END,
                    )
                )
        except Exception as e:
            traceback.print_exc()
            self.put_error(e)

    async def resume(self, message):
        self.start_message()
        self.start_text()

        try:
            if not self.running:
                return

            self.running = True
            ret = await self.graph_runner.resume(message, self.rc)

            if ret is not None:
                self.send_interrupt(ret)
            else:
                await self._save_state()
                self.event_queue.put_nowait(
                    StreamItem(
                        item_type=ItemType.END,
                    )
                )
        except Exception as e:
            logger.error("error: {}", e)
            self.put_error(e)

    async def _save_state(self):
        state = self.graph_runner.get_state(
            {
                "thread_id": self.session_id.hex,
                "configurable": self.rc,
            }
        )
        result: str = self.uc.result_factory(state)
        # save state as state.yaml as artifact
        artifact = await SessionArtifactService.create_artifact_with_binary(
            self.db,
            self.session_id,
            "result.md",
            result.encode(),
            "result.md",
            "text/markdown",
        )
        url = f"/api/sessions/artifacts/{artifact.id}/download"
        logger.info("sending file: {}", url)
        self.event_queue.put_nowait(
            StreamItem(
                item_type=ItemType.FILE,
                file={
                    "filename": "result.md",
                    "url": url,
                    "size": len(result.encode()),
                    "type": "text/markdown",
                },
            )
        )

    def stop(self):
        self.graph_runner.stop()
        self.event_queue.put_nowait(
            StreamItem(
                item_type=ItemType.ERROR,
                message="Session stopped",
            )
        )
        self.running = False

    def put_error(self, e: Exception):
        self.event_queue.put_nowait(
            StreamItem(
                item_type=ItemType.ERROR,
                message=str(e),
            )
        )

    def send_interrupt(self, i: Interrupt):
        self.event_queue.put_nowait(
            StreamItem(
                item_type=ItemType.INTERRUPT,
                message=str(i.value),
            )
        )

    async def flush_messages(self, db: AsyncSession):
        await SessionService.save_chat_history(
            db, self.session_id, self.message, role="assistant"
        )
        self.message = ""

    async def _on_message(self, item: StreamItem):
        self.append_text(item.message or "")

    async def _on_interrupt(self, item: StreamItem):
        # Finish previous message
        self.end_text()

        # Start a fresh assistant message conveying the interrupt content
        self.start_text()
        self.append_text(item.message or "")
        self.end_text()
        self.finish_message()
        self.terminate_stream()
        await self.flush_messages(self.db)

    async def _on_end(self, item: StreamItem):
        self.end_text()
        self.finish_message()

        # Data stream end marker compatible with AI SDK
        self.terminate_stream()
        self.should_close = True
        await self.flush_messages(self.db)

    async def _on_error(self, item: StreamItem):
        self.send_err(item.message or "Unknown error")

    async def _on_file(self, item: StreamItem):
        self.end_text()
        self.send_file(
            item.file["type"],
            item.file["url"],
            item.file["filename"],
            item.file["size"],
        )
        self.start_text()

    async def _on_action_request(self, item: StreamItem):
        # 发送action请求到前端作为专门的action事件
        action_request = dict(item.action_request or {})
        event_action_id = (
            action_request.get("timestamp")
            or action_request.get("action_id")
            or self._get_timestamp()
        )
        action_request["timestamp"] = event_action_id
        action_request.setdefault("action_id", event_action_id)
        self.send_action_request(action_request)
        # 确保消息被立即发送：先刷新消息，然后设置等待标志
        await self.flush_messages(self.db)
        # 设置标志表示正在等待action响应
        # 注意：消息已经在sse_buffer中，会在下一次__anext__调用时返回
        self.waiting_for_action_response = True
        logger.info(f"Action request sent to frontend: action_id={event_action_id}, buffer_size={len(self.sse_buffer)}")

    async def __anext__(self):
        # If we have buffered SSE events (e.g., start then text-start), emit them first
        part = self.get_next_part()
        if part is not None:
            return part

        # If we marked the stream to close and there's no buffered event, end the iterator
        if self.should_close:
            raise StopAsyncIteration

        # 如果正在等待action响应，不要阻塞，继续处理队列
        # wait_for_action_response 会在另一个地方等待响应
        # 这里只需要检查是否有缓冲的事件需要处理
        if self.waiting_for_action_response:
            part = self.get_next_part()
            if part is not None:
                return part
            # 如果还在等待响应，不要阻塞，继续处理队列
            # 但可以短暂等待，避免CPU占用过高
            await asyncio.sleep(0.01)

        item: StreamItem = await self.event_queue.get()

        if item.item_type == ItemType.MESSAGE:
            await self._on_message(item)

        elif item.item_type == ItemType.INTERRUPT:
            await self._on_interrupt(item)

        elif item.item_type == ItemType.END:
            await self._on_end(item)

        elif item.item_type == ItemType.ERROR:
            await self._on_error(item)

        elif item.item_type == ItemType.FILE:
            await self._on_file(item)

        elif item.item_type == ItemType.ACTION_REQUEST:
            await self._on_action_request(item)
            # 处理完action请求后，立即检查缓冲区，确保消息被发送
            next_part = self.get_next_part()
            if next_part is not None:
                logger.info(f"Action request message ready to send, buffer_size={len(self.sse_buffer)}")
                return next_part

        next_part = self.get_next_part()
        if next_part is not None:
            return next_part
        # Fallback: if nothing to send, continue to next iteration
        raise StopAsyncIteration

    def push_action_request(self, action_data: dict):
        """推送action请求到前端"""
        self.event_queue.put_nowait(
            StreamItem(
                item_type=ItemType.ACTION_REQUEST,
                action_request=action_data,
            )
        )

    async def wait_for_action_response(self, timeout: int = 300, event_name: str = "default", session_id: Optional[str] = None) -> Optional[dict]:
        """等待前端action响应"""
        import time

        start = time.monotonic()
        print(f"[GraphRunnerChatStreamer] wait_for_action_response 开始等待，超时: {timeout}秒 @ {start:.6f}")
        self.pending_action_response = None
        self.waiting_for_action_response = True

        if session_id is None:
            session_id = str(self.session_id) if getattr(self, "session_id", None) else "no-session"
        else:
            session_id = str(session_id)
        # 始终使用  session_id:event_name  作为Future键，避免ISO时间戳中的":"被误判为复合键
        composite_key = f"{session_id}:{event_name}"
        # 同时检查复合键和裸event键的缓存
        if composite_key in self.cached_responses:
            cached = self.cached_responses.pop(composite_key)
            end = time.monotonic()
            print(
                f"[GraphRunnerChatStreamer] wait_for_action_response 命中缓存响应 @ {end:.6f}，耗时 {end - start:.6f}s"
            )
            self.waiting_for_action_response = False
            cached.pop("__action_id", None)
            return cached
        if event_name in self.cached_responses:
            cached = self.cached_responses.pop(event_name)
            end = time.monotonic()
            print(
                f"[GraphRunnerChatStreamer] wait_for_action_response 命中裸键缓存响应 @ {end:.6f}，耗时 {end - start:.6f}s"
            )
            self.waiting_for_action_response = False
            cached.pop("__action_id", None)
            return cached

        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        self.response_waiters[composite_key] = (loop, waiter)
        print(
            f"[GraphRunnerChatStreamer] wait_for_action_response 为事件 {event_name} 创建Future (id={id(waiter)})"
        )

        try:
            response = await asyncio.wait_for(waiter, timeout=timeout)
            end = time.monotonic()
            print(f"[GraphRunnerChatStreamer] wait_for_action_response 收到响应 @ {end:.6f}，耗时 {end - start:.6f}s")

            self.pending_action_response = None
            self.waiting_for_action_response = False
            self.response_waiters.pop(composite_key, None)
            print(f"[GraphRunnerChatStreamer] wait_for_action_response 清理完成")
            if isinstance(response, dict):
                response.pop("__action_id", None)
            return response
        except asyncio.TimeoutError:
            end = time.monotonic()
            print(f"[GraphRunnerChatStreamer] wait_for_action_response 超时！超时时间: {timeout}秒 @ {end:.6f}，耗时 {end - start:.6f}s")
            self.pending_action_response = None
            self.waiting_for_action_response = False
            loop_waiter = self.response_waiters.pop(composite_key, None)
            if loop_waiter:
                _, waiter_obj = loop_waiter
                if waiter_obj and not waiter_obj.done():
                    waiter_obj.cancel()
            return None

    def set_action_response(self, response: dict):
        """设置action响应"""
        import time
        timestamp = time.time()
        monotonic_ts = time.monotonic()
        print(
            f"[GraphRunnerChatStreamer] set_action_response 被调用 (时间戳: {timestamp}, monotonic={monotonic_ts:.6f})"
        )
        print(f"[GraphRunnerChatStreamer] 响应: {response}")
        candidate_keys: list[str] = []
        session_id_value = None
        if isinstance(response, dict):
            action_id = response.get("__action_id")
            session_id_value = response.get("session_id") or response.get("sessionId")
            if action_id is not None:
                candidate_keys.append(str(action_id))
            response_type = response.get("type") or response.get("event")
            if response_type:
                candidate_keys.append(str(response_type))
        if "default" not in candidate_keys:
            candidate_keys.append("default")

        delivered = False
        default_session = str(session_id_value or self.session_id or "no-session")

        for event_key in candidate_keys:
            composite_key = f"{default_session}:{event_key}"
            loop_waiter = self.response_waiters.get(composite_key)
            print(
                f"[GraphRunnerChatStreamer] set_action_response 检查事件 {composite_key} | waiter: {loop_waiter}"
            )
            if not loop_waiter:
                continue

            loop, waiter = loop_waiter
            if waiter and not waiter.done():
                print(
                    f"[GraphRunnerChatStreamer] set_action_response 即将唤醒Future id={id(waiter)} (事件 {composite_key})"
                )
                loop.call_soon_threadsafe(waiter.set_result, dict(response))
                self.response_waiters.pop(composite_key, None)
                delivered = True
                print(f"[GraphRunnerChatStreamer] 响应已分发给事件 {composite_key}")
                break
            else:
                print(
                    f"[GraphRunnerChatStreamer] set_action_response waiter已完成或无效，事件 {event_key}"
                )

        # 宽松匹配：如果仍未投递，尝试匹配任何以 :event_key 结尾的waiter（例如 no-session:event）
        if not delivered and self.response_waiters:
            suffix_candidates = [f":{k}" for k in candidate_keys]
            for wait_key, loop_waiter in list(self.response_waiters.items()):
                if any(wait_key.endswith(sfx) for sfx in suffix_candidates):
                    print(f"[GraphRunnerChatStreamer] set_action_response 使用后备匹配命中 {wait_key}")
                    loop, waiter = loop_waiter
                    if waiter and not waiter.done():
                        loop.call_soon_threadsafe(waiter.set_result, dict(response))
                        self.response_waiters.pop(wait_key, None)
                        delivered = True
                        print(f"[GraphRunnerChatStreamer] 响应已分发给后备事件 {wait_key}")
                        break

        if not delivered:
            # 缓存为裸event键和复合键两种形式，提升命中率
            for event_key in candidate_keys:
                self.cached_responses[event_key] = dict(response)
                composite_key = f"{default_session}:{event_key}"
                self.cached_responses[composite_key] = dict(response)
            print(
                f"[GraphRunnerChatStreamer] 未找到等待者，已缓存事件候选键: {candidate_keys}"
            )

        self.pending_action_response = None
        self.waiting_for_action_response = False
        print(f"[GraphRunnerChatStreamer] 响应已设置，等待中的 wait_for_action_response 应该立即返回")

    async def on_close(self, message: Message):
        logger.info("closed with message: {}", message)

    async def persist_plan_state(
        self,
        plan_summary: Optional[dict] = None,
        execution_state: Optional[dict] = None,
        confirmation: Optional[dict] = None,
    ) -> None:
        config = self.rc.setdefault("configurable", {})
        plan_state = dict(config.get("plan_state") or {})

        if plan_summary is not None:
            plan_state["summary"] = plan_summary

        if execution_state:
            existing = plan_state.get("execution_state", {})
            for step_id, state_payload in execution_state.items():
                merged = existing.get(step_id, {})
                if not isinstance(merged, dict):
                    merged = {}
                if isinstance(state_payload, dict):
                    merged.update(state_payload)
                else:
                    merged["status"] = state_payload
                existing[step_id] = merged
            plan_state["execution_state"] = existing

            summary = plan_state.get("summary")
            if isinstance(summary, dict):
                steps = summary.get("steps")
                if isinstance(steps, list):
                    for step in steps:
                        step_id = step.get("step_id") or step.get("stepId")
                        if step_id and step_id in plan_state["execution_state"]:
                            status_value = plan_state["execution_state"][step_id].get("status")
                            if status_value:
                                step["status"] = status_value

        if confirmation is not None:
            plan_state["confirmation"] = confirmation

        plan_state["updated_at"] = datetime.utcnow().isoformat()
        config["plan_state"] = plan_state

        sanitized = self._sanitize_configuration_for_storage(config)
        await SessionService.update_session(self.db, self.session_id, sanitized)

        self.rc["configurable"] = {**config, "plan_state": plan_state}

    def _sanitize_configuration_for_storage(self, config: dict) -> dict:
        def ensure_serializable(value):
            try:
                json.dumps(value)
                return value
            except TypeError:
                if isinstance(value, dict):
                    return {
                        k: ensure_serializable(v)
                        for k, v in value.items()
                        if k not in {"sse_streamer", "ui_callback"}
                    }
                if isinstance(value, (list, tuple, set)):
                    return [ensure_serializable(item) for item in value]
                if isinstance(value, (str, int, float, bool)) or value is None:
                    return value
                return str(value)

        sanitized = {}
        for key, value in config.items():
            if key in {"sse_streamer", "ui_callback"}:
                continue
            sanitized[key] = ensure_serializable(value)
        return sanitized


async def get_graph_from_session(
    db: AsyncSession, session_id: UUID, user_message: str
) -> Tuple[StateGraph, State, RunnableConfig, Usecase]:
    from langgraph.checkpoint.memory import InMemorySaver
    from sqlalchemy.exc import NoResultFound
    from fastapi import HTTPException

    try:
        session = await SessionService.get_session(db, session_id)
    except NoResultFound:
        # Session不存在
        logger.warning(f"Session {session_id} not found in database")
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found. Please create a session first."
        )
    except Exception as e:
        # 其他异常
        logger.error(f"Error getting session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting session: {str(e)}"
        )
    rc = json.loads(session.configuration)
    rc = {
        "thread_id": session_id.hex,
        "configurable": {
            **rc,  # 保留原有配置
            "uuid": session_id.hex,  # 添加 uuid 字段
            "thread_id": session_id.hex,  # 确保 thread_id 在 configurable 中
        },
    }
    uc = Usecases.get_usecase(session.usecase)
    init_state = uc.init_state_factory(user_message)
    build_graph = uc.graph_factory
    g = build_graph().compile(checkpointer=InMemorySaver())
    return g, init_state, rc, uc


def get_resume_input_from_session_id(session_id: str, user_message: str) -> State:
    return user_message


SESSIONS: dict[UUID, GraphRunnerChatStreamer] = {}
SESSION_LOCK = Lock()


async def _open_session(session_id: UUID, db: AsyncSession, request: ChatRequest):
    """
    Open a new session
    """
    graph, state, rc, uc = await get_graph_from_session(
        db, request.session_id, request.get_text_message()
    )
    graph_runner = GraphRunner(
        graph=graph,
    )
    streamer = GraphRunnerChatStreamer(
        db=db,
        session_id=session_id,
        graph_runner=graph_runner,
        initial_state=state,
        rc=rc,
        uc=uc,
    )
    asyncio.create_task(streamer.run())
    SESSIONS[request.session_id] = streamer
    return streamer


class ActionResponse(BaseModel):
    session_id: UUID
    action_id: str | int
    response: dict


@router.post(path="/action-response")
async def action_response(request: ActionResponse, db: AsyncSession = Depends(get_db)):
    """
    Handle action response from frontend
    """
    print(f"[action-response] 收到前端响应: session_id={request.session_id} (type: {type(request.session_id)}), action_id={request.action_id}")
    print(f"[action-response] 响应内容: {request.response}")
    print(f"[action-response] 当前SESSIONS keys (类型和值): {[(type(k), str(k)) for k in SESSIONS.keys()]}")
    
    # 确保session_id是UUID类型
    session_key = request.session_id if isinstance(request.session_id, UUID) else UUID(str(request.session_id))
    
    if session_key in SESSIONS:
        streamer = SESSIONS[session_key]
        print(f"[action-response] ✅ 找到对应的streamer (session_id={session_key})，设置响应...")
        response_payload = dict(request.response)
        response_payload.setdefault("__action_id", str(request.action_id))
        response_payload.setdefault("session_id", str(session_key))
        streamer.set_action_response(response_payload)
        print(f"[action-response] 响应已设置，返回成功")
        # 不创建新任务，直接返回成功
        # 让TaskExecutor继续等待响应
        return {"status": "success"}
    else:
        print(f"[action-response] ❌ 错误: Session {session_key} 不存在于SESSIONS中")
        print(f"[action-response] 尝试查找的key: {session_key} (type: {type(session_key)})")
        print(f"[action-response] SESSIONS中的所有keys: {[str(k) for k in SESSIONS.keys()]}")
        return {"status": "error", "message": "Session not found"}


@router.post(path="/")
@router.post(path="")
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Initiate a chat or resume
    """
    SESSION_LOCK.acquire()
    await SessionService.save_chat_history(
        db, request.session_id, request.get_text_message(), role="user"
    )
    try:
        if request.session_id in SESSIONS:
            streamer = SESSIONS[request.session_id]
            streamer.db = db
            resume_input = get_resume_input_from_session_id(
                request.session_id, request.get_text_message()
            )
            asyncio.create_task(streamer.resume(resume_input))
        else:
            streamer = await _open_session(request.session_id, db, request)
            print(f"[chat] 创建了新session并存储: session_id={request.session_id} (type: {type(request.session_id)})")
            print(f"[chat] 当前SESSIONS中的keys: {[str(k) for k in SESSIONS.keys()]}")
    finally:
        SESSION_LOCK.release()
    # Add AI SDK UI data stream header
    return EventSourceResponse(
        streamer,
        client_close_handler_callable=streamer.on_close,
        headers={"x-vercel-ai-ui-message-stream": "v1"},
    )
