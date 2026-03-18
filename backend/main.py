import json
import os
import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import asyncio
import threading

from config import CORS_ORIGINS, API_HOST, API_PORT, API_RELOAD
from checkpointer import get_checkpointer
from agent_service import (
    AGENT_AVAILABLE,
    invoke_agent_sync,
    create_global_state,
    collect_sandbox_output_files,
    collect_files_from_new_sandbox,
    format_agent_response,
)
from progress_tracker import (
    create_progress_tracker,
    get_progress_tracker,
    remove_progress_tracker,
    ProgressEventType,
    ProgressEvent,
)
from session_storage import get_session_storage

_session_metadata_cache = {}
_current_opensandbox_id = None


def generate_session_id() -> str:
    """生成唯一的会话 ID"""
    return str(uuid.uuid4())


def save_session_metadata(session_id: str, metadata: dict):
    metadata_file = Path(f"./sandbox/sessions/{session_id}/metadata.json")
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)
    _session_metadata_cache[session_id] = metadata


def get_session_metadata(session_id: str) -> dict:
    if session_id in _session_metadata_cache:
        return _session_metadata_cache[session_id]

    metadata_file = Path(f"./sandbox/sessions/{session_id}/metadata.json")
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            _session_metadata_cache[session_id] = metadata
            return metadata
    return {}


app = FastAPI(
    title="Bio-Agent API",
    description="Backend API for Bio-Agent Demo System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    task_type: str
    result: dict


class HITLResumeRequest(BaseModel):
    session_id: str
    hitl_id: str
    confirmed: bool
    feedback: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "bio-agent-backend",
        "version": "1.0.0",
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with SSE streaming"""

    async def event_generator():
        session_id = request.session_id or generate_session_id()
        tracker = None
        agent_result = None

        try:
            yield {
                "event": "status",
                "data": json.dumps(
                    {
                        "stage": "init",
                        "message": "Initializing agent...",
                        "session_id": session_id,
                    }
                ),
            }

            # 创建进度跟踪器
            tracker = create_progress_tracker(session_id)
            progress_callback = tracker.create_callback()

            yield {
                "event": "status",
                "data": json.dumps(
                    {
                        "stage": "creating_state",
                        "message": "Creating agent state...",
                        "session_id": session_id,
                    }
                ),
            }

            await asyncio.sleep(0.1)

            # 创建状态，传入进度回调
            state = create_global_state(request.message, session_id, progress_callback)

            yield {
                "event": "status",
                "data": json.dumps(
                    {
                        "stage": "invoking_agent",
                        "message": "Starting agent execution...",
                        "session_id": session_id,
                    }
                ),
            }

            # 在后台线程中执行 agent
            agent_task = asyncio.create_task(
                asyncio.to_thread(invoke_agent_sync, state)
            )

            # 同时监听进度事件和 agent 完成
            while not agent_task.done():
                # 检查是否有进度事件
                progress_event = await tracker.get_event(timeout=0.1)
                if progress_event:
                    yield {
                        "event": "progress",
                        "data": json.dumps(progress_event.model_dump()),
                    }

                await asyncio.sleep(0.05)

            # 获取 agent 结果
            agent_result = agent_task.result()

            # 发送剩余的进度事件
            while True:
                progress_event = await tracker.get_event(timeout=0.01)
                if not progress_event:
                    break
                yield {
                    "event": "progress",
                    "data": json.dumps(progress_event.model_dump()),
                }

            yield {
                "event": "status",
                "data": json.dumps(
                    {
                        "stage": "formatting_response",
                        "message": "Formatting response...",
                    }
                ),
            }

            # 关键修复：检查是否是 HITL 中断
            # LangGraph 的 interrupt() 不会抛出异常，而是返回包含 __interrupt__ 的结果
            is_hitl_interrupt = (
                isinstance(agent_result, dict)
                and "__interrupt__" in agent_result
                and len(agent_result["__interrupt__"]) > 0
            )

            if is_hitl_interrupt:
                # HITL 中断状态，发送 HITL_REQUEST 事件
                print(
                    f"[chat] Detected HITL interrupt in result: {agent_result.get('__interrupt__')}"
                )
                interrupt_value = agent_result["__interrupt__"][0]
                hitl_request = getattr(interrupt_value, "value", None)

                yield {
                    "event": "progress",
                    "data": json.dumps(
                        ProgressEvent(
                            event_type=ProgressEventType.HITL_REQUEST,
                            message="Waiting for user confirmation",
                            details={
                                "hitl_status": "waiting",
                                "hitl_request": hitl_request,
                            },
                        ).model_dump()
                    ),
                }
                return

            # 在format_agent_response之前检查HITL状态，防止在HITL等待时发送错误的事件
            hitl_status = getattr(agent_result, "hitl_status", None)
            if hitl_status in ("waiting", "waiting_no_interrupt"):
                # HITL中断状态，不format响应，也不发送task_complete事件
                print(
                    f"[chat] HITL status is {hitl_status}, skipping format_agent_response"
                )
                return

            response = format_agent_response(agent_result)
            print(f"[chat] Response session_id: {response.get('session_id')}")
            print(
                f"[chat] Response result keys: {list(response.get('result', {}).keys())}"
            )

            if response.get("session_id"):
                merged_result = response.get("result", {}).get("merged_result", {})
                print(
                    f"[chat] Merged result keys: {list(merged_result.keys()) if merged_result else 'None'}"
                )
                print(
                    f"[chat] opensandbox_id from merged_result: {merged_result.get('opensandbox_id')}"
                )
                print(
                    f"[chat] sandbox_data_dir from merged_result: {merged_result.get('sandbox_data_dir')}"
                )

                session_metadata = {
                    "opensandbox_id": merged_result.get("opensandbox_id"),
                    "sandbox_data_dir": merged_result.get("sandbox_data_dir"),
                    "sandbox_output_dir": merged_result.get("sandbox_output_dir"),
                    "session_id": response.get("session_id"),
                }
                print(f"[chat] Saving session metadata: {session_metadata}")
                save_session_metadata(response["session_id"], session_metadata)

                # 同时保存到全局变量
                global _current_opensandbox_id
                if merged_result.get("opensandbox_id"):
                    _current_opensandbox_id = merged_result.get("opensandbox_id")
                    print(
                        f"[chat] Updated global opensandbox_id: {_current_opensandbox_id}"
                    )

            # 发送输出文件列表
            if response.get("output_files"):
                yield {
                    "event": "output_files",
                    "data": json.dumps(
                        {
                            "files": response["output_files"],
                            "count": response["output_files_count"],
                            "session_id": session_id,
                        }
                    ),
                }

            yield {"event": "done", "data": json.dumps(response)}

        except Exception as e:
            # 检查是否是GraphInterrupt（HITL）
            exception_type = type(e).__name__
            print(f"[chat] Exception occurred: {exception_type}: {str(e)[:200]}")
            if "GraphInterrupt" in exception_type or "Interrupt" in exception_type:
                # HITL中断，发送剩余的进度事件（包括hitl_request），然后保持tracker
                print(f"[chat] HITL interrupt detected: {exception_type}")
                print(f"[chat] Sending remaining progress events...")

                # 发送剩余的进度事件
                while True:
                    progress_event = await tracker.get_event(timeout=0.01)
                    if not progress_event:
                        break
                    print(f"[chat] Sending progress event: {progress_event.event_type}")
                    yield {
                        "event": "progress",
                        "data": json.dumps(progress_event.model_dump()),
                    }

                print(
                    f"[chat] Keeping tracker for HITL session: {session_id} (NOT removing)"
                )
                # 不发送error事件，让前端知道这是正常的HITL中断
                return  # 直接返回，不进入finally（或者进入但不删除）
            else:
                # 其他错误，返回错误信息
                print(f"[chat] Non-HITL error: {exception_type}")
                yield {
                    "event": "error",
                    "data": json.dumps(
                        {
                            "error": str(e),
                            "error_type": exception_type,
                        }
                    ),
                }
        finally:
            # 完全移除tracker删除逻辑，让tracker保持活跃状态
            # HITL和resume都依赖同一个tracker，不应该删除它
            print(
                f"[chat] Finally block: session_id={session_id}, tracker={tracker is not None}"
            )
            # 只在明确要求时才删除tracker（通过单独的API）
            # 这里永远不删除，避免HITL时tracker丢失

    return EventSourceResponse(event_generator())


@app.post("/api/chat/resume")
async def chat_resume(request: HITLResumeRequest):
    """Resume chat after HITL confirmation/rejection"""

    async def generate_resume_event():
        session_id = request.session_id
        print(f"[chat_resume] ========== Resume Request Started ==========")
        print(f"[chat_resume] Session ID: {session_id}")
        print(f"[chat_resume] Request: {request}")

        tracker = get_progress_tracker(session_id)
        print(f"[chat_resume] Tracker found: {tracker is not None}")

        if not tracker:
            print(f"[chat_resume] ERROR: Tracker not found for session: {session_id}")
            from progress_tracker import _global_trackers

            print(f"[chat_resume] Available sessions: {list(_global_trackers.keys())}")
            print(f"[chat_resume] ========== Resume Request Aborted ==========")
            yield {
                "event": "error",
                "data": json.dumps({"error": f"Session {session_id} not found"}),
            }
            return

        # 关键修复：重新设置 progress_callback 到全局 registry
        # 因为 resume_agent 不经过 invoke_agent_sync，不会设置 progress_callback
        progress_callback = tracker.create_callback()
        from progress_tracker import set_progress_callback

        set_progress_callback(session_id, progress_callback)
        print(f"[chat_resume] Set progress callback for session: {session_id}")

        # 关键修复：重新设置 progress_callback 到全局 registry
        # 因为 resume_agent 不经过 invoke_agent_sync，不会设置 progress_callback
        try:
            progress_callback = tracker.create_callback()
            from progress_tracker import set_progress_callback

            set_progress_callback(session_id, progress_callback)
            print(f"[chat_resume] Set progress callback for session: {session_id}")
        except Exception as e:
            print(f"[chat_resume] ERROR setting progress callback: {e}")
            import traceback

            traceback.print_exc()

        print(f"[chat_resume] Tracker found and active: {tracker._active}")
        print(f"[chat_resume] ========== Resume Request Continuing ==========")

        try:
            if not AGENT_AVAILABLE:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": "Agent not available"}),
                }
                return

            from agent.main_graph import build_main_graph
            from checkpointer import get_checkpointer

            try:
                from langgraph.types import Command
            except ImportError:
                Command = None

            checkpointer = get_checkpointer()
            checkpointer_saver = checkpointer.get_saver(session_id)

            if not checkpointer_saver:
                print(f"[chat_resume] ERROR: Checkpointer saver is None!")
                yield {
                    "event": "error",
                    "data": json.dumps({"error": "Checkpointer saver not found"}),
                }
                return

            print(f"[chat_resume] Checkpointer saver: {checkpointer_saver}")
            print(f"[chat_resume] Checkpointer type: {type(checkpointer_saver)}")

            # 检查checkpoint状态
            try:
                checkpoint = await checkpointer_saver.aget(
                    config={"configurable": {"thread_id": session_id}}
                )
                if checkpoint:
                    print(
                        f"[chat_resume] Found checkpoint: {checkpoint.get('id', 'N/A')}"
                    )
                    print(
                        f"[chat_resume] Channel values keys: {list(checkpoint.get('channel_values', {}).keys())[:10]}"
                    )
                    hitl_status = checkpoint.get("channel_values", {}).get(
                        "hitl_status"
                    )
                    print(f"[chat_resume] HITL status from checkpoint: {hitl_status}")
                else:
                    print(f"[chat_resume] WARNING: No checkpoint found for session!")
            except Exception as e:
                print(f"[chat_resume] Error checking checkpoint: {e}")

            graph_with_checkpointer = build_main_graph(checkpointer=checkpointer_saver)
            print(f"[chat_resume] Graph with checkpointer: {graph_with_checkpointer}")

            resume_value = {
                "type": "task_review_response",
                "confirmed": request.confirmed,
                "feedback": request.feedback or "",
                "parameters": request.parameters or {},
            }

            print(
                f"[chat_resume] Resuming session {session_id} with resume_value:",
                resume_value,
            )
            print(f"[chat_resume] Checkpointer saver: {checkpointer_saver}")
            print(f"[chat_resume] Graph: {graph_with_checkpointer}")

            # 检查HITL state文件
            try:
                hitl_state = checkpointer.load_hitl_state(session_id)
                if hitl_state:
                    print(f"[chat_resume] HITL state found: {list(hitl_state.keys())}")
                    print(
                        f"[chat_resume] HITL request keys: {list(hitl_state.get('hitl_request', {}).keys()) if hitl_state.get('hitl_request') else 'N/A'}"
                    )
                    print(
                        f"[chat_resume] HITL response: {hitl_state.get('hitl_response', 'N/A')}"
                    )
                else:
                    print(f"[chat_resume] WARNING: No HITL state found!")
            except Exception as e:
                print(f"[chat_resume] Error loading HITL state: {e}")

            yield {
                "event": "status",
                "data": json.dumps(
                    {
                        "stage": "resuming",
                        "message": "Resuming execution with user response...",
                        "session_id": session_id,
                    }
                ),
            }

            await asyncio.sleep(0.1)

            async def resume_agent():
                try:
                    print(f"[chat_resume] >>> Starting resume_agent execution")
                    if Command is not None:
                        print(f"[chat_resume] Using Command(resume=...)")
                        result = await graph_with_checkpointer.ainvoke(
                            Command(resume=resume_value),
                            config={"configurable": {"thread_id": session_id}},
                        )
                    else:
                        print(f"[chat_resume] Using resume_value directly")
                        result = await graph_with_checkpointer.ainvoke(
                            resume_value,
                            config={"configurable": {"thread_id": session_id}},
                        )
                    print(
                        f"[chat_resume] >>> resume_agent completed, result type: {type(result)}"
                    )
                    return result
                except Exception as e:
                    print(f"[chat_resume] >>> Error during resume: {e}")
                    import traceback

                    traceback.print_exc()
                    raise

            print(f"[chat_resume] >>> Creating resume_agent task")
            agent_task = asyncio.create_task(resume_agent())

            print(f"[chat_resume] >>> Starting progress monitoring loop")
            loop_count = 0
            while not agent_task.done():
                loop_count += 1
                if loop_count % 20 == 0:  # 每秒打印一次（0.05*20=1秒）
                    print(
                        f"[chat_resume] Monitoring loop {loop_count}, agent_task.done={agent_task.done()}"
                    )

                progress_event = await tracker.get_event(timeout=0.1)
                if progress_event:
                    print(f"[chat_resume] Progress event: {progress_event.event_type}")
                    yield {
                        "event": "progress",
                        "data": json.dumps(progress_event.model_dump()),
                    }

                await asyncio.sleep(0.05)

            print(
                f"[chat_resume] >>> Agent task completed after {loop_count} iterations"
            )
            agent_result = agent_task.result()
            print(f"[chat_resume] >>> Got agent result, type: {type(agent_result)}")

            # 关键修复：延长等待时间，确保所有事件都被发送
            # 特别是 HITL 事件，可能在 agent_task 完成后才被调用
            print(f"[chat_resume] >>> Collecting remaining events...")
            collected_events = 0
            max_wait_loops = 20  # 等待最多 1 秒（0.05 * 20）
            for i in range(max_wait_loops):
                progress_event = await tracker.get_event(timeout=0.05)
                if progress_event:
                    collected_events += 1
                    print(
                        f"[chat_resume] >>> Collected event {collected_events}: {progress_event.event_type}"
                    )
                    yield {
                        "event": "progress",
                        "data": json.dumps(progress_event.model_dump()),
                    }

            print(f"[chat_resume] >>> Collected {collected_events} remaining events")

            # 再次检查是否有事件（最后清理）
            while True:
                progress_event = await tracker.get_event(timeout=0.01)
                if not progress_event:
                    break
                collected_events += 1
                print(
                    f"[chat_resume] >>> Final event {collected_events}: {progress_event.event_type}"
                )
                yield {
                    "event": "progress",
                    "data": json.dumps(progress_event.model_dump()),
                }

            print(f"[chat_resume] >>> Total collected events: {collected_events}")

            yield {
                "event": "status",
                "data": json.dumps(
                    {
                        "stage": "formatting_response",
                        "message": "Formatting response...",
                    }
                ),
            }

            # 关键修复：检查是否是 HITL 中断
            # LangGraph 的 interrupt() 不会抛出异常，而是返回包含 __interrupt__ 的结果
            is_hitl_interrupt = (
                isinstance(agent_result, dict)
                and "__interrupt__" in agent_result
                and len(agent_result["__interrupt__"]) > 0
            )

            print(f"[chat_resume] Checking if HITL interrupt: {is_hitl_interrupt}")
            print(f"[chat_resume] agent_result type: {type(agent_result)}")
            print(
                f"[chat_resume] agent_result keys: {agent_result.keys() if isinstance(agent_result, dict) else 'N/A'}"
            )

            response = None  # 初始化 response

            if is_hitl_interrupt:
                # HITL 中断状态，发送 HITL_REQUEST 事件
                print(
                    f"[chat_resume] Detected HITL interrupt in result: {agent_result.get('__interrupt__')}"
                )
                interrupt_value = agent_result["__interrupt__"][0]
                hitl_request = getattr(interrupt_value, "value", None)
                print(f"[chat_resume] Extracted hitl_request: {hitl_request}")

                hitl_event = ProgressEvent(
                    event_type=ProgressEventType.HITL_REQUEST,
                    message="Waiting for user confirmation",
                    details={
                        "hitl_status": "waiting",
                        "hitl_request": hitl_request,
                    },
                )
                await tracker.emit(hitl_event)
                print(f"[chat_resume] HITL_REQUEST event emitted")

                # 立即发送HITL_REQUEST事件（直接yield，不依赖queue）
                yield {
                    "event": "progress",
                    "data": json.dumps(hitl_event.model_dump()),
                }
                print(f"[chat_resume] HITL_REQUEST event yielded to SSE")

                # 对于 HITL 中断，不发送 done 事件，直接返回
                # 让前端通过 progress 事件处理 HITL 请求
                print(f"[chat_resume] HITL interrupt detected, not sending done event")
                return
            else:
                response = format_agent_response(agent_result)

                hitl_status = getattr(agent_result, "hitl_status", None)
                print(f"[chat_resume] hitl_status: {hitl_status}")

                if hitl_status in ("waiting", "waiting_no_interrupt"):
                    hitl_event = ProgressEvent(
                        event_type=ProgressEventType.HITL_REQUEST,
                        message="Waiting for user confirmation",
                        details={"hitl_status": hitl_status, "result": response},
                    )
                    await tracker.emit(hitl_event)

                    # 立即发送HITL_REQUEST事件（直接yield，不依赖queue）
                    yield {
                        "event": "progress",
                        "data": json.dumps(hitl_event.model_dump()),
                    }
                    print(f"[chat_resume] HITL_REQUEST event yielded to SSE")

                    # HITL 状态，不发送 done 事件
                    print(
                        f"[chat_resume] HITL waiting status detected, not sending done event"
                    )
                    return
                else:
                    await tracker.emit(
                        ProgressEvent(
                            event_type=ProgressEventType.TASK_COMPLETE,
                            message="Task completed",
                            details={"result": response},
                        )
                    )
                    print(f"[chat_resume] TASK_COMPLETE event emitted")

            # 只有在非 HITL 状态下才发送 done 事件
            print(f"[chat_resume] Sending done event")
            yield {"event": "done", "data": json.dumps(response)}

        except Exception as e:
            print(f"[chat_resume] Error: {e}")
            import traceback

            traceback.print_exc()
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                ),
            }
        finally:
            # 不删除tracker，让它保持活跃状态
            # resume完成后，可能还会有后续的HITL，所以tracker需要保留
            print(
                f"[chat_resume] Finally block: session_id={session_id}, tracker={tracker is not None}"
            )
            # 只有通过单独的cleanup API才会删除tracker

    return EventSourceResponse(generate_resume_event())


@app.post("/api/chat/submit")
async def chat_submit(request: ChatRequest):
    """Submit a chat task and return session_id for SSE streaming"""
    session_id = request.session_id or generate_session_id()
    tracker = create_progress_tracker(session_id)

    async def run_agent_background():
        try:
            progress_callback = tracker.create_callback()
            state = create_global_state(request.message, session_id, progress_callback)
            result = await asyncio.to_thread(invoke_agent_sync, state)

            # 关键修复：检查是否是 HITL 中断
            # LangGraph 的 interrupt() 不会抛出异常，而是返回包含 __interrupt__ 的结果
            is_hitl_interrupt = (
                isinstance(result, dict)
                and "__interrupt__" in result
                and len(result["__interrupt__"]) > 0
            )

            if is_hitl_interrupt:
                # HITL 中断状态，发送 HITL_REQUEST 事件
                print(
                    f"[chat_submit] Detected HITL interrupt in result: {result.get('__interrupt__')}"
                )
                interrupt_value = result["__interrupt__"][0]
                hitl_request = getattr(interrupt_value, "value", None)

                await tracker.emit(
                    ProgressEvent(
                        event_type=ProgressEventType.HITL_REQUEST,
                        message="Waiting for user confirmation",
                        details={
                            "hitl_status": "waiting",
                            "hitl_request": hitl_request,
                        },
                    )
                )
                return

            # 在format_agent_response之前检查HITL状态，防止在HITL等待时发送错误的事件
            hitl_status = getattr(result, "hitl_status", None)
            if hitl_status in ("waiting", "waiting_no_interrupt"):
                # HITL中断状态，不发送TASK_COMPLETE事件
                print(
                    f"[chat_submit] HITL status is {hitl_status}, skipping task completion event"
                )
                return

            response = format_agent_response(result)

            if response.get("session_id"):
                merged_result = response.get("result", {}).get("merged_result", {})
                session_metadata = {
                    "opensandbox_id": merged_result.get("opensandbox_id"),
                    "sandbox_data_dir": merged_result.get("sandbox_data_dir"),
                    "sandbox_output_dir": merged_result.get("sandbox_output_dir"),
                    "session_id": response.get("session_id"),
                }
                save_session_metadata(response["session_id"], session_metadata)

                global _current_opensandbox_id
                if merged_result.get("opensandbox_id"):
                    _current_opensandbox_id = merged_result.get("opensandbox_id")

            hitl_status = getattr(result, "hitl_status", None)
            if hitl_status in ("waiting", "waiting_no_interrupt"):
                await tracker.emit(
                    ProgressEvent(
                        event_type=ProgressEventType.HITL_REQUEST,
                        message="Waiting for user confirmation",
                        details={"hitl_status": hitl_status, "result": response},
                    )
                )
            else:
                # 检查是否是 GraphInterrupt 导致的异常
                if hasattr(result, "hitl_request") and result.hitl_request is not None:
                    # 如果有 hitl_request，说明是 HITL 中断，发送 HITL_REQUEST 事件
                    await tracker.emit(
                        ProgressEvent(
                            event_type=ProgressEventType.HITL_REQUEST,
                            message="Waiting for user confirmation (HITL interrupt)",
                            details={
                                "hitl_status": hitl_status or "waiting",
                                "result": response,
                                "hitl_request": result.hitl_request,
                            },
                        )
                    )
                else:
                    # 正常完成，发送 TASK_COMPLETE 事件
                    await tracker.emit(
                        ProgressEvent(
                            event_type=ProgressEventType.TASK_COMPLETE,
                            message="Task completed",
                            details={"result": response},
                        )
                    )
        except Exception as e:
            exception_type = type(e).__name__
            print(f"[chat_submit] Exception occurred: {exception_type}: {str(e)[:200]}")

            # 检查是否是 GraphInterrupt（HITL）
            if "GraphInterrupt" in exception_type or "Interrupt" in exception_type:
                # HITL中断，发送剩余的进度事件（包括hitl_request），然后保持tracker
                print(f"[chat_submit] HITL interrupt detected: {exception_type}")
                print(f"[chat_submit] Sending remaining progress events...")

                # 发送剩余的进度事件
                while True:
                    progress_event = await tracker.get_event(timeout=0.01)
                    if not progress_event:
                        break
                    print(
                        f"[chat_submit] Sending progress event: {progress_event.event_type}"
                    )
                    await tracker.emit(progress_event)

                print(
                    f"[chat_submit] Keeping tracker for HITL session: {session_id} (NOT removing)"
                )
                # 不发送error事件，让前端知道这是正常的HITL中断
                # 关键修复：确保不会发送 TASK_COMPLETE 事件
                return
            else:
                # 其他错误，发送错误事件
                print(f"[chat_submit] Non-HITL error: {exception_type}")
                await tracker.emit(
                    ProgressEvent(
                        event_type=ProgressEventType.ERROR,
                        message=str(e),
                        details={"error": str(e), "error_type": type(e).__name__},
                    )
                )

    asyncio.create_task(run_agent_background())

    return {"session_id": session_id, "status": "started"}


@app.get("/api/chat/stream/{session_id}")
async def chat_stream(session_id: str):
    """SSE stream endpoint for real-time execution logs"""

    async def event_generator():
        tracker = get_progress_tracker(session_id)

        if not tracker:
            yield {
                "event": "error",
                "data": json.dumps({"error": f"Session {session_id} not found"}),
            }
            return

        heartbeat_counter = 0
        max_heartbeat_interval = 30

        try:
            while True:
                progress_event = await tracker.get_event(timeout=1.0)

                if progress_event:
                    heartbeat_counter = 0

                    yield {
                        "event": "progress",
                        "data": json.dumps(progress_event.model_dump()),
                    }

                    if progress_event.event_type in (
                        ProgressEventType.TASK_COMPLETE,
                        ProgressEventType.HITL_REQUEST,
                    ):
                        break

                else:
                    heartbeat_counter += 1

                    if heartbeat_counter >= max_heartbeat_interval:
                        yield {
                            "event": "heartbeat",
                            "data": json.dumps(
                                {"timestamp": asyncio.get_event_loop().time()}
                            ),
                        }
                        heartbeat_counter = 0

                    if not tracker._active:
                        break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e), "error_type": type(e).__name__}),
            }
        finally:
            # 不删除tracker，让它在HITL和resume之间保持活跃
            print(
                f"[chat_stream] Finally block: session_id={session_id}, NOT removing tracker"
            )

    return EventSourceResponse(event_generator())


@app.get("/api/sessions")
async def get_sessions():
    """Get list of all sessions"""
    storage = get_session_storage()
    sessions = storage.get_session_list()
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Get all messages for a session"""
    storage = get_session_storage()
    messages = storage.get_messages(session_id)
    return {"session_id": session_id, "messages": messages, "count": len(messages)}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its messages"""
    storage = get_session_storage()
    success = storage.delete_session(session_id)
    if success:
        return {"status": "success", "message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@app.get("/api/sessions/{session_id}/files")
async def get_session_files(session_id: str):
    """
    Get list of output files for a session

    查询沙盒 output 目录下的所有文件，支持本地和远程（OpenSandbox）模式。

    Args:
        session_id: Session ID

    Returns:
        {
            "session_id": "xxx",
            "files": [...],
            "count": 10,
            "source": "local" | "remote"
        }
    """
    files = []
    source = "unknown"

    # 1. 尝试从本地沙盒目录收集文件
    sandbox_dir = f"./sandbox/sessions/{session_id}"
    print(
        f"[get_session_files] Checking local dir: {sandbox_dir}, exists: {os.path.exists(sandbox_dir)}"
    )
    if os.path.exists(sandbox_dir):
        files = collect_sandbox_output_files(sandbox_dir)
        source = "local"

    # 2. 尝试从服务器挂载目录读取（OpenSandbox 挂载的目录）
    if not files:
        # 检查 Windows 和 Linux 路径
        server_mount_paths = [
            f"/data/sessions/{session_id}",  # Linux 服务器挂载路径
            f"D:/data/sessions/{session_id}",  # Windows 可能的路径
            f"C:/data/sessions/{session_id}",
        ]

        for mount_path in server_mount_paths:
            if os.path.exists(mount_path):
                print(f"[get_session_files] Found mounted dir: {mount_path}")
                files = collect_sandbox_output_files(mount_path)
                source = "mounted"
                break

    # 3. 通过创建临时沙盒读取挂载目录
    if not files:
        print(f"[get_session_files] Trying to read via new sandbox...")
        try:
            files = await collect_files_from_new_sandbox(session_id)
            if files:
                source = "sandbox"
        except Exception as e:
            print(f"[get_session_files] Error reading from sandbox: {e}")

    # 3. 如果还是没有文件，检查会话是否存在
    if not files and not os.path.exists(sandbox_dir):
        # 检查是否有元数据（说明会话存在，只是在远程）
        metadata = get_session_metadata(session_id)
        if not metadata:
            # 检查 SessionStorage 中是否存在该 session
            storage = get_session_storage()
            messages = storage.get_messages(session_id)
            if not messages:
                raise HTTPException(
                    status_code=404, detail=f"Session {session_id} not found"
                )

    print(f"[get_session_files] Returning {len(files)} files, source: {source}")
    return {
        "session_id": session_id,
        "files": files,
        "count": len(files),
        "source": source,
    }


@app.get("/api/download/{session_id}/{file_path:path}")
async def download_file(session_id: str, file_path: str):
    """
    Download a file from sandbox output directory

    Uses temporary sandbox to read files from mounted volume.
    """
    from agent_service import _get_cmd_stdout
    from fastapi.responses import Response
    import base64

    sandbox_dir = Path(f"./sandbox/sessions/{session_id}")

    local_paths = [
        sandbox_dir / "output" / file_path,
        sandbox_dir / file_path,
    ]

    for full_path in local_paths:
        if full_path.exists() and full_path.is_file():
            ext = full_path.suffix.lower()
            content_type = _get_content_type(ext)
            return FileResponse(
                path=str(full_path),
                media_type=content_type,
                filename=full_path.name,
            )

    try:
        from opensandbox.sandbox import Sandbox
        from opensandbox.config import ConnectionConfig
        import urllib.request

        domain = os.getenv("SANDBOX_DOMAIN", "localhost:8080")
        api_key = os.getenv("SANDBOX_API_KEY") or os.getenv("OPEN_SANDBOX_API_KEY")

        connection_config = ConnectionConfig(
            domain=domain, api_key=api_key, debug=False
        )

        image = os.getenv("OPENSANDBOX_IMAGE", "python:3.11-slim")
        volume_bindings_env = os.getenv(
            "OPENSANDBOX_VOLUME_BINDINGS",
            "/data/sessions:/data/sessions,/data:/data:ro",
        )
        volume_bindings = []
        for binding in volume_bindings_env.split(","):
            binding = binding.strip()
            if binding:
                parts = binding.split(":")
                if len(parts) >= 2:
                    volume_bindings.append(
                        {"hostPath": parts[0], "containerPath": parts[1]}
                    )

        base_url = f"http://{domain}" if not domain.startswith("http") else domain
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        create_body = {
            "image": {"uri": image},
            "timeout": 300,
            "resourceLimits": {"cpu": "1", "memory": "2Gi"},
            "entrypoint": ["tail", "-f", "/dev/null"],
        }
        if volume_bindings:
            create_body["volumeBindings"] = volume_bindings

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(
            f"{base_url}/sandboxes",
            data=json.dumps(create_body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        sandbox_id = payload.get("id")
        print(f"[download_file] Sandbox created: {sandbox_id}")

        sandbox = await Sandbox.connect(sandbox_id, connection_config=connection_config)

        sandbox_paths = [
            f"/data/sessions/{session_id}/output/{file_path}",
            f"/data/sessions/{session_id}/{file_path}",
        ]

        file_content = None
        actual_path = None

        for sandbox_file_path in sandbox_paths:
            try:
                cmd = f"cat {sandbox_file_path} | base64 2>/dev/null || echo ''"
                result = await sandbox.commands.run(cmd)
                stdout = _get_cmd_stdout(result)

                if stdout and stdout.strip():
                    file_content = base64.b64decode(stdout.strip())
                    actual_path = sandbox_file_path
                    print(f"[download_file] Found file at: {sandbox_file_path}")
                    break
            except Exception as e:
                print(f"[download_file] Error reading {sandbox_file_path}: {e}")
                continue

        try:
            await sandbox.kill()
        except:
            pass

        if file_content:
            filename = os.path.basename(file_path)
            ext = os.path.splitext(filename)[1].lower()
            content_type = _get_content_type(ext)

            return Response(
                content=file_content,
                media_type=content_type,
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

    except Exception as e:
        print(f"[download_file] Error: {e}")
        import traceback

        traceback.print_exc()

    raise HTTPException(status_code=404, detail=f"File not found: {file_path}")


@app.get("/api/sessions/{session_id}/files/download")
async def download_session_file(session_id: str, file_path: str):
    """
    Download a specific file from sandbox output directory

    Supports both local and remote (OpenSandbox) file retrieval.

    Args:
        session_id: Session ID
        file_path: Relative path to the file (e.g., "output/results.csv")
    """
    sandbox_dir = Path(f"./sandbox/sessions/{session_id}")
    full_path = (sandbox_dir / file_path).resolve() if sandbox_dir.exists() else None

    # 安全检查
    if full_path:
        try:
            full_path.relative_to(sandbox_dir.resolve())
        except ValueError:
            raise HTTPException(
                status_code=403, detail="Access denied: file path outside sandbox"
            )

    # 尝试从本地读取
    if full_path and full_path.exists() and full_path.is_file():
        ext = full_path.suffix.lower()
        content_type = _get_content_type(ext)

        return FileResponse(
            path=str(full_path),
            media_type=content_type,
            filename=full_path.name,
            headers={"Content-Disposition": f'attachment; filename="{full_path.name}"'},
        )

    # 本地不存在，尝试从远程沙盒读取
    metadata = get_session_metadata(session_id)
    opensandbox_id = metadata.get("opensandbox_id")
    sandbox_data_dir = metadata.get("sandbox_data_dir") or metadata.get(
        "sandbox_output_dir", ""
    ).rstrip("/output")

    if opensandbox_id and sandbox_data_dir:
        try:
            remote_file_path = f"{sandbox_data_dir}/{file_path}"

            agent_utils_dir = os.path.join(
                os.path.dirname(__file__), "..", "agent", "utils"
            )
            import sys

            if agent_utils_dir not in sys.path:
                sys.path.insert(0, agent_utils_dir)

            from opensandbox_helper import OpenSandboxHelper

            helper = OpenSandboxHelper()
            file_content = await helper.read_file(
                remote_file_path, sandbox_id=opensandbox_id
            )

            if file_content:
                filename = os.path.basename(file_path)
                ext = os.path.splitext(filename)[1].lower()
                content_type = _get_content_type(ext)

                from fastapi.responses import Response

                return Response(
                    content=file_content
                    if isinstance(file_content, bytes)
                    else file_content.encode("utf-8"),
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f'attachment; filename="{filename}"'
                    },
                )
        except Exception as e:
            print(f"[download_session_file] Failed to fetch from remote sandbox: {e}")
            import traceback

            traceback.print_exc()

    # 既没有本地文件也没有远程沙盒
    if not sandbox_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    raise HTTPException(status_code=404, detail=f"File not found: {file_path}")


def _get_content_type(ext: str) -> str:
    """Get content type based on file extension"""
    content_types = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".html": "text/html",
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tsv": "text/tab-separated-values",
        ".fasta": "text/plain",
        ".fa": "text/plain",
    }
    return content_types.get(ext, "application/octet-stream")


@app.get("/api/sessions/{session_id}/files/preview")
async def preview_session_file(session_id: str, file_path: str, max_lines: int = 100):
    """
    Preview a text file from sandbox output directory

    Args:
        session_id: Session ID
        file_path: Relative path to the file
        max_lines: Maximum number of lines to preview (default 100)
    """
    sandbox_dir = Path(f"./sandbox/sessions/{session_id}")

    if not sandbox_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # 安全检查
    full_path = (sandbox_dir / file_path).resolve()

    try:
        full_path.relative_to(sandbox_dir.resolve())
    except ValueError:
        raise HTTPException(
            status_code=403, detail="Access denied: file path outside sandbox"
        )

    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    if not full_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {file_path}")

    # 检查文件大小，限制预览大文件
    file_size = full_path.stat().st_size
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=413, detail="File too large to preview")

    ext = full_path.suffix.lower()

    # 二进制文件不支持预览
    binary_extensions = {".h5ad", ".rds", ".pdf", ".png", ".jpg", ".jpeg"}
    if ext in binary_extensions:
        return {
            "session_id": session_id,
            "file_path": file_path,
            "preview": f"[Binary file: {ext}, use download endpoint]",
            "file_type": ext,
            "file_size": file_size,
        }

    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(f"... (truncated, total {i}+ lines)")
                    break
                lines.append(line.rstrip("\n\r"))

        return {
            "session_id": session_id,
            "file_path": file_path,
            "preview": "\n".join(lines),
            "line_count": len(lines),
            "file_type": ext,
            "file_size": file_size,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Bio-Agent API",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
    )
