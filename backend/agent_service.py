import sys
import os
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGENT_DIR = os.path.join(PROJECT_ROOT, "agent")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

try:
    from agent.state import GlobalState
    from agent.main_graph import build_main_graph

    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Agent modules not available: {e}")
    print("Running in demo mode without agent integration")
    AGENT_AVAILABLE = False
    GlobalState = None


def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())


def create_global_state(
    user_input: str,
    session_id: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Create a state dictionary for the agent"""
    if not session_id:
        session_id = generate_session_id()

    sandbox_dir = f"./sandbox/sessions/{session_id}"

    if AGENT_AVAILABLE:
        return GlobalState(
            user_input=user_input,
            sandbox_dir=sandbox_dir,
            session_id=session_id,
            progress_callback=progress_callback,
        )
    else:
        return {
            "user_input": user_input,
            "sandbox_dir": sandbox_dir,
            "session_id": session_id,
            "progress_callback": progress_callback,
        }


def invoke_agent_sync(state: Any) -> Any:
    """Invoke the agent synchronously and return the final state"""
    if not AGENT_AVAILABLE:
        return {
            **state,
            "merged_result": {
                "message": "Agent is running in demo mode",
                "note": "Install agent dependencies to enable full functionality",
                "user_input": state.get("user_input", ""),
            },
            "user_task_type": "DEMO_MODE",
        }

    progress_callback = None
    if hasattr(state, "progress_callback"):
        progress_callback = state.progress_callback
    elif isinstance(state, dict):
        progress_callback = state.get("progress_callback")

    console_redirector = None
    if progress_callback:
        try:
            from agent.utils.console_output_redirector import (
                ConsoleOutputRedirector,
            )

            console_redirector = ConsoleOutputRedirector(
                progress_callback=progress_callback,
                capture_print=True,
                min_interval_ms=100,
            )
            console_redirector.start_capture()
            if progress_callback:
                progress_callback(
                    event_type="console_output",
                    message="🎮 控制台输出捕获已启动",
                    details={"phase": "console_capture_started"},
                )
        except Exception as e:
            print(f"[agent_service] Failed to start console capture: {e}")

    try:
        graph = build_main_graph()
        result = graph.invoke(state)
        return result
    except Exception as e:
        if hasattr(state, "merged_result"):
            state.merged_result = {
                "error": str(e),
                "status": "failed",
            }
        else:
            state["merged_result"] = {
                "error": str(e),
                "status": "failed",
            }
        return state
    finally:
        if console_redirector:
            try:
                console_redirector.stop_capture()
                if progress_callback:
                    progress_callback(
                        event_type="console_output",
                        message="🛑 控制台输出捕获已停止",
                        details={"phase": "console_capture_stopped"},
                    )
            except Exception as e:
                print(f"[agent_service] Error stopping console capture: {e}")


def collect_sandbox_output_files(sandbox_dir: str) -> List[Dict[str, Any]]:
    """
    Collect output files from sandbox directory

    Args:
        sandbox_dir: Path to sandbox directory

    Returns:
        List of file info dicts with name, path, size, type
    """
    files = []

    if not sandbox_dir:
        return files

    sandbox_path = Path(sandbox_dir)
    output_dir = sandbox_path / "output"

    # 如果 output 目录不存在，尝试直接在 sandbox_dir 下查找
    if not output_dir.exists():
        output_dir = sandbox_path

    if not output_dir.exists():
        return files

    # 支持的文件类型
    supported_extensions = {
        ".csv": "CSV Data",
        ".json": "JSON Data",
        ".txt": "Text File",
        ".md": "Markdown",
        ".tsv": "TSV Data",
        ".fasta": "FASTA Sequence",
        ".fa": "FASTA Sequence",
        ".airr": "AIRR TSV Data",
        ".h5ad": "AnnData Object",
        ".rds": "R Data Object",
        ".pdf": "PDF Report",
        ".png": "Image",
        ".jpg": "Image",
        ".jpeg": "Image",
        ".svg": "Image",
        ".html": "HTML Report",
    }

    try:
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in supported_extensions or ext == "":
                    try:
                        rel_path = file_path.relative_to(output_dir)
                    except ValueError:
                        rel_path = file_path.name

                    file_size = file_path.stat().st_size

                    files.append(
                        {
                            "name": file_path.name,
                            "path": str(file_path),
                            "relative_path": str(rel_path),
                            "size": file_size,
                            "size_formatted": _format_file_size(file_size),
                            "type": supported_extensions.get(ext, "Unknown"),
                            "extension": ext,
                        }
                    )
    except Exception as e:
        print(f"[collect_sandbox_output_files] Error: {e}")

    return files


def _format_file_size(size: int) -> str:
    """Format file size in human-readable format"""
    size_float = float(size)
    for unit in ["B", "KB", "MB", "GB"]:
        if size_float < 1024:
            return f"{size_float:.1f} {unit}"
        size_float /= 1024
    return f"{size_float:.1f} TB"


def format_agent_response(result: Any) -> Dict[str, Any]:
    """Format the agent response for API output"""
    sandbox_dir = None

    if AGENT_AVAILABLE and hasattr(result, "session_id"):
        sandbox_dir = getattr(result, "sandbox_dir", None)
        response = {
            "session_id": result.session_id,
            "task_type": result.user_task_type.value
            if result.user_task_type
            else "UNKNOWN",
            "result": {
                "merged_result": result.merged_result,
                "completed_tasks": {
                    task_id: {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "content": task.content,
                        "result": task.result,
                    }
                    for task_id, task in result.completed_tasks.items()
                },
                "file_paths": result.file_paths,
                "execution_plan": result.execution_plan,
            },
            "sandbox_dir": sandbox_dir,
        }

        if result.supervisor_reasoning:
            response["supervisor"] = {
                "decision": result.supervisor_decision,
                "reasoning": result.supervisor_reasoning,
            }

        # 生成用户友好的摘要
        summary = _generate_user_friendly_summary(result)
        response["summary"] = summary

        # 将answer也放在顶层，方便前端访问
        if summary.get("answer"):
            response["answer"] = summary["answer"]

    else:
        sandbox_dir = result.get("sandbox_dir", "")
        response = {
            "session_id": result.get("session_id", "unknown"),
            "task_type": "DEMO_MODE",
            "result": {
                "merged_result": result.get("merged_result", {}),
                "completed_tasks": {},
                "file_paths": {},
                "execution_plan": None,
            },
            "sandbox_dir": sandbox_dir,
        }
        merged_result = result.get("merged_result", {})
        response["summary"] = {
            "answer": merged_result.get("general_qa_answer")
            or merged_result.get("message", "执行完成"),
            "task_type": "DEMO_MODE",
            "status": "completed",
        }
        response["answer"] = response["summary"]["answer"]

    # 收集沙盒输出文件
    output_files = []

    # 1. 首先检查 result.merged_result 中是否已有文件信息（由 result_evaluator 收集）
    if hasattr(result, "merged_result") and result.merged_result:
        if "output_files" in result.merged_result:
            output_files = result.merged_result["output_files"]

    # 2. 如果没有，尝试从本地沙盒目录收集（本地模式）
    if not output_files and sandbox_dir:
        output_files = collect_sandbox_output_files(sandbox_dir)

    # 3. 如果还是没有，尝试从远程沙盒收集
    if not output_files and hasattr(result, "merged_result") and result.merged_result:
        sandbox_data_dir = result.merged_result.get(
            "sandbox_data_dir"
        ) or result.merged_result.get("sandbox_output_dir", "").rstrip("/output")
        opensandbox_id = result.merged_result.get("opensandbox_id")

        if sandbox_data_dir and opensandbox_id:
            try:
                import sys
                import os
                import asyncio

                agent_utils_dir = os.path.join(
                    os.path.dirname(__file__), "..", "agent", "utils"
                )
                if agent_utils_dir not in sys.path:
                    sys.path.insert(0, agent_utils_dir)

                from opensandbox_helper import OpenSandboxHelper

                helper = OpenSandboxHelper()

                async def collect_remote_files():
                    try:
                        remote_files = await helper.list_files(
                            f"{sandbox_data_dir}/output",
                            recursive=True,
                            sandbox_id=opensandbox_id,
                        )
                        return remote_files
                    except Exception as e:
                        print(f"[collect_remote_files] Error: {e}")
                        return []

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        remote_files = loop.run_in_executor(
                            pool, asyncio.run, collect_remote_files()
                        )
                        remote_files = asyncio.get_event_loop().run_until_complete(
                            remote_files
                        )
                else:
                    remote_files = asyncio.run(collect_remote_files())

                supported_extensions = {
                    ".csv": "CSV Data",
                    ".json": "JSON Data",
                    ".txt": "Text File",
                    ".md": "Markdown",
                    ".tsv": "TSV Data",
                    ".fasta": "FASTA Sequence",
                    ".fa": "FASTA Sequence",
                    ".airr": "AIRR TSV Data",
                    ".h5ad": "AnnData Object",
                    ".rds": "R Data Object",
                    ".pdf": "PDF Report",
                    ".png": "Image",
                    ".jpg": "Image",
                    ".jpeg": "Image",
                    ".svg": "Image",
                    ".html": "HTML Report",
                }

                for file_path in remote_files:
                    if file_path.endswith("/"):
                        continue
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in supported_extensions or ext == "":
                        rel_path = file_path.replace(
                            f"{sandbox_data_dir}/output/", ""
                        ).replace(f"{sandbox_data_dir}/", "")
                        output_files.append(
                            {
                                "name": os.path.basename(file_path),
                                "path": file_path,
                                "relative_path": rel_path,
                                "size": 0,
                                "size_formatted": "Unknown",
                                "type": supported_extensions.get(ext, "Unknown"),
                                "extension": ext,
                            }
                        )

                print(
                    f"[format_agent_response] Collected {len(output_files)} files from remote sandbox"
                )
            except Exception as e:
                print(
                    f"[format_agent_response] Failed to collect files from remote sandbox: {e}"
                )
                import traceback

                traceback.print_exc()

    if output_files:
        response["output_files"] = output_files
        response["output_files_count"] = len(output_files)

    return response


def _generate_user_friendly_summary(result: Any) -> Dict[str, Any]:
    """Generate a user-friendly summary from the agent result"""
    summary = {
        "answer": None,
        "task_type": None,
        "status": "completed",
        "key_findings": [],
    }

    if hasattr(result, "user_task_type") and result.user_task_type:
        summary["task_type"] = result.user_task_type.value

    if hasattr(result, "merged_result") and result.merged_result:
        merged = result.merged_result

        # 优先查找 general_qa_answer
        if "general_qa_answer" in merged and merged["general_qa_answer"]:
            summary["answer"] = merged["general_qa_answer"]
        # 其次查找 general_qa_conclusion
        elif "general_qa_conclusion" in merged and merged["general_qa_conclusion"]:
            summary["answer"] = merged["general_qa_conclusion"]
        # 🔥 新增：查找 final_answer（可能在子图状态中）
        elif "final_answer" in merged and merged["final_answer"]:
            summary["answer"] = merged["final_answer"]
        # 🔥 新增：查找 executor_results 中的信息
        elif "executor_results" in merged:
            exec_results = merged["executor_results"]
            completed = exec_results.get("completed_count", 0)
            total = exec_results.get("total_tasks", 0)
            if total > 0:
                summary["answer"] = (
                    f"✅ 任务执行完成\n\n成功完成 {completed}/{total} 个任务"
                )

        if "error" in merged:
            summary["status"] = "failed"
            summary["error"] = merged.get("error", "Unknown error")

    # 🔥 新增：如果没有找到答案，尝试从其他地方提取
    if not summary["answer"]:
        # 检查是否有execution_plan
        if hasattr(result, "execution_plan") and result.execution_plan:
            summary["answer"] = (
                "📋 实验计划已生成\n\n" + result.execution_plan[:500] + "..."
            )
        # 默认消息
        else:
            summary["answer"] = "✅ 任务执行完成"

    return summary
