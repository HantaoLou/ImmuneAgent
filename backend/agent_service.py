import sys
import os
import uuid
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List


def _collect_log_text(log_entries: Optional[Any]) -> str:
    if not log_entries:
        return ""
    if isinstance(log_entries, str):
        return log_entries
    texts = []
    for entry in log_entries:
        text = getattr(entry, "text", None)
        if text is None:
            text = str(entry)
        texts.append(text)
    return "\n".join(texts)


def _get_cmd_stdout(result: Any) -> str:
    """获取命令执行的标准输出"""
    logs = getattr(result, "logs", None)
    if not logs:
        return ""
    stdout = getattr(logs, "stdout", None)
    if not stdout:
        return ""
    if isinstance(stdout, str):
        return stdout
    return "\n".join(getattr(entry, "text", str(entry)) for entry in stdout)


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
    print(f"[agent_service] Agent not available: {e}")
    AGENT_AVAILABLE = False
    GlobalState = None

    def build_main_graph(*args, **kwargs):
        raise ImportError("Agent module not available")


def generate_session_id() -> str:
    return f"{int(__import__('time').time() * 1000)}-{uuid.uuid4().hex[:12]}"


def create_global_state(
    message: str,
    session_id: str,
    progress_callback: Optional[Callable] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    if not AGENT_AVAILABLE:
        return None

    from agent.state import GlobalState, UserTaskType

    sandbox_dir = f"/data/sessions/{session_id}"

    uploaded_files = []
    if attachments:
        for att in attachments:
            sandbox_path = att.get("sandboxPath")
            if sandbox_path:
                uploaded_files.append(sandbox_path)
                print(f"[agent_service] Added uploaded file: {sandbox_path}")

    state = GlobalState(
        session_id=session_id,
        user_input=message,
        sandbox_dir=sandbox_dir,
        user_task_type=UserTaskType.GENERAL_QA,
        uploaded_files=uploaded_files,
    )

    # 将callback设置到全局registry，而不是存储在state中
    if progress_callback:
        from progress_tracker import set_progress_callback

        set_progress_callback(session_id, progress_callback)
        print(f"[agent_service] Set progress callback for session: {session_id}")

    return state


def invoke_agent_sync(state: Any) -> Any:
    if not AGENT_AVAILABLE:
        return {
            "session_id": state.session_id,
            "merged_result": {"message": "Agent not available"},
        }

    import asyncio
    from agent.utils.console_output_redirector import (
        ConsoleOutputRedirector,
        set_global_redirector,
    )

    async def run_agent():
        from checkpointer import get_checkpointer

        checkpointer_saver = get_checkpointer().get_saver(state.session_id)
        graph = build_main_graph(checkpointer=checkpointer_saver)

        if checkpointer_saver:
            result = await graph.ainvoke(
                state, config={"configurable": {"thread_id": state.session_id}}
            )
        else:
            result = await graph.ainvoke(state)
        return result

    redirector = None
    # 从全局registry获取progress_callback（通过session_id）
    progress_callback = None
    if state.session_id:
        try:
            from progress_tracker import get_progress_callback

            progress_callback = get_progress_callback(state.session_id)
            print(
                f"[agent_service] Got progress callback from registry: {progress_callback is not None}"
            )
        except Exception as e:
            print(f"[agent_service] Failed to get progress callback: {e}")

    if progress_callback:
        try:
            redirector = ConsoleOutputRedirector(
                progress_callback=progress_callback,
                capture_print=True,
                min_interval_ms=200,
            )
            redirector.start_capture()
            set_global_redirector(redirector)
            print("[agent_service] Console output capture started")
        except Exception as e:
            print(f"[agent_service] Failed to start console capture: {e}")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(run_agent())
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.run_until_complete(loop.shutdown_default_executor())
                    import time

                    time.sleep(0.1)
                except Exception:
                    pass
                finally:
                    loop.close()
    except Exception as e:
        print(f"[agent_service] Error running agent: {e}")
        raise
    finally:
        # 停止控制台输出重定向
        if redirector:
            try:
                redirector.stop_capture()
                set_global_redirector(None)
                print("[agent_service] Console output capture stopped")
            except Exception as e:
                print(f"[agent_service] Failed to stop console capture: {e}")


def collect_sandbox_output_files(sandbox_dir: str) -> List[Dict[str, Any]]:
    files = []
    sandbox_path = Path(sandbox_dir)

    if not sandbox_path.exists():
        return files

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
        ".xlsx": "Excel File",
        ".xls": "Excel File",
    }

    scan_dirs = [
        ("input", "Input File"),
        ("output", "Output File"),
    ]

    try:
        for dir_name, file_type_prefix in scan_dirs:
            target_dir = sandbox_path / dir_name
            if not target_dir.exists():
                continue

            for file_path in target_dir.rglob("*"):
                if file_path.is_file():
                    file_name = file_path.name
                    ext = file_path.suffix.lower()

                    if ext in supported_extensions or ext == "":
                        file_size = file_path.stat().st_size
                        rel_path = str(file_path.relative_to(target_dir))

                        files.append(
                            {
                                "name": file_name,
                                "path": str(file_path),
                                "relative_path": f"{dir_name}/{rel_path}",
                                "size": file_size,
                                "size_formatted": _format_file_size(file_size),
                                "type": supported_extensions.get(ext, "Unknown"),
                                "extension": ext,
                                "source": "local",
                                "category": dir_name,
                            }
                        )
    except Exception as e:
        print(f"[collect_sandbox_output_files] Error: {e}")

    return files


def _format_file_size(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _parse_file_size(size_str: str) -> int:
    try:
        size_str = size_str.strip().upper()
        if not size_str:
            return 0

        units = {
            "B": 1,
            "K": 1024,
            "KB": 1024,
            "M": 1024**2,
            "MB": 1024**2,
            "G": 1024**3,
            "GB": 1024**3,
            "T": 1024**4,
            "TB": 1024**4,
        }

        for unit, multiplier in units.items():
            if size_str.endswith(unit):
                num = float(size_str[: -len(unit)])
                return int(num * multiplier)

        return int(float(size_str))
    except (ValueError, TypeError):
        return 0


async def collect_files_from_new_sandbox(session_id: str) -> List[Dict[str, Any]]:
    """
    Create a temporary sandbox and collect output files from mounted directory
    """
    files = []

    try:
        try:
            from opensandbox.sandbox import Sandbox
            from opensandbox.config import ConnectionConfig
        except ImportError as e:
            print(
                f"[collect_files_from_new_sandbox] OpenSandbox SDK not available: {e}"
            )
            return files

        domain = os.getenv("SANDBOX_DOMAIN", "localhost:8080")
        api_key = os.getenv("SANDBOX_API_KEY") or os.getenv("OPEN_SANDBOX_API_KEY")

        connection_config = ConnectionConfig(
            domain=domain, api_key=api_key, debug=False
        )
        print(f"[collect_files_from_new_sandbox] Creating temporary sandbox...")

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
                    vb = {"hostPath": parts[0], "containerPath": parts[1]}
                    volume_bindings.append(vb)

        print(f"[collect_files_from_new_sandbox] Volume bindings: {volume_bindings}")

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

        import urllib.request
        import urllib.error

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
        print(f"[collect_files_from_new_sandbox] Sandbox created: {sandbox_id}")

        sandbox = await Sandbox.connect(sandbox_id, connection_config=connection_config)
        print(f"[collect_files_from_new_sandbox] Connected to sandbox")

        possible_paths = [
            (f"/data/sessions/{session_id}/input", "input"),
            (f"/data/sessions/{session_id}/output", "output"),
        ]

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

        for sandbox_internal_dir, category in possible_paths:
            print(f"[collect_files_from_new_sandbox] Checking: {sandbox_internal_dir}")

            try:
                ls_result = await sandbox.commands.run(
                    f"ls -la {sandbox_internal_dir} 2>&1 || echo 'NOT_FOUND'"
                )
                ls_stdout = _get_cmd_stdout(ls_result)
                print(f"[collect_files_from_new_sandbox] ls output: {ls_stdout[:500]}")

                cmd = f"find {sandbox_internal_dir} -type f 2>/dev/null || true"
                result = await sandbox.commands.run(cmd)

                stdout = _get_cmd_stdout(result)
                print(f"[collect_files_from_new_sandbox] find output: {stdout[:500]}")

                file_paths = [
                    p.strip() for p in stdout.strip().split("\n") if p.strip()
                ]

                if not file_paths:
                    continue

                print(
                    f"[collect_files_from_new_sandbox] Found {len(file_paths)} files in {category}"
                )

                for file_path in file_paths:
                    file_name = os.path.basename(file_path)
                    ext = os.path.splitext(file_name)[1].lower()

                    if ext in supported_extensions or ext == "":
                        size_cmd = f"stat -c %s {file_path} 2>/dev/null || stat -f %z {file_path} 2>/dev/null || echo 0"
                        size_result = await sandbox.commands.run(size_cmd)
                        size_stdout = getattr(size_result, "stdout", "0") or "0"
                        file_size = (
                            int(size_stdout.strip().split("\n")[0])
                            if size_stdout.strip()
                            else 0
                        )

                        rel_path = file_path.replace(sandbox_internal_dir + "/", "")

                        files.append(
                            {
                                "name": file_name,
                                "path": file_path,
                                "relative_path": f"{category}/{rel_path}",
                                "size": file_size,
                                "size_formatted": _format_file_size(file_size),
                                "type": supported_extensions.get(ext, "Unknown"),
                                "extension": ext,
                                "source": "remote",
                                "category": category,
                            }
                        )

            except Exception as e:
                print(f"[collect_files_from_new_sandbox] Error: {e}")
                continue

        if files:
            print(f"[collect_files_from_new_sandbox] Total files: {len(files)}")

        try:
            await sandbox.kill()
            print(f"[collect_files_from_new_sandbox] Sandbox killed")
        except Exception as e:
            print(f"[collect_files_from_new_sandbox] Error killing sandbox: {e}")

    except Exception as e:
        print(f"[collect_files_from_new_sandbox] Error: {e}")
        import traceback

        traceback.print_exc()

    return files


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

        summary = _generate_user_friendly_summary(result)
        response["summary"] = summary

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
            or merged_result.get("message", "Execution completed"),
            "task_type": "DEMO_MODE",
            "status": "completed",
        }
        response["answer"] = response["summary"]["answer"]

    output_files = []

    if hasattr(result, "merged_result") and result.merged_result:
        if "output_files" in result.merged_result:
            output_files = result.merged_result["output_files"]

    if not output_files and sandbox_dir:
        output_files = collect_sandbox_output_files(sandbox_dir)

    if output_files:
        response["output_files"] = output_files
        response["output_files_count"] = len(output_files)

    return response


def _generate_user_friendly_summary(result: Any) -> Dict[str, Any]:
    if not AGENT_AVAILABLE:
        return {"answer": "Demo mode", "task_type": "DEMO_MODE", "status": "completed"}

    try:
        from agent.state import UserTaskType
    except ImportError:
        UserTaskType = None

    task_type = result.user_task_type.value if result.user_task_type else "UNKNOWN"

    if UserTaskType and result.user_task_type == UserTaskType.GENERAL_QA:
        answer = result.merged_result.get("general_qa_answer", "")
        if answer:
            return {"answer": answer, "task_type": task_type, "status": "completed"}

    completed_count = len(result.completed_tasks) if result.completed_tasks else 0

    return {
        "answer": f"Task completed. {completed_count} subtasks processed.",
        "task_type": task_type,
        "status": "completed",
        "completed_tasks": completed_count,
    }
