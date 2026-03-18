"""
File-based Checkpointer for LangGraph

Implements persistent storage using SqliteSaver with per-session database files.
Sessions survive server restarts and support resumption.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import threading
import asyncio

try:
    # 优先使用 MemorySaver，因为它使用 pickle 可以序列化函数
    # SqliteSaver 使用 msgpack 无法序列化函数（如 progress_callback）
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.base import CheckpointTuple
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    # 创建带有pickle_fallback的序列化器，可以序列化函数对象
    serde = JsonPlusSerializer(pickle_fallback=True)

    # 将 MemorySaver 映射到 SqliteSaver 变量名，保持兼容性
    def SqliteSaver():
        return MemorySaver(serde=serde)

    SQLITE_SAVER_AVAILABLE = True
    print(
        "[Checkpointer] Using MemorySaver with pickle_fallback (supports function serialization)"
    )
except ImportError:
    # 如果 MemorySaver 不可用，尝试 SqliteSaver
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.checkpoint.base import CheckpointTuple

        SQLITE_SAVER_AVAILABLE = True
        print("[Checkpointer] Using SqliteSaver (msgpack, cannot serialize functions)")
    except ImportError:
        SQLITE_SAVER_AVAILABLE = False
        SqliteSaver = None
        CheckpointTuple = None
        print("[Checkpointer] No checkpoint saver available")


class SessionCheckpointer:
    """
    File-based checkpointer with per-session SQLite storage.

    Features:
    - Independent database per session (avoid concurrency issues)
    - Automatic directory creation
    - Session resumption support
    - Cleanup of expired sessions
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, base_dir: Optional[str] = None):
        if self._initialized:
            return
        self._initialized = True

        self.base_dir = Path(
            base_dir or os.path.join(os.path.dirname(__file__), "data", "checkpoints")
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._savers: Dict[str, Any] = {}
        self._saver_locks: Dict[str, threading.Lock] = {}

        print(f"[Checkpointer] Initialized with base dir: {self.base_dir}")

    def _get_session_db_path(self, session_id: str) -> Path:
        """Get the database file path for a session"""
        return self.base_dir / session_id / "checkpoint.db"

    def _get_lock(self, session_id: str) -> threading.Lock:
        """Get or create a lock for a session"""
        if session_id not in self._saver_locks:
            self._saver_locks[session_id] = threading.Lock()
        return self._saver_locks[session_id]

    def get_saver(self, session_id: str) -> Optional[Any]:
        """
        Get or create a checkpoint saver for a session.

        Args:
            session_id: Session ID

        Returns:
            CheckpointSaver instance (MemorySaver or SqliteSaver) or None if not available
        """
        if not SQLITE_SAVER_AVAILABLE:
            print("[Checkpointer] No checkpoint saver available")
            return None

        lock = self._get_lock(session_id)

        with lock:
            if session_id in self._savers:
                return self._savers[session_id]

            try:
                # 使用 MemorySaver（支持函数序列化）
                saver = SqliteSaver()
                print(f"[Checkpointer] Created MemorySaver for session: {session_id}")

                self._savers[session_id] = saver
                return saver

            except Exception as e:
                print(f"[Checkpointer] Error creating saver: {e}")
                return None

    def save_state(
        self, session_id: str, state: Dict[str, Any], thread_id: str = "default"
    ) -> bool:
        """
        Save state to checkpoint.

        Args:
            session_id: Session ID
            state: State dictionary to save
            thread_id: Thread ID for checkpoint

        Returns:
            True if saved successfully
        """
        lock = self._get_lock(session_id)

        with lock:
            try:
                checkpoint_file = (
                    self._get_session_db_path(session_id).parent / "state.json"
                )
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

                checkpoint_data = {
                    "session_id": session_id,
                    "thread_id": thread_id,
                    "state": state,
                    "timestamp": datetime.now().isoformat(),
                }

                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(
                        checkpoint_data, f, ensure_ascii=False, indent=2, default=str
                    )

                print(f"[Checkpointer] Saved state for session: {session_id}")
                return True

            except Exception as e:
                print(f"[Checkpointer] Error saving state: {e}")
                return False

    def load_state(
        self, session_id: str, thread_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """
        Load state from checkpoint.

        Args:
            session_id: Session ID
            thread_id: Thread ID for checkpoint

        Returns:
            State dictionary or None if not found
        """
        lock = self._get_lock(session_id)

        with lock:
            try:
                checkpoint_file = (
                    self._get_session_db_path(session_id).parent / "state.json"
                )

                if not checkpoint_file.exists():
                    return None

                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)

                print(f"[Checkpointer] Loaded state for session: {session_id}")
                return checkpoint_data.get("state")

            except Exception as e:
                print(f"[Checkpointer] Error loading state: {e}")
                return None

    def save_hitl_state(
        self,
        session_id: str,
        hitl_request: Optional[Dict[str, Any]] = None,
        hitl_response: Optional[Dict[str, Any]] = None,
        task_md_content: Optional[str] = None,
        missing_parameters: Optional[List[Dict[str, Any]]] = None,
        clear_hitl_request: bool = False,
    ):
        """
        Save HITL-specific state for session resumption.

        Args:
            session_id: Session ID
            hitl_request: HITL request data
            hitl_response: HITL response from user
            task_md_content: Generated task.md content
            missing_parameters: List of missing parameters
            clear_hitl_request: If True, clear the hitl_request field

        Returns:
            True if saved successfully
        """
        lock = self._get_lock(session_id)

        with lock:
            try:
                hitl_file = (
                    self._get_session_db_path(session_id).parent / "hitl_state.json"
                )
                hitl_file.parent.mkdir(parents=True, exist_ok=True)

                existing_data = {}
                if hitl_file.exists():
                    try:
                        with open(hitl_file, "r", encoding="utf-8") as f:
                            existing_data = json.load(f)
                    except Exception:
                        pass

                if hitl_request is not None:
                    existing_data["hitl_request"] = hitl_request
                if hitl_response is not None:
                    existing_data["hitl_response"] = hitl_response
                if task_md_content is not None:
                    existing_data["task_md_content"] = task_md_content
                if missing_parameters is not None:
                    existing_data["missing_parameters"] = missing_parameters

                if clear_hitl_request:
                    existing_data.pop("hitl_request", None)

                existing_data["updated_at"] = datetime.now().isoformat()

                with open(hitl_file, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)

                print(f"[Checkpointer] Saved HITL state for session: {session_id}")
                return True

            except Exception as e:
                print(f"[Checkpointer] Error saving HITL state: {e}")
                return False

    def load_hitl_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load HITL-specific state.

        Args:
            session_id: Session ID

        Returns:
            HITL state dictionary or None if not found
        """
        lock = self._get_lock(session_id)

        with lock:
            try:
                hitl_file = (
                    self._get_session_db_path(session_id).parent / "hitl_state.json"
                )

                if not hitl_file.exists():
                    return None

                with open(hitl_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                print(f"[Checkpointer] Loaded HITL state for session: {session_id}")
                return data

            except Exception as e:
                print(f"[Checkpointer] Error loading HITL state: {e}")
                return None

    def clear_hitl_state(self, session_id: str) -> bool:
        """
        Clear HITL state after confirmation.

        Args:
            session_id: Session ID

        Returns:
            True if cleared successfully
        """
        lock = self._get_lock(session_id)

        with lock:
            try:
                hitl_file = (
                    self._get_session_db_path(session_id).parent / "hitl_state.json"
                )

                if hitl_file.exists():
                    hitl_file.unlink()
                    print(
                        f"[Checkpointer] Cleared HITL state for session: {session_id}"
                    )

                return True

            except Exception as e:
                print(f"[Checkpointer] Error clearing HITL state: {e}")
                return False

    def session_exists(self, session_id: str) -> bool:
        """Check if a session has checkpoint data"""
        checkpoint_dir = self._get_session_db_path(session_id).parent
        return checkpoint_dir.exists()

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions with checkpoint data.

        Returns:
            List of session info dicts
        """
        sessions = []

        try:
            for session_dir in self.base_dir.iterdir():
                if session_dir.is_dir():
                    state_file = session_dir / "state.json"
                    hitl_file = session_dir / "hitl_state.json"

                    session_info = {
                        "session_id": session_dir.name,
                        "has_checkpoint": state_file.exists(),
                        "has_hitl_state": hitl_file.exists(),
                    }

                    if state_file.exists():
                        try:
                            stat = state_file.stat()
                            session_info["checkpoint_time"] = datetime.fromtimestamp(
                                stat.st_mtime
                            ).isoformat()
                        except Exception:
                            pass

                    if hitl_file.exists():
                        try:
                            with open(hitl_file, "r", encoding="utf-8") as f:
                                hitl_data = json.load(f)
                            session_info["hitl_waiting"] = (
                                hitl_data.get("hitl_request") is not None
                            )
                            session_info["hitl_updated"] = hitl_data.get("updated_at")
                        except Exception:
                            pass

                    sessions.append(session_info)

        except Exception as e:
            print(f"[Checkpointer] Error listing sessions: {e}")

        return sessions

    def cleanup_session(self, session_id: str) -> bool:
        """
        Clean up all checkpoint data for a session.

        Args:
            session_id: Session ID

        Returns:
            True if cleaned up successfully
        """
        lock = self._get_lock(session_id)

        with lock:
            try:
                session_dir = self._get_session_db_path(session_id).parent

                if session_dir.exists():
                    shutil.rmtree(session_dir)

                if session_id in self._savers:
                    del self._savers[session_id]

                if session_id in self._saver_locks:
                    del self._saver_locks[session_id]

                print(f"[Checkpointer] Cleaned up session: {session_id}")
                return True

            except Exception as e:
                print(f"[Checkpointer] Error cleaning up session: {e}")
                return False

    def cleanup_expired_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up sessions older than specified age.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of sessions cleaned up
        """
        cleaned = 0
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)

        try:
            for session_dir in self.base_dir.iterdir():
                if session_dir.is_dir():
                    try:
                        stat = session_dir.stat()
                        if stat.st_mtime < cutoff:
                            session_id = session_dir.name
                            if self.cleanup_session(session_id):
                                cleaned += 1
                    except Exception:
                        continue

        except Exception as e:
            print(f"[Checkpointer] Error cleaning up expired sessions: {e}")

        print(f"[Checkpointer] Cleaned up {cleaned} expired sessions")
        return cleaned


_checkpointer_instance: Optional[SessionCheckpointer] = None


def get_checkpointer() -> SessionCheckpointer:
    """Get the global SessionCheckpointer instance"""
    global _checkpointer_instance
    if _checkpointer_instance is None:
        _checkpointer_instance = SessionCheckpointer()
    return _checkpointer_instance
