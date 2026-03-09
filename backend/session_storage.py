"""
Session Storage - Persistent storage for session messages
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import threading


class SessionStorage:
    """Persistent storage for session messages"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.storage_dir = Path(__file__).parent / "data" / "sessions"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._file_locks: Dict[str, threading.Lock] = {}
        print(f"[SessionStorage] Initialized with storage dir: {self.storage_dir}")

    def _get_session_file(self, session_id: str) -> Path:
        """Get the storage file path for a session"""
        return self.storage_dir / session_id / "messages.json"

    def _get_lock(self, session_id: str) -> threading.Lock:
        """Get or create a lock for a session"""
        if session_id not in self._file_locks:
            self._file_locks[session_id] = threading.Lock()
        return self._file_locks[session_id]

    def save_message(self, session_id: str, message: Dict[str, Any]) -> bool:
        """
        Save a message to session storage

        Args:
            session_id: Session ID
            message: Message data (should be JSON serializable)

        Returns:
            True if saved successfully
        """
        lock = self._get_lock(session_id)

        with lock:
            try:
                session_file = self._get_session_file(session_id)
                session_file.parent.mkdir(parents=True, exist_ok=True)

                # Load existing messages
                messages = []
                if session_file.exists():
                    try:
                        with open(session_file, "r", encoding="utf-8") as f:
                            messages = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        messages = []

                # Append new message
                messages.append(message)

                # Save back to file
                with open(session_file, "w", encoding="utf-8") as f:
                    json.dump(messages, f, ensure_ascii=False, indent=2)

                return True

            except Exception as e:
                print(f"[SessionStorage] Error saving message: {e}")
                return False

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages for a session

        Args:
            session_id: Session ID

        Returns:
            List of messages
        """
        lock = self._get_lock(session_id)

        with lock:
            try:
                session_file = self._get_session_file(session_id)

                if not session_file.exists():
                    return []

                with open(session_file, "r", encoding="utf-8") as f:
                    return json.load(f)

            except Exception as e:
                print(f"[SessionStorage] Error loading messages: {e}")
                return []

    def get_session_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all sessions with metadata

        Returns:
            List of session info dicts
        """
        sessions = []

        try:
            for session_dir in self.storage_dir.iterdir():
                if session_dir.is_dir():
                    session_file = session_dir / "messages.json"
                    if session_file.exists():
                        try:
                            with open(session_file, "r", encoding="utf-8") as f:
                                messages = json.load(f)

                            if messages:
                                first_msg = messages[0]
                                last_msg = messages[-1]

                                sessions.append(
                                    {
                                        "session_id": session_dir.name,
                                        "message_count": len(messages),
                                        "created_at": first_msg.get("timestamp", ""),
                                        "updated_at": last_msg.get("timestamp", ""),
                                        "first_message": first_msg.get("message", "")[
                                            :100
                                        ],
                                    }
                                )
                        except (json.JSONDecodeError, IOError):
                            continue
        except Exception as e:
            print(f"[SessionStorage] Error listing sessions: {e}")

        # Sort by updated_at descending
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its messages

        Args:
            session_id: Session ID

        Returns:
            True if deleted successfully
        """
        lock = self._get_lock(session_id)

        with lock:
            try:
                import shutil

                session_dir = self._get_session_file(session_id).parent

                if session_dir.exists():
                    shutil.rmtree(session_dir)

                # Clean up lock
                if session_id in self._file_locks:
                    del self._file_locks[session_id]

                return True

            except Exception as e:
                print(f"[SessionStorage] Error deleting session: {e}")
                return False


# Global singleton instance
_storage_instance: Optional[SessionStorage] = None


def get_session_storage() -> SessionStorage:
    """Get the global SessionStorage instance"""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = SessionStorage()
    return _storage_instance
