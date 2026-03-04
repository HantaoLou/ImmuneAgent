"""General-purpose key-value store for agent state persistence.

Extends existing cache pattern (code_cache_manager.py, paperqa_cache.py) with:
    - sqlite3 WAL mode for concurrent multi-agent safety
    - TTL support with auto-cleanup
    - Namespace isolation per sub-agent

DB location: agent/.cache/agent_memory.db
Schema: (namespace TEXT, key TEXT, value TEXT, created_at REAL, expires_at REAL)

LangChain 1.0+ Compatibility:
    - Uses @tool decorator from langchain_core.tools
    - Can be directly bound to LLM via .bind_tools()
"""

import logging
import sqlite3
import time
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# DB lives alongside existing caches (code_cache_manager.py uses agent/.cache/)
_DB_DIR = Path(__file__).parent.parent / ".cache"
_DB_PATH = _DB_DIR / "agent_memory.db"
_conn = None


def _get_conn() -> sqlite3.Connection:
    """Get or create a thread-safe sqlite3 connection with WAL mode."""
    global _conn
    if _conn is not None:
        return _conn
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute(
        """CREATE TABLE IF NOT EXISTS kv (
            namespace TEXT NOT NULL,
            key       TEXT NOT NULL,
            value     TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL,
            PRIMARY KEY (namespace, key)
        )"""
    )
    _conn.commit()
    return _conn


def _cleanup_expired(conn: sqlite3.Connection) -> None:
    """Remove expired entries."""
    conn.execute(
        "DELETE FROM kv WHERE expires_at IS NOT NULL AND expires_at < ?",
        (time.time(),),
    )
    conn.commit()


@tool
def agent_memory(
    action: str,
    key: str = None,
    value: str = None,
    namespace: str = "default",
    ttl_seconds: int = 0,
) -> str:
    """General-purpose key-value store for agent state persistence.

    Args:
        action: "set", "get", "list", "delete", "clear"
        key: Key name (required for set/get/delete)
        value: Value to store (required for set)
        namespace: Namespace for isolation (default: "default")
        ttl_seconds: Time-to-live in seconds (0 = no expiry)

    Returns:
        Formatted result string
    """
    try:
        conn = _get_conn()
        _cleanup_expired(conn)

        if action == "set":
            if not key or value is None:
                return "[agent_memory] Error: 'set' requires both key and value."
            now = time.time()
            expires = now + ttl_seconds if ttl_seconds > 0 else None
            conn.execute(
                "INSERT OR REPLACE INTO kv (namespace, key, value, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
                (namespace, key, value, now, expires),
            )
            conn.commit()
            ttl_msg = f" (TTL: {ttl_seconds}s)" if ttl_seconds > 0 else ""
            return f"[agent_memory] Stored '{key}' in namespace '{namespace}'{ttl_msg}."

        elif action == "get":
            if not key:
                return "[agent_memory] Error: 'get' requires a key."
            row = conn.execute(
                "SELECT value, created_at, expires_at FROM kv WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
            if not row:
                return f"[agent_memory] Key '{key}' not found in namespace '{namespace}'."
            return f"[agent_memory] {key} = {row[0]}"

        elif action == "list":
            rows = conn.execute(
                "SELECT key, LENGTH(value), created_at FROM kv WHERE namespace = ? ORDER BY created_at DESC",
                (namespace,),
            ).fetchall()
            if not rows:
                return f"[agent_memory] Namespace '{namespace}' is empty."
            lines = [f"[agent_memory] {len(rows)} keys in namespace '{namespace}':"]
            for k, vlen, ts in rows:
                lines.append(f"  {k} ({vlen} chars)")
            return "\n".join(lines)

        elif action == "delete":
            if not key:
                return "[agent_memory] Error: 'delete' requires a key."
            deleted = conn.execute(
                "DELETE FROM kv WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).rowcount
            conn.commit()
            if deleted:
                return f"[agent_memory] Deleted '{key}' from namespace '{namespace}'."
            return f"[agent_memory] Key '{key}' not found in namespace '{namespace}'."

        elif action == "clear":
            deleted = conn.execute(
                "DELETE FROM kv WHERE namespace = ?",
                (namespace,),
            ).rowcount
            conn.commit()
            return f"[agent_memory] Cleared namespace '{namespace}' ({deleted} keys removed)."

        else:
            return f"[agent_memory] Unknown action '{action}'. Use: set, get, list, delete, clear."

    except Exception as e:
        logger.error(f"agent_memory error: {e}")
        return f"[agent_memory] Error: {e}"


def get_memory_tools() -> list:
    """Return memory tools as LangChain tools.
    
    Returns:
        List of LangChain tool objects that can be directly bound to LLM.
    """
    return [agent_memory]


def get_memory_tools_dict() -> dict:
    """Return memory tools for backward compatibility (namespace injection)."""
    return {"agent_memory": agent_memory}
