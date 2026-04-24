"""
Mem0 Memory Manager - Based on immuneagent_memory implementation reference

Features:
1. Vectorize user input, search for similar questions (cache hit detection)
2. Store Immunity outputs (experiment plans, research results, etc.)
3. Store Todo-List execution results
4. Only store memories when all tasks are completed perfectly

Usage flow:
1. immunity_node starts -> query_similar_immunity() checks cache
2. If cache hit -> return cached result directly, skip subsequent steps
3. If cache miss -> continue executing immunity subgraph
4. After flow ends -> store_immunity_trace() saves successful results (only when all succeed)

Reference project: C:/Users/53966/xwechat_files/wxid_xsjtlj48c8no21_bda3/msg/file/2026-01/memory
"""

from typing import Optional, Dict, Any, List, Tuple
import asyncio
import logging
import hashlib
import json
import threading
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Keys that should preserve their string values (case-insensitive)
PATH_PRESERVING_SUFFIXES = ("_path", "_file", "_dir", "_url")
PATH_PRESERVING_EXACT = (
    "file",
    "path",
    "dir",
    "url",
    "filepath",
    "filename",
    "directory",
)


def _should_preserve_string(key: Optional[str]) -> bool:
    """Check if a key represents a path-like value that should be preserved."""
    if key is None:
        return False
    key_lower = key.lower()
    if key_lower in PATH_PRESERVING_EXACT:
        return True
    for suffix in PATH_PRESERVING_SUFFIXES:
        if key_lower.endswith(suffix):
            return True
    return False


def _is_file_path(text: str) -> bool:
    """
    Heuristic detection for file paths without explicit key names.

    Bioinformatics paths often appear as positional arguments:
    - /data/sequences/heavy_chain.fasta
    - ./results/alphafold_output.pdb

    Detect: starts with / or ./ AND has file extension in last 5 chars.
    """
    if not isinstance(text, str) or len(text) < 3:
        return False
    # Check for absolute or relative path prefix
    if not (text.startswith("/") or text.startswith("./")):
        return False
    # Check for file extension in last 5 characters
    last_5 = text[-5:]
    return "." in last_5


def sanitize_payload(
    data: Any, max_str_len: int = 200, max_list_len: int = 5, _key: Optional[str] = None
) -> Any:
    """
    Recursively truncate large strings and lists in tool calls.

    CRITICAL: Prevents Qdrant payload limit errors on immunological datasets.
    IMPORTANT: Preserves path-related values to maintain data provenance.

    Keys ending in _path, _file, _dir, _url are NOT truncated.
    """
    if isinstance(data, dict):
        return {
            k: sanitize_payload(v, max_str_len, max_list_len, _key=k)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        if len(data) > max_list_len:
            truncated = [
                sanitize_payload(x, max_str_len, max_list_len, _key=None)
                for x in data[:max_list_len]
            ]
            truncated.append(f"...<{len(data) - max_list_len} more items>...")
            return truncated
        return [sanitize_payload(x, max_str_len, max_list_len, _key=None) for x in data]
    elif isinstance(data, str):
        # Preserve path-related values regardless of length
        # Check 1: Key name indicates a path
        if _should_preserve_string(_key):
            return data
        # Check 2: Value looks like a file path (heuristic for positional args)
        if _is_file_path(data):
            return data
        if len(data) > max_str_len:
            return data[:max_str_len] + "...<truncated>"
        return data
    return data


def generate_input_hash(user_input: str) -> str:
    """Generate hash of user input"""
    normalized = user_input.strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


class ImmunityTrace(BaseModel):
    """Schema for immunity execution trace storage"""

    user_input: str = Field(description="Original user input")
    input_hash: str = Field(description="Input hash")
    query_summary: str = Field(description="Original user query (truncated)")

    # Immunity outputs
    optimized_questions: List[str] = Field(
        default_factory=list, description="Optimized query list"
    )
    research_summary: str = Field(
        default="", description="Research summary (truncated)"
    )
    hypothesis_summary: str = Field(
        default="", description="Hypothesis summary (truncated)"
    )
    final_enhanced_plan: str = Field(
        default="", description="Final enhanced plan (truncated)"
    )
    final_evaluation: str = Field(
        default="", description="Final evaluation (truncated)"
    )
    execution_plan: str = Field(default="", description="Execution plan (truncated)")

    # Todo-List summary
    todo_list_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Todo-List execution summary"
    )
    tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list, description="Sanitized tool calls"
    )
    tool_sequence: List[str] = Field(
        default_factory=list, description="Tool names in order"
    )

    # Status
    status: str = Field(description="success | partial | failed")
    output_paths: List[str] = Field(
        default_factory=list, description="File paths produced"
    )
    execution_time_seconds: float = Field(default=0.0)

    # Metadata
    session_id: str = Field(default="", description="Session ID")
    created_at: str = Field(default="", description="Creation time")


class ImmunityMemory:
    """
    Mem0-backed immunity memory for Bio-Agent.

    NOTE: Use get_memory_client() from singleton instead of instantiating
    directly to avoid connection churn.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = "immunity_agent"
        self._memory = None
        self._use_async = True
        self._init_lock = asyncio.Lock()

    async def _get_memory(self):
        """Lazy initialization with async/sync fallback. Thread-safe."""
        if self._memory is not None:
            return self._memory

        # First check if mem0 module is available
        try:
            import mem0
        except ImportError:
            logger.warning("mem0 module not installed, memory features disabled")
            self._memory = None
            self._use_async = False
            return None

        async with self._init_lock:
            if self._memory is not None:
                return self._memory

            try:
                from mem0 import AsyncMemory

                # AsyncMemory.from_config may return a coroutine, need to await
                config = {
                    "llm": self.config.get(
                        "llm",
                        {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
                    ),
                    "embedder": self.config.get(
                        "embedder",
                        {
                            "provider": "openai",
                            "config": {"model": "text-embedding-3-small"},
                        },
                    ),
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "collection_name": "immunity_traces",
                            "host": self.config.get("qdrant_host", "localhost"),
                            "port": self.config.get("qdrant_port", 6333),
                        },
                    },
                }
                # Check if from_config is a coroutine
                memory_instance = AsyncMemory.from_config(config)
                if hasattr(memory_instance, "__await__"):
                    # If it's a coroutine, await it
                    self._memory = await memory_instance
                else:
                    self._memory = memory_instance
                self._use_async = True
                logger.info("Initialized AsyncMemory client for immunity")
            except Exception as e:
                logger.warning(f"AsyncMemory failed, falling back to sync: {e}")
                try:
                    from mem0 import Memory

                    self._memory = Memory.from_config(
                        {
                            "llm": self.config.get(
                                "llm",
                                {
                                    "provider": "openai",
                                    "config": {"model": "gpt-4o-mini"},
                                },
                            ),
                            "embedder": self.config.get(
                                "embedder",
                                {
                                    "provider": "openai",
                                    "config": {"model": "text-embedding-3-small"},
                                },
                            ),
                            "vector_store": {
                                "provider": "qdrant",
                                "config": {
                                    "collection_name": "immunity_traces",
                                    "host": self.config.get("qdrant_host", "localhost"),
                                    "port": self.config.get("qdrant_port", 6333),
                                },
                            },
                        }
                    )
                    self._use_async = False
                    logger.info("Initialized sync Memory client (fallback)")
                except Exception as e2:
                    logger.error(f"Failed to initialize Memory (sync fallback): {e2}")
                    self._memory = None
                    self._use_async = False

        return self._memory

    async def query_similar_immunity(
        self, user_input: str, limit: int = 3, score_threshold: float = 0.90
    ) -> Tuple[bool, Optional[ImmunityTrace]]:
        """
        Query similar successful immunity execution records

        Args:
            user_input: User input
            limit: Maximum number of results to return
            score_threshold: Similarity threshold (default 0.90, requires high similarity)

        Returns:
            (is_cached, trace): Whether cache was hit, and cached content
        """
        memory = await self._get_memory()

        # If memory is not initialized (mem0 module unavailable), return miss directly
        if memory is None:
            logger.info("Memory not available, skipping cache check")
            return (False, None)

        # Generate input hash for exact matching
        input_hash = generate_input_hash(user_input)

        filters = {"metadata.status": "success"}

        try:
            if self._use_async:
                results = await memory.search(
                    query=user_input,
                    agent_id=self.agent_id,
                    filters=filters,
                    limit=limit,
                )
            else:
                results = await asyncio.to_thread(
                    memory.search,
                    query=user_input,
                    agent_id=self.agent_id,
                    filters=filters,
                    limit=limit,
                )

            all_results = results.get("results", [])

            # First try exact matching (via hash)
            for r in all_results:
                metadata = r.get("metadata", {})
                if metadata.get("input_hash") == input_hash:
                    logger.info(f"[Mem0] \u2705 Cache hit (exact match): {input_hash}")
                    trace = ImmunityTrace(**metadata)
                    return True, trace

            # Then try semantic similarity
            for r in all_results:
                score = r.get("score", 0)
                if score >= score_threshold:
                    logger.info(
                        f"[Mem0] \u2705 Cache hit (semantic match): similarity {score:.2f}"
                    )
                    metadata = r.get("metadata", {})
                    trace = ImmunityTrace(**metadata)
                    return True, trace

            logger.info(f"[Mem0] \u274c Cache miss: {input_hash}")
            return False, None

        except Exception as e:
            logger.error(f"[Mem0] Memory query failed: {e}")
            return False, None

    async def store_immunity_trace(self, trace: ImmunityTrace) -> str:
        """
        Store successful immunity execution record

        Args:
            trace: Immunity execution trace

        Returns:
            trace_id: Stored record ID
        """
        memory = await self._get_memory()

        # Clean data
        trace.tool_calls = [sanitize_payload(tc) for tc in trace.tool_calls]
        trace.tool_sequence = [
            tc.get("name", tc.get("tool_name", "unknown")) for tc in trace.tool_calls
        ]

        # Truncate long texts
        max_text_len = 1000
        if len(trace.research_summary) > max_text_len:
            trace.research_summary = (
                trace.research_summary[:max_text_len] + "...<truncated>"
            )
        if len(trace.hypothesis_summary) > max_text_len:
            trace.hypothesis_summary = (
                trace.hypothesis_summary[:max_text_len] + "...<truncated>"
            )
        if len(trace.final_enhanced_plan) > max_text_len:
            trace.final_enhanced_plan = (
                trace.final_enhanced_plan[:max_text_len] + "...<truncated>"
            )
        if len(trace.final_evaluation) > max_text_len:
            trace.final_evaluation = (
                trace.final_evaluation[:max_text_len] + "...<truncated>"
            )
        if len(trace.execution_plan) > max_text_len:
            trace.execution_plan = (
                trace.execution_plan[:max_text_len] + "...<truncated>"
            )

        try:
            if self._use_async:
                result = await memory.add(
                    messages=[
                        {
                            "role": "assistant",
                            "content": f"Immunity Analysis: {trace.query_summary}",
                        }
                    ],
                    agent_id=self.agent_id,
                    metadata=trace.model_dump(),
                    infer=False,
                )
            else:
                result = await asyncio.to_thread(
                    memory.add,
                    messages=[
                        {
                            "role": "assistant",
                            "content": f"Immunity Analysis: {trace.query_summary}",
                        }
                    ],
                    agent_id=self.agent_id,
                    metadata=trace.model_dump(),
                    infer=False,
                )

            trace_id = result.get("id", "")
            logger.info(f"[Mem0] Stored immunity trace: {trace_id}")
            return trace_id

        except Exception as e:
            logger.error(f"[Mem0] Memory store failed: {e}")
            return ""


# =============================================================================
# Singleton Pattern - Avoid Qdrant connection storms
# =============================================================================

_memory_instance: Optional[ImmunityMemory] = None
_memory_lock = threading.Lock()
_initialized_config_hash: Optional[int] = None


def _config_hash(config: Dict[str, Any]) -> int:
    """Create a stable hash of config for detecting config changes."""
    return hash(json.dumps(config, sort_keys=True, default=str))


def get_memory_client(config: Dict[str, Any] = None) -> ImmunityMemory:
    """
    Get or create the singleton ImmunityMemory instance.
    Thread-safe. If config changes, the singleton is recreated.
    """
    global _memory_instance, _initialized_config_hash

    # Default configuration
    if config is None:
        config = {
            "qdrant_host": "localhost",
            "qdrant_port": 6333,
        }

    new_hash = _config_hash(config)

    with _memory_lock:
        if _memory_instance is None or _initialized_config_hash != new_hash:
            logger.info("[Mem0] Initializing ImmunityMemory singleton")
            _memory_instance = ImmunityMemory(config)
            _initialized_config_hash = new_hash

        return _memory_instance


def reset_memory_client() -> None:
    """Reset the singleton instance. Useful for testing or config changes."""
    global _memory_instance, _initialized_config_hash

    with _memory_lock:
        if _memory_instance is not None:
            logger.info("[Mem0] Resetting ImmunityMemory singleton")
        _memory_instance = None
        _initialized_config_hash = None


# =============================================================================
# Convenience functions - Sync wrappers
# =============================================================================


def check_immunity_cache_sync(
    user_input: str, score_threshold: float = 0.90
) -> Tuple[bool, Optional[ImmunityTrace]]:
    """
    Sync convenience function: Check Immunity cache

    Args:
        user_input: User input
        score_threshold: Similarity threshold

    Returns:
        (is_cached, trace): Whether cache was hit, and cached content
    """
    memory = get_memory_client()

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, use thread pool
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    memory.query_similar_immunity(
                        user_input, score_threshold=score_threshold
                    ),
                )
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(
                memory.query_similar_immunity(
                    user_input, score_threshold=score_threshold
                )
            )
    except RuntimeError:
        # No event loop, create a new one
        return asyncio.run(
            memory.query_similar_immunity(user_input, score_threshold=score_threshold)
        )
    except Exception as e:
        logger.error(f"[Mem0] check_immunity_cache_sync failed: {e}")
        return False, None


def save_immunity_trace_sync(
    user_input: str,
    optimized_questions: List[str],
    research_summary: str,
    hypothesis_summary: str,
    final_enhanced_plan: str,
    final_evaluation: str,
    execution_plan: str,
    todo_list_summary: Dict[str, Any],
    tool_calls: List[Dict[str, Any]] = None,
    output_paths: List[str] = None,
    execution_time_seconds: float = 0.0,
    session_id: str = None,
    status: str = "success",
) -> str:
    """
    Sync convenience function: Save Immunity execution record

    Args:
        user_input: Original user input
        optimized_questions: List of optimized queries
        research_summary: Research summary
        hypothesis_summary: Hypothesis summary
        final_enhanced_plan: Final enhanced plan
        final_evaluation: Final evaluation
        execution_plan: Execution plan
        todo_list_summary: Todo-List summary
        tool_calls: Tool call list
        output_paths: Output file path list
        execution_time_seconds: Execution time (seconds)
        session_id: Session ID
        status: Status (success | partial | failed)

    Returns:
        trace_id: Stored record ID
    """
    memory = get_memory_client()

    trace = ImmunityTrace(
        user_input=user_input,
        input_hash=generate_input_hash(user_input),
        query_summary=user_input[:500],
        optimized_questions=optimized_questions,
        research_summary=research_summary,
        hypothesis_summary=hypothesis_summary,
        final_enhanced_plan=final_enhanced_plan,
        final_evaluation=final_evaluation,
        execution_plan=execution_plan,
        todo_list_summary=todo_list_summary,
        tool_calls=tool_calls or [],
        tool_sequence=[],
        status=status,
        output_paths=output_paths or [],
        execution_time_seconds=execution_time_seconds,
        session_id=session_id or "",
        created_at=datetime.now().isoformat(),
    )

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, memory.store_immunity_trace(trace)
                )
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(memory.store_immunity_trace(trace))
    except RuntimeError:
        return asyncio.run(memory.store_immunity_trace(trace))
    except Exception as e:
        logger.error(f"[Mem0] save_immunity_trace_sync failed: {e}")
        return ""


def check_all_tasks_completed_successfully(
    merged_result: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check whether all tasks have completed perfectly

    Args:
        merged_result: Merged results (containing executor_results and result_evaluation)

    Returns:
        (all_success, summary): Whether all succeeded, and summary info
    """
    summary = {
        "total_tasks": 0,
        "completed_tasks": 0,
        "failed_tasks": 0,
        "success_rate": 0.0,
        "is_perfect": False,
    }

    # Get task statistics from executor_results
    executor_results = merged_result.get("executor_results", {})
    if executor_results:
        summary["total_tasks"] = executor_results.get("total_tasks", 0)
        summary["completed_tasks"] = executor_results.get("completed_count", 0)
        summary["failed_tasks"] = executor_results.get("failed_count", 0)

    # Calculate success rate
    if summary["total_tasks"] > 0:
        summary["success_rate"] = summary["completed_tasks"] / summary["total_tasks"]

    # Determine if all tasks completed perfectly
    summary["is_perfect"] = (
        summary["total_tasks"] > 0
        and summary["failed_tasks"] == 0
        and summary["success_rate"] >= 1.0
    )

    return summary["is_perfect"], summary
