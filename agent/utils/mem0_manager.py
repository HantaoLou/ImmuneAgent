"""
Mem0 记忆管理器 - 基于 immuneagent_memory 实现参考

功能:
1. 向量化用户输入，搜索相似问题（缓存命中检测）
2. 存储 Immunity 产出（实验计划、研究结果等）
3. 存储 Todo-List 执行结果
4. 仅在所有任务完美完成时才存储记忆

使用流程:
1. immunity_node 开始时 → query_similar_immunity() 检查缓存
2. 如果缓存命中 → 直接返回缓存结果，跳过后续步骤
3. 如果缓存未命中 → 继续执行 immunity 子图
4. 流程结束后 → store_immunity_trace() 保存成功结果（仅在全部成功时）

参考项目: C:/Users/53966/xwechat_files/wxid_xsjtlj48c8no21_bda3/msg/file/2026-01/memory
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
    """生成用户输入的哈希值"""
    normalized = user_input.strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


class ImmunityTrace(BaseModel):
    """Schema for immunity execution trace storage"""

    user_input: str = Field(description="用户原始输入")
    input_hash: str = Field(description="输入哈希")
    query_summary: str = Field(description="Original user query (truncated)")

    # Immunity 产出
    optimized_questions: List[str] = Field(
        default_factory=list, description="优化查询列表"
    )
    research_summary: str = Field(default="", description="研究摘要 (truncated)")
    hypothesis_summary: str = Field(default="", description="假设摘要 (truncated)")
    final_enhanced_plan: str = Field(default="", description="最终增强计划 (truncated)")
    final_evaluation: str = Field(default="", description="最终评估 (truncated)")
    execution_plan: str = Field(default="", description="执行计划 (truncated)")

    # Todo-List 摘要
    todo_list_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Todo-List 执行摘要"
    )
    tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list, description="Sanitized tool calls"
    )
    tool_sequence: List[str] = Field(
        default_factory=list, description="Tool names in order"
    )

    # 状态
    status: str = Field(description="success | partial | failed")
    output_paths: List[str] = Field(
        default_factory=list, description="File paths produced"
    )
    execution_time_seconds: float = Field(default=0.0)

    # 元数据
    session_id: str = Field(default="", description="会话 ID")
    created_at: str = Field(default="", description="创建时间")


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

        # 首先检查 mem0 模块是否可用
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

                # AsyncMemory.from_config 可能返回协程，需要 await
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
                # 检查 from_config 是否是协程
                memory_instance = AsyncMemory.from_config(config)
                if hasattr(memory_instance, "__await__"):
                    # 如果是协程，await 它
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
        查询相似的成功 immunity 执行记录

        Args:
            user_input: 用户输入
            limit: 返回结果数量限制
            score_threshold: 相似度阈值（默认 0.90，要求高度相似）

        Returns:
            (is_cached, trace): 是否命中缓存，以及缓存内容
        """
        memory = await self._get_memory()

        # 如果 memory 未初始化（mem0 模块不可用），直接返回未命中
        if memory is None:
            logger.info("Memory not available, skipping cache check")
            return (False, None)

        # 生成输入哈希用于精确匹配
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

            # 首先尝试精确匹配（通过 hash）
            for r in all_results:
                metadata = r.get("metadata", {})
                if metadata.get("input_hash") == input_hash:
                    logger.info(f"[Mem0] ✅ 缓存命中（精确匹配）: {input_hash}")
                    trace = ImmunityTrace(**metadata)
                    return True, trace

            # 然后尝试语义相似性
            for r in all_results:
                score = r.get("score", 0)
                if score >= score_threshold:
                    logger.info(f"[Mem0] ✅ 缓存命中（语义匹配）: 相似度 {score:.2f}")
                    metadata = r.get("metadata", {})
                    trace = ImmunityTrace(**metadata)
                    return True, trace

            logger.info(f"[Mem0] ❌ 缓存未命中: {input_hash}")
            return False, None

        except Exception as e:
            logger.error(f"[Mem0] Memory query failed: {e}")
            return False, None

    async def store_immunity_trace(self, trace: ImmunityTrace) -> str:
        """
        存储成功的 immunity 执行记录

        Args:
            trace: Immunity 执行轨迹

        Returns:
            trace_id: 存储的记录 ID
        """
        memory = await self._get_memory()

        # 清理数据
        trace.tool_calls = [sanitize_payload(tc) for tc in trace.tool_calls]
        trace.tool_sequence = [
            tc.get("name", tc.get("tool_name", "unknown")) for tc in trace.tool_calls
        ]

        # 截断长文本
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
# Singleton Pattern - 避免 Qdrant 连接风暴
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

    # 默认配置
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
# 便捷函数 - 同步包装器
# =============================================================================


def check_immunity_cache_sync(
    user_input: str, score_threshold: float = 0.90
) -> Tuple[bool, Optional[ImmunityTrace]]:
    """
    同步便捷函数：检查 Immunity 缓存

    Args:
        user_input: 用户输入
        score_threshold: 相似度阈值

    Returns:
        (is_cached, trace): 是否命中缓存，以及缓存内容
    """
    memory = get_memory_client()

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果已经在异步上下文中，使用线程池
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
        # 没有事件循环，创建一个新的
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
    同步便捷函数：保存 Immunity 执行记录

    Args:
        user_input: 用户原始输入
        optimized_questions: 优化后的查询列表
        research_summary: 研究摘要
        hypothesis_summary: 假设摘要
        final_enhanced_plan: 最终增强计划
        final_evaluation: 最终评估
        execution_plan: 执行计划
        todo_list_summary: Todo-List 摘要
        tool_calls: 工具调用列表
        output_paths: 输出文件路径列表
        execution_time_seconds: 执行时间（秒）
        session_id: 会话 ID
        status: 状态 (success | partial | failed)

    Returns:
        trace_id: 存储的记录 ID
    """
    memory = get_memory_client()

    trace = ImmunityTrace(
        user_input=user_input,
        input_hash=generate_input_hash(user_input),
        query_summary=user_input[:500],  # 截断
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
    检查所有任务是否都完美完成

    Args:
        merged_result: 合并结果（包含 executor_results 和 result_evaluation）

    Returns:
        (all_success, summary): 是否全部成功，以及摘要信息
    """
    summary = {
        "total_tasks": 0,
        "completed_tasks": 0,
        "failed_tasks": 0,
        "success_rate": 0.0,
        "is_perfect": False,
    }

    # 从 executor_results 获取任务统计
    executor_results = merged_result.get("executor_results", {})
    if executor_results:
        summary["total_tasks"] = executor_results.get("total_tasks", 0)
        summary["completed_tasks"] = executor_results.get("completed_count", 0)
        summary["failed_tasks"] = executor_results.get("failed_count", 0)

    # 计算成功率
    if summary["total_tasks"] > 0:
        summary["success_rate"] = summary["completed_tasks"] / summary["total_tasks"]

    # 判断是否完美完成
    summary["is_perfect"] = (
        summary["total_tasks"] > 0
        and summary["failed_tasks"] == 0
        and summary["success_rate"] >= 1.0
    )

    return summary["is_perfect"], summary
