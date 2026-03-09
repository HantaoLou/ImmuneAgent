"""
Immunity Agent Subgraph

Complete workflow:
Query Decomposition → Retrieval → Deep Research → Hypothesis Generation → Planning ⭐ → Evaluation

Reference implementation: antibody_gen/agent/usecases/immunity
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import re
import os
import time
import asyncio
import sys
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from state import GlobalState
from utils.llm_factory import (
    create_reasoning_advanced_llm,
    create_reasoning_llm,
    create_bioinformatics_llm,
)
from .state import ImmunityState
from .prompts import ImmunityPrompts

# Mem0 记忆管理
try:
    from utils.mem0_manager import (
        get_memory_client,
        check_immunity_cache_sync,
        generate_input_hash,
    )

    MEM0_AVAILABLE = True
except ImportError as e:
    MEM0_AVAILABLE = False
    print(f"[Immunity] 警告: mem0_manager 不可用: {e}")

# Import search tools for LLM binding
from tools.search import web_search, knowledge_search, read_webpage

# Import deep_research subgraph
from nodes.subagents.deep_research.deep_researcher import (
    deep_researcher,
    get_default_config as get_deep_research_config,
)

# Add agent directory to path
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))


# ===================== Progress Logging Utilities =====================


class ImmunityProgressLogger:
    """
    进度日志记录器 - 提供详细的进度输出和时间统计
    """

    # 阶段定义
    STAGES = [
        ("query_decomposition", "📝 Query Decomposition", 1),
        ("retrieval", "📚 Immunology Retrieval", 2),
        ("deep_research", "🔬 Deep Research Analysis", 3),
        ("hypothesis_generation", "🧬 Hypothesis Generation", 4),
        ("planning", "🔧 Plan Generation", 5),
        ("evaluation", "📊 Plan Evaluation", 6),
    ]
    TOTAL_STAGES = 6

    def __init__(self):
        self.stage_start_times: Dict[str, float] = {}
        self.workflow_start_time: float = 0
        self.current_stage: str = ""

    def _timestamp(self) -> str:
        """获取当前时间戳"""
        return datetime.now().strftime("%H:%M:%S")

    def _flush_print(self, msg: str):
        """强制刷新输出"""
        print(msg)
        sys.stdout.flush()

    def start_workflow(self):
        """开始工作流"""
        self.workflow_start_time = time.perf_counter()
        self._flush_print("\n" + "=" * 80)
        self._flush_print(f"[{self._timestamp()}] 🚀 IMMUNITY SUBGRAPH 工作流开始")
        self._flush_print("=" * 80)

    def end_workflow(self, success: bool = True):
        """结束工作流"""
        total_time = time.perf_counter() - self.workflow_start_time
        status = "✅ 成功完成" if success else "❌ 失败"
        self._flush_print("\n" + "=" * 80)
        self._flush_print(f"[{self._timestamp()}] {status}")
        self._flush_print(f"  总耗时: {total_time:.2f} 秒 ({total_time / 60:.1f} 分钟)")
        self._flush_print("=" * 80)

    def start_stage(self, stage_name: str, description: str = ""):
        """开始一个阶段"""
        self.current_stage = stage_name
        self.stage_start_times[stage_name] = time.perf_counter()

        # 查找阶段编号
        stage_num = 0
        for name, _, num in self.STAGES:
            if name == stage_name:
                stage_num = num
                break

        progress_pct = (stage_num / self.TOTAL_STAGES) * 100

        self._flush_print("\n" + "-" * 80)
        self._flush_print(
            f"[{self._timestamp()}] 🔄 STAGE {stage_num}/{self.TOTAL_STAGES} ({progress_pct:.0f}%): {stage_name}"
        )
        if description:
            self._flush_print(f"  📋 {description}")
        self._flush_print("-" * 80)

    def end_stage(self, stage_name: str, success: bool = True, details: str = ""):
        """结束一个阶段"""
        elapsed = time.perf_counter() - self.stage_start_times.get(
            stage_name, time.perf_counter()
        )

        # 查找阶段编号
        stage_num = 0
        for name, _, num in self.STAGES:
            if name == stage_name:
                stage_num = num
                break

        progress_pct = (stage_num / self.TOTAL_STAGES) * 100
        status = "✅" if success else "❌"

        self._flush_print(
            f"[{self._timestamp()}] {status} STAGE {stage_num}/{self.TOTAL_STAGES} 完成 ({progress_pct:.0f}%)"
        )
        self._flush_print(f"  ⏱️ 阶段耗时: {elapsed:.2f} 秒")
        if details:
            self._flush_print(f"  📊 {details}")

    def log_llm_start(self, model_info: dict, prompt_len: int):
        """记录 LLM 调用开始"""
        self._flush_print(f"[{self._timestamp()}] 🤖 开始 LLM 调用...")
        self._flush_print(f"  📦 模型: {model_info.get('model', 'unknown')}")
        self._flush_print(f"  🌡️ 温度: {model_info.get('temperature', 'N/A')}")
        self._flush_print(f"  ⏰ 超时: {model_info.get('timeout', 'N/A')}s")
        self._flush_print(f"  📝 Prompt 长度: {prompt_len} 字符")
        self._flush_print(f"  ⏳ 等待响应中...")

    def log_llm_end(self, elapsed: float, response_len: int = 0, success: bool = True):
        """记录 LLM 调用结束"""
        status = "✅" if success else "❌"
        self._flush_print(f"[{self._timestamp()}] {status} LLM 调用完成")
        self._flush_print(f"  ⏱️ 响应时间: {elapsed:.2f} 秒")
        if response_len > 0:
            self._flush_print(f"  📤 响应长度: {response_len} 字符")

    def log_info(self, message: str):
        """记录信息"""
        self._flush_print(f"[{self._timestamp()}] ℹ️ {message}")

    def log_warning(self, message: str):
        """记录警告"""
        self._flush_print(f"[{self._timestamp()}] ⚠️ {message}")

    def log_error(self, message: str, error: Exception = None):
        """记录错误"""
        self._flush_print(f"[{self._timestamp()}] ❌ {message}")
        if error:
            self._flush_print(f"  错误详情: {type(error).__name__}: {str(error)}")


# 全局进度记录器实例
_progress_logger = ImmunityProgressLogger()


# ===================== Helper Functions =====================


def _get_llm_with_callback(state, purpose="bioinformatics"):
    """
    创建带progress_callback的LLM实例

    Args:
        state: ImmunityState实例
        purpose: LLM用途（"bioinformatics", "reasoning", "code"等）

    Returns:
        LLM实例（带或不带progress_callback）
    """
    # 🔥 获取progress_callback（优先从state，其次从parent_state）
    progress_callback = None
    if hasattr(state, "progress_callback") and state.progress_callback:
        progress_callback = state.progress_callback
    elif hasattr(state, "parent_state") and state.parent_state:
        progress_callback = getattr(state.parent_state, "progress_callback", None)

    # 创建带SSE推送的LLM实例
    if progress_callback:
        from utils.llm_factory import create_llm_with_callback

        return create_llm_with_callback(
            purpose=purpose, progress_callback=progress_callback
        )
    else:
        # 回退到普通创建
        if purpose == "bioinformatics":
            return create_bioinformatics_llm()
        elif purpose == "reasoning":
            return create_reasoning_llm()
        else:
            return create_bioinformatics_llm()


def _load_tools_json() -> str:
    """
    Load tool information (JSON format)

    Returns:
        JSON string of tool information
    """
    mcp_tools_path = agent_dir / "config" / "mcp_tools.json"

    try:
        if mcp_tools_path.exists():
            with open(mcp_tools_path, "r", encoding="utf-8") as f:
                tools_data = json.load(f)
                return json.dumps(tools_data, ensure_ascii=False, indent=2)
        else:
            print(f"⚠️ mcp_tools.json does not exist: {mcp_tools_path}")
            return "[]"
    except Exception as e:
        print(f"⚠️ Failed to load tool information: {e}")
        return "[]"


def _get_opensandbox_id(state: "ImmunityState") -> Optional[str]:
    """
    从 state 或 parent_state 获取 opensandbox_id

    遵循架构原则：所有子图通过 parent_state.merged_result 获取 opensandbox_id
    以复用 supervisor 创建的沙盒实例
    """
    # 1. 从 parent_state.merged_result 获取（优先）
    if state.parent_state:
        merged_result = getattr(state.parent_state, "merged_result", None) or {}
        if isinstance(merged_result, dict):
            opensandbox_id = merged_result.get("opensandbox_id")
            if opensandbox_id:
                print(f"[Immunity] 获取到 opensandbox_id: {opensandbox_id}")
                return opensandbox_id

    # 2. 从 parent_state 直接获取
    if state.parent_state:
        opensandbox_id = getattr(state.parent_state, "opensandbox_id", None)
        if opensandbox_id:
            print(f"[Immunity] 从 parent_state 获取到 opensandbox_id: {opensandbox_id}")
            return opensandbox_id

    # 3. 从 state 本身获取（如果有）
    opensandbox_id = getattr(state, "opensandbox_id", None)
    if opensandbox_id:
        print(f"[Immunity] 从 state 获取到 opensandbox_id: {opensandbox_id}")
        return opensandbox_id

    print("[Immunity] ⚠️ 未获取到 opensandbox_id，将创建新沙盒")
    return None


def _save_report(
    content: str,
    report_type: str,
    sandbox_dir: Optional[str] = None,
    local_sandbox_dir: Optional[str] = None,
    opensandbox_id: Optional[str] = None,
    progress_callback: Optional[callable] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Save report to file in sandbox output directory

    报告保存策略（遵循架构原则：通过 CodeAct 统一执行）：
    1. 远程沙盒环境：通过 CodeAct 执行代码保存到 {sandbox_dir}/output/reports/
    2. 本地环境回退：保存到 {local_sandbox_dir}/output/reports/
    3. 🔥 同时通过SSE推送文件内容到前端

    Args:
        content: Report content
        report_type: Report type (retrieval, deep_research, hypothesis, planning, evaluation)
        sandbox_dir: Sandbox directory (sandbox_data_dir like /data/sessions/{session_id})
        local_sandbox_dir: Local sandbox directory (fallback for local testing)
        opensandbox_id: OpenSandbox instance ID to reuse (IMPORTANT for session continuity)
        progress_callback: SSE进度回调函数，用于推送文件内容到前端
        session_id: 会话ID，用于日志记录

    Returns:
        Saved file path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{report_type}_{timestamp}.md"

    print(f"[Immunity] _save_report 调用:")
    print(f"  - report_type: {report_type}")
    print(f"  - sandbox_dir: {sandbox_dir}")
    print(f"  - local_sandbox_dir: {local_sandbox_dir}")
    print(f"  - opensandbox_id: {opensandbox_id}")

    # 尝试保存的路径列表（按优先级排序）
    save_paths = []

    # 1. 沙盒环境路径：/data/sessions/{session_id}/output/reports/
    #    通过 CodeAct 执行代码保存到远程沙盒（架构原则：唯一与 OpenSandbox 沟通的入口）
    if sandbox_dir:
        sandbox_dir_normalized = str(sandbox_dir).replace("\\", "/")
        is_unix_path = sandbox_dir_normalized.startswith("/")
        remote_path = f"{sandbox_dir_normalized.rstrip('/')}/output/reports/{filename}"

        # 远程沙盒路径，必须通过 CodeAct 保存
        if is_unix_path:
            try:
                # 使用 CodeAct 统一接口保存文件（架构原则：唯一与 OpenSandbox 沟通的入口）
                from utils.codeact_executor import (
                    execute_code_via_codeact,
                    is_codeact_available,
                )

                if is_codeact_available():
                    print(f"[Immunity] CodeAct 可用，准备保存报告到远程沙盒...")

                    # 转义内容中的特殊字符（包括换行符）
                    escaped_content = (
                        content.replace("\\", "\\\\")
                        .replace('"""', '\\"\\"\\"')
                        .replace("'''", "\\'\\'\\'")
                    )

                    # 容器内路径
                    container_path = remote_path.replace(
                        "/data/sessions/", "/tmp/sessions/", 1
                    )

                    # 使用更安全的代码模板（避免三引号问题）
                    import base64

                    content_b64 = base64.b64encode(content.encode("utf-8")).decode(
                        "ascii"
                    )

                    save_code = f'''
import os
import base64

# Base64 编码的内容
content_b64 = "{content_b64}"
file_path = "{container_path}"

try:
    # 解码内容
    content = base64.b64decode(content_b64).decode('utf-8')
    
    # 创建目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"__REPORT_SAVED__:{{file_path}}")
    print(f"__CONTENT_LENGTH__:{{len(content)}}")
except Exception as e:
    print(f"__REPORT_ERROR__:{{str(e)}}")
'''

                    print(f"[Immunity] 调用 CodeAct 保存报告...")
                    print(f"  - container_path: {container_path}")
                    print(f"  - opensandbox_id: {opensandbox_id}")

                    result = execute_code_via_codeact(
                        task_description=f"保存 {report_type} 报告到远程沙盒",
                        code_template=save_code,
                        sandbox_id=opensandbox_id,  # 传递 sandbox_id 复用沙盒
                        timeout_seconds=60,
                        keep_alive=True,
                    )

                    print(f"[Immunity] CodeAct 执行结果:")
                    print(f"  - status: {result.status}")
                    print(
                        f"  - output: {result.output[:200] if result.output else 'N/A'}..."
                    )
                    print(f"  - error: {result.error}")
                    print(f"  - sandbox_id: {result.sandbox_id}")

                    # 修改判断逻辑：只要输出中包含 __REPORT_SAVED__ 就认为成功
                    # 不再要求 result.is_success()，因为沙盒连接过程可能有一些非致命错误
                    if result.output and "__REPORT_SAVED__:" in result.output:
                        print(
                            f"📄 {report_type} report saved to remote sandbox via CodeAct: {remote_path}"
                        )
                        if progress_callback:
                            try:
                                report_type_display = {
                                    "retrieval": "检索报告",
                                    "deep_research": "深度研究报告",
                                    "hypothesis": "假设生成报告",
                                    "planning": "实验计划",
                                    "evaluation": "评估报告",
                                }.get(report_type, report_type)
                                progress_callback(
                                    event_type="file_content",
                                    message=f"📄 {report_type_display}已保存到沙盒",
                                    details={
                                        "file_type": report_type,
                                        "file_name": filename,
                                        "file_path": remote_path,
                                        "content": content,
                                        "content_length": len(content),
                                        "node": "immunity",
                                        "session_id": session_id,
                                    },
                                )
                                print(
                                    f"  ✅ 已推送 {report_type} 文件内容到前端 ({len(content)} 字符)"
                                )
                            except Exception as e:
                                print(f"  ⚠️ 推送文件内容到前端失败: {e}")
                        return remote_path
                    else:
                        print(
                            f"⚠️ Failed to save to remote sandbox via CodeAct: {result.error}"
                        )
                        print(f"ℹ️ Falling back to local save")
                else:
                    print(f"ℹ️ CodeAct not available, falling back to local save")

            except Exception as e:
                print(f"⚠️ Failed to save to remote sandbox via CodeAct: {e}")
                import traceback

                traceback.print_exc()
                print(f"ℹ️ Falling back to local save")

        # 如果是本地 Unix 路径，直接保存
        if is_unix_path and os.name != "nt":
            save_paths.append(
                (
                    "sandbox",
                    Path(sandbox_dir_normalized) / "output" / "reports" / filename,
                )
            )

    # 2. 本地回退路径：{local_sandbox_dir}/output/reports/
    if local_sandbox_dir:
        local_dir_normalized = str(local_sandbox_dir).replace("\\", "/")
        save_paths.append(
            ("local", Path(local_dir_normalized) / "output" / "reports" / filename)
        )

    # 3. 如果没有提供任何路径，使用当前目录下的 output/reports
    if not save_paths:
        save_paths.append(("fallback", Path("output") / "reports" / filename))

    # 尝试每个路径直到成功
    for path_type, report_file in save_paths:
        try:
            report_file.parent.mkdir(parents=True, exist_ok=True)

            with open(report_file, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"📄 {report_type} report saved to: {report_file} ({path_type})")
            if progress_callback:
                try:
                    report_type_display = {
                        "retrieval": "检索报告",
                        "deep_research": "深度研究报告",
                        "hypothesis": "假设生成报告",
                        "planning": "实验计划",
                        "evaluation": "评估报告",
                    }.get(report_type, report_type)
                    progress_callback(
                        event_type="file_content",
                        message=f"📄 {report_type_display}已保存到本地",
                        details={
                            "file_type": report_type,
                            "file_name": filename,
                            "file_path": str(report_file),
                            "content": content,
                            "content_length": len(content),
                            "node": "immunity",
                            "session_id": session_id,
                        },
                    )
                    print(
                        f"  ✅ 已推送 {report_type} 文件内容到前端 ({len(content)} 字符)"
                    )
                except Exception as e:
                    print(f"  ⚠️ 推送文件内容到前端失败: {e}")
            return str(report_file)
        except Exception as e:
            print(
                f"⚠️ Failed to save {report_type} report to {path_type} path {report_file}: {e}"
            )
            continue

    print(f"❌ All save attempts failed for {report_type} report")
    return ""


def _clean_json_response(response_text: str) -> Dict[str, Any]:
    """
    Clean and parse JSON response

    Args:
        response_text: Text returned by LLM

    Returns:
        Parsed JSON dictionary
    """
    # Try direct parsing
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting JSON code blocks
    json_block_patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]

    for pattern in json_block_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Try extracting the first JSON object
    json_match = re.search(r"\{[^}]+\}", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # If all fail, return empty dictionary
    return {}


# ===================== Stage 0: Cache Check Node (Mem0) =====================


def cache_check_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 0: Cache Check Node (Mem0)

    检查是否有相似的问题已经在 Mem0 中缓存

    如果缓存命中：
    - 直接从缓存加载所有结果
    - 设置 skip_immunity_stages=True 跳过后续阶段

    如果缓存未命中：
    - 继续执行后续阶段
    """
    _progress_logger.start_stage("cache_check", "检查 Mem0 缓存")

    if not MEM0_AVAILABLE:
        _progress_logger.log_info("Mem0 不可用，跳过缓存检查")
        _progress_logger.end_stage(
            "cache_check", success=True, details="跳过（Mem0 不可用）"
        )
        return state

    if not state.original_question:
        _progress_logger.log_warning("没有原始问题，跳过缓存检查")
        _progress_logger.end_stage(
            "cache_check", success=True, details="跳过（无输入）"
        )
        return state

    try:
        _progress_logger.log_info(f"检查缓存: {state.original_question[:100]}...")

        # 生成输入哈希
        input_hash = generate_input_hash(state.original_question)
        state.cache_input_hash = input_hash
        _progress_logger.log_info(f"输入哈希: {input_hash}")

        # 检查缓存
        is_cached, cached_trace = check_immunity_cache_sync(
            user_input=state.original_question,
            score_threshold=0.90,  # 要求 90% 以上相似度
        )

        if is_cached and cached_trace:
            _progress_logger.log_info("✅ 缓存命中！从 Mem0 加载结果...")

            # 从缓存加载所有结果
            state.cache_hit = True
            state.skip_immunity_stages = True

            # 加载优化查询
            state.optimized_questions = cached_trace.optimized_questions or [
                state.original_question
            ]
            state.optimized_question = "; ".join(state.optimized_questions)

            # 加载研究结果
            state.research_summary = cached_trace.research_summary or ""
            state.research_confidence = 80.0  # 缓存结果默认高置信度

            # 加载假设结果
            state.hypothesis_summary = cached_trace.hypothesis_summary or ""
            state.hypothesis_confidence = 80.0

            # 加载计划
            state.final_enhanced_plan = cached_trace.final_enhanced_plan or ""
            state.research_informed_plan = cached_trace.final_enhanced_plan or ""
            state.generated_plan = cached_trace.final_enhanced_plan or ""
            state.execution_plan = cached_trace.execution_plan or ""

            # 加载评估
            state.final_evaluation = cached_trace.final_evaluation or ""

            # 加载 Todo-List 摘要
            if cached_trace.todo_list_summary:
                state.decomposed_tasks = cached_trace.todo_list_summary.get("tasks", [])

            _progress_logger.log_info(
                f"  - 优化查询数: {len(state.optimized_questions)}"
            )
            _progress_logger.log_info(
                f"  - 研究摘要长度: {len(state.research_summary)}"
            )
            _progress_logger.log_info(f"  - 计划长度: {len(state.final_enhanced_plan)}")

            _progress_logger.end_stage(
                "cache_check", success=True, details="✅ 缓存命中"
            )
        else:
            _progress_logger.log_info("❌ 缓存未命中，将继续执行 immunity 阶段")
            state.cache_hit = False
            state.skip_immunity_stages = False
            _progress_logger.end_stage(
                "cache_check", success=True, details="缓存未命中"
            )

    except Exception as e:
        _progress_logger.log_error("缓存检查失败", e)
        import traceback

        traceback.print_exc()
        state.cache_hit = False
        state.skip_immunity_stages = False
        _progress_logger.end_stage("cache_check", success=False, details=str(e)[:50])

    return state


# ===================== Stage 1: Query Decomposition Node =====================


def query_decomposition_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 1: Query Decomposition Node

    Decompose user question into optimized sub-questions.
    This node has search tools bound to help understand research context.
    """
    # 启动工作流（在第一个节点执行时）
    if not _progress_logger.workflow_start_time:
        _progress_logger.start_workflow()

    _progress_logger.start_stage("query_decomposition", "将用户问题分解为优化的子查询")

    if not state.original_question:
        _progress_logger.log_warning("没有原始问题，跳过查询分解")
        _progress_logger.end_stage(
            "query_decomposition", success=True, details="跳过（无输入）"
        )
        return state

    _progress_logger.log_info(f"原始问题: {state.original_question[:150]}...")

    # 🔥 获取progress_callback（优先从parent_state，然后从state）
    progress_callback = None
    if hasattr(state, "progress_callback") and state.progress_callback:
        progress_callback = state.progress_callback
    elif hasattr(state, "parent_state") and state.parent_state:
        progress_callback = getattr(state.parent_state, "progress_callback", None)

    # 创建带SSE推送的LLM实例
    if progress_callback:
        from utils.llm_factory import create_llm_with_callback

        llm = create_llm_with_callback(
            purpose="bioinformatics", progress_callback=progress_callback
        )
    else:
        llm = _get_llm_with_callback(state, "bioinformatics")
    if not llm:
        _progress_logger.log_warning("LLM 不可用，使用原始问题")
        state.optimized_questions = [state.original_question]
        state.optimized_question = state.original_question
        _progress_logger.end_stage(
            "query_decomposition", success=True, details="降级处理（无 LLM）"
        )
        return state

    try:
        from langchain_core.messages import (
            SystemMessage,
            HumanMessage,
            AIMessage,
            ToolMessage,
        )
        from langchain_core.output_parsers import JsonOutputParser

        tools_info = _load_tools_json()

        # Use reference project's QUERY_EXPANSION_PROMPT
        query_expansion_prompt = ImmunityPrompts.QUERY_EXPANSION_PROMPT.format(
            tools_info=tools_info, query=state.original_question
        )

        # Define output schema
        class QueryExpansion(BaseModel):
            queries: List[str]

        output_parser = JsonOutputParser(pydantic_object=QueryExpansion)

        messages = [
            SystemMessage(
                content="You are a professional query optimization expert capable of decomposing complex research queries into multiple optimized sub-queries."
            ),
            HumanMessage(content=query_expansion_prompt),
        ]

        # Bind search tools to LLM for understanding research context
        llm_with_tools = llm.bind_tools([web_search, knowledge_search])

        # 记录 LLM 调用
        llm_info = {
            "model": getattr(llm, "model", getattr(llm, "model_name", "unknown")),
            "temperature": getattr(llm, "temperature", "N/A"),
            "timeout": getattr(llm, "timeout", "N/A"),
            "tools_bound": ["web_search", "knowledge_search"],
        }
        _progress_logger.log_llm_start(llm_info, len(query_expansion_prompt))

        start_time = time.perf_counter()

        # Invoke LLM with tools, handle tool calls if any
        max_tool_iterations = 2  # Limit tool call iterations for query decomposition
        tool_iterations = 0

        while tool_iterations < max_tool_iterations:
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                _progress_logger.log_llm_end(elapsed, 0, success=False)
                raise

            # Check if LLM made tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_iterations += 1
                _progress_logger.log_info(
                    f"  🔧 LLM 请求工具调用 (迭代 {tool_iterations}/{max_tool_iterations})"
                )

                # Add AI message with tool calls to conversation
                messages.append(response)

                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "")

                    _progress_logger.log_info(
                        f"    - 执行工具: {tool_name}({tool_args})"
                    )

                    # Execute tool
                    tool_result = _execute_tool_call(tool_name, tool_args)

                    # Add tool result to messages
                    messages.append(
                        ToolMessage(content=tool_result, tool_call_id=tool_id)
                    )

                # Continue loop to get final response
                continue

            # No tool calls, we have the final response
            break

        elapsed = time.perf_counter() - start_time
        _progress_logger.log_llm_end(elapsed, len(str(response)) if response else 0)

        if tool_iterations > 0:
            _progress_logger.log_info(f"  📊 工具调用总次数: {tool_iterations}")

        # Parse the response to extract queries
        response_content = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Try to parse as JSON for structured output
        try:
            parsed = output_parser.parse(response_content)
            if isinstance(parsed, dict) and "queries" in parsed:
                state.optimized_questions = parsed["queries"]
            elif hasattr(parsed, "queries"):
                state.optimized_questions = parsed.queries
            else:
                # Fallback: use structured output on original LLM
                structured_llm = llm.with_structured_output(QueryExpansion)
                structured_response = structured_llm.invoke(messages)
                if hasattr(structured_response, "queries"):
                    state.optimized_questions = structured_response.queries
                elif isinstance(structured_response, dict):
                    state.optimized_questions = structured_response.get(
                        "queries", [state.original_question]
                    )
                else:
                    state.optimized_questions = [state.original_question]
        except Exception as parse_error:
            _progress_logger.log_warning(
                f"JSON 解析失败，尝试 structured_output: {parse_error}"
            )
            # Fallback: use structured output
            structured_llm = llm.with_structured_output(QueryExpansion)
            structured_response = structured_llm.invoke(messages)
            if hasattr(structured_response, "queries"):
                state.optimized_questions = structured_response.queries
            elif isinstance(structured_response, dict):
                state.optimized_questions = structured_response.get(
                    "queries", [state.original_question]
                )
            else:
                state.optimized_questions = [state.original_question]

        if not state.optimized_questions:
            state.optimized_questions = [state.original_question]

        state.optimized_question = "; ".join(state.optimized_questions)

        # 输出分解结果
        _progress_logger.log_info(f"生成的子查询数量: {len(state.optimized_questions)}")
        for i, q in enumerate(state.optimized_questions, 1):
            _progress_logger.log_info(f"  子查询 {i}: {q[:80]}...")

        _progress_logger.end_stage(
            "query_decomposition",
            success=True,
            details=f"生成 {len(state.optimized_questions)} 个优化查询",
        )

    except Exception as e:
        _progress_logger.log_error("查询分解失败", e)
        state.optimized_questions = [state.original_question]
        state.optimized_question = state.original_question
        _progress_logger.end_stage(
            "query_decomposition", success=False, details="降级使用原始问题"
        )

    return state


# ===================== Stage 2: Retrieval Node =====================


def retrieval_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 2: Retrieval Node

    Parallel execution of three retrieval methods:
    1. retrieve: Retrieve from Qdrant vector database
    2. web_search_node: Tavily API web search
    3. web_retrieval_search: Multiple Web source retrieval

    Retrieval results are used for:
    - Stage 3: Deep research analysis
    - Stage 5: Plan generation (citation references)
    """
    _progress_logger.start_stage("retrieval", "并行执行三种检索方法")

    if not state.optimized_questions:
        _progress_logger.log_warning("没有优化查询，跳过检索")
        _progress_logger.end_stage("retrieval", success=True, details="跳过（无查询）")
        return state

    try:
        from .retrieval_tools import parallel_retrieval_sync

        _progress_logger.log_info("执行三种检索方法：")
        _progress_logger.log_info("  1. Qdrant 向量数据库检索")
        _progress_logger.log_info("  2. Tavily API 网页搜索")
        _progress_logger.log_info("  3. Web 多源检索")
        _progress_logger.log_info(f"检索查询数量: {len(state.optimized_questions)}")

        start_time = time.perf_counter()

        retrieval_results = parallel_retrieval_sync(
            queries=state.optimized_questions,
            original_question=state.original_question,
            k_per_query=10,
        )

        elapsed = time.perf_counter() - start_time

        # Extract retrieval results
        state.context = retrieval_results.get("context", "")
        state.retrieval_docs = retrieval_results.get("retrieval_docs", [])
        state.citations = retrieval_results.get("citations", [])

        # Generate retrieval summary
        retrieval_summary = f"""
Retrieval Completed (Parallel Retrieval):
- Optimized queries count: {len(state.optimized_questions)}
- Retrieved documents count: {len(state.retrieval_docs)}
- Citations count: {len(state.citations)}
- Context length: {len(state.context)} characters

Retrieval Methods:
1. Qdrant vector database retrieval
2. Tavily API web search
3. Web retrieval (multiple sources)

Retrieved Documents (Top 10):
{chr(10).join([f"{i + 1}. **{doc.get('title', 'N/A')}** (Relevance: {doc.get('relevance_score', 0):.2f})" + chr(10) + f"   - Source: {doc.get('source', 'N/A')}" + chr(10) + f"   - Summary: {doc.get('summary', '')[:200]}..." for i, doc in enumerate(state.retrieval_docs[:10])])}

Main Citations (Top 10):
{chr(10).join([f"{i + 1}. {cite.get('author', 'N/A')} et al. ({cite.get('year', 'N/A')}). {cite.get('title', 'N/A')}. *{cite.get('journal', 'N/A')}*" + (f" DOI: {cite.get('doi', '')}" if cite.get("doi") else "") for i, cite in enumerate(state.citations[:10])])}
"""

        _progress_logger.log_info(f"检索完成，耗时: {elapsed:.2f} 秒")
        _progress_logger.log_info(f"  - 检索文档数: {len(state.retrieval_docs)}")
        _progress_logger.log_info(f"  - 引用数: {len(state.citations)}")
        _progress_logger.log_info(f"  - 上下文长度: {len(state.context)} 字符")

        # Save retrieval report
        report_path = _save_report(
            retrieval_summary,
            "retrieval",
            state.sandbox_dir,
            state.local_sandbox_dir,
            opensandbox_id=_get_opensandbox_id(state),
            progress_callback=state.progress_callback,
            session_id=state.session_id,
        )
        state.retrieval_report_path = report_path

        _progress_logger.end_stage(
            "retrieval",
            success=True,
            details=f"文档: {len(state.retrieval_docs)}, 引用: {len(state.citations)}, 耗时: {elapsed:.1f}s",
        )

    except Exception as e:
        _progress_logger.log_error("检索失败", e)
        import traceback

        traceback.print_exc()
        # Use empty context on failure
        state.context = ""
        state.retrieval_docs = []
        state.citations = []
        _progress_logger.end_stage(
            "retrieval", success=False, details="检索出错，使用空上下文"
        )

    return state


# ===================== Stage 3: Deep Research Node (using deep_research subgraph) =====================


def deep_research_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 3: Deep Research Node

    Uses the deep_research subgraph to conduct in-depth analysis of research questions.
    The deep_research subgraph provides multi-step research with web search and synthesis.
    """
    _progress_logger.start_stage("deep_research", "使用 deep_research 子图进行多步分析")

    if not state.original_question:
        _progress_logger.log_warning("没有原始问题，跳过深度研究")
        _progress_logger.end_stage(
            "deep_research", success=True, details="跳过（无输入）"
        )
        return state

    try:
        # Prepare the research question combining original question and retrieval context
        research_question = state.original_question

        # Add retrieval context if available
        if state.context:
            research_question = f"""
研究主题: {state.original_question}

已检索的背景资料:
{state.context[:4000]}

请基于以上背景资料，深入研究并回答上述问题。
"""

        # Add optimized queries if available
        if state.optimized_questions:
            sub_queries = "\n".join([f"- {q}" for q in state.optimized_questions[:5]])
            research_question += f"\n\n重点关注以下子问题:\n{sub_queries}"

        _progress_logger.log_info(f"研究问题: {state.original_question[:100]}...")
        _progress_logger.log_info("使用 deep_research 子图进行多步分析...")
        _progress_logger.log_info(
            "配置: max_iterations=3, max_concurrent=2, max_tool_calls=6"
        )

        # Get deep_research subgraph configuration
        dr_config = get_deep_research_config(
            thread_id=f"immunity_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            max_researcher_iterations=3,  # Limit iterations for efficiency
            max_concurrent_research_units=2,
            max_react_tool_calls=6,
        )

        # Prepare input for deep_research subgraph
        research_input = {"messages": [{"role": "user", "content": research_question}]}

        # Run deep_research subgraph asynchronously
        async def run_deep_research_async():
            from langgraph.checkpoint.memory import MemorySaver
            from nodes.subagents.deep_research.deep_researcher import (
                deep_researcher_builder,
            )

            _progress_logger.log_info("正在编译 deep_research 子图...")

            # Compile with memory checkpointing
            graph = deep_researcher_builder.compile(checkpointer=MemorySaver())

            _progress_logger.log_info(
                "开始执行 deep_research 子图（可能需要较长时间）..."
            )

            return await graph.ainvoke(research_input, dr_config)

        # Execute async function
        start_time = time.perf_counter()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new event loop
                import concurrent.futures

                _progress_logger.log_info("检测到运行中的事件循环，使用线程池执行...")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, run_deep_research_async())
                    result = future.result(timeout=300)  # 5 minute timeout
            else:
                _progress_logger.log_info("使用现有事件循环执行...")
                result = loop.run_until_complete(run_deep_research_async())
        except RuntimeError:
            # No event loop, create one
            _progress_logger.log_info("创建新的事件循环...")
            result = asyncio.run(run_deep_research_async())

        elapsed = time.perf_counter() - start_time
        _progress_logger.log_info(f"deep_research 子图执行完成，耗时: {elapsed:.2f} 秒")

        # Extract results from deep_research subgraph
        final_report = result.get("final_report", "")
        research_brief = result.get("research_brief", "")
        notes = result.get("notes", [])

        # Map results back to ImmunityState
        if final_report:
            state.research_summary = f"""
<research_findings>
    <research_finding>
        {final_report}
    </research_finding>
</research_findings>
"""
            state.deep_research_findings = {
                "final_report": final_report,
                "research_brief": research_brief,
                "notes": notes,
                "topic": state.original_question,
            }

            # Extract insights from notes
            state.research_insights = []
            state.research_evidence = []
            for note in notes[:10]:
                if isinstance(note, str):
                    state.research_insights.append(note[:500])  # Truncate long notes

            # Set confidence based on results
            state.research_confidence = 80.0 if final_report else 50.0

            _progress_logger.log_info(f"深度研究完成:")
            _progress_logger.log_info(f"  - 最终报告长度: {len(final_report)} 字符")
            _progress_logger.log_info(f"  - 研究简报长度: {len(research_brief)} 字符")
            _progress_logger.log_info(f"  - 笔记数量: {len(notes)}")
            _progress_logger.log_info(f"  - 置信度: {state.research_confidence:.1f}%")

            # Save research report
            report_path = _save_report(
                state.research_summary,
                "deep_research",
                state.sandbox_dir,
                state.local_sandbox_dir,
                opensandbox_id=_get_opensandbox_id(state),
                progress_callback=state.progress_callback,
                session_id=state.session_id,
            )

            _progress_logger.end_stage(
                "deep_research",
                success=True,
                details=f"报告: {len(final_report)}字符, 笔记: {len(notes)}, 耗时: {elapsed:.1f}s",
            )
        else:
            _progress_logger.log_warning("深度研究返回空结果，使用降级方案")
            # Fallback to simple context-based research
            state.research_summary = f"""
<research_findings>
    <research_finding>
        Research Topic: {state.original_question}
        
        Based on retrieval context:
        {state.context[:2000] if state.context else "No context available"}
        
        Optimized queries:
        {chr(10).join([f"- {q}" for q in state.optimized_questions[:5]])}
    </research_finding>
</research_findings>
"""
            state.research_confidence = 50.0
            _progress_logger.end_stage(
                "deep_research", success=True, details="降级使用检索上下文"
            )

    except Exception as e:
        _progress_logger.log_error("深度研究失败", e)
        import traceback

        traceback.print_exc()

        # Fallback: create research summary from available context
        state.research_summary = f"""
<research_findings>
    <research_finding>
        Research Topic: {state.original_question}
        
        Note: Deep research subgraph encountered an error. Using available context.
        
        Context from retrieval:
        {state.context[:2000] if state.context else "No context available"}
        
        Optimized queries:
        {chr(10).join([f"- {q}" for q in state.optimized_questions[:5]])}
    </research_finding>
</research_findings>
"""
        state.research_confidence = 40.0
        _progress_logger.end_stage(
            "deep_research", success=False, details="使用上下文降级方案"
        )

    return state


# ===================== Stage 4: Hypothesis Generation Node =====================


def hypothesis_generation_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 4: Hypothesis Generation Node

    Generate testable hypotheses based on research results
    """
    _progress_logger.start_stage(
        "hypothesis_generation", "基于研究结果生成可测试的假设"
    )

    if not state.research_summary:
        _progress_logger.log_warning("没有研究结果，跳过假设生成")
        _progress_logger.end_stage(
            "hypothesis_generation", success=True, details="跳过（无研究结果）"
        )
        return state

    llm = _get_llm_with_callback(state, "bioinformatics")
    if not llm:
        _progress_logger.log_warning("LLM 不可用，跳过假设生成")
        _progress_logger.end_stage(
            "hypothesis_generation", success=True, details="跳过（无 LLM）"
        )
        return state

    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        # Use reference project's HYPOTHESIS_GENERATION_PROMPT
        context = state.context if state.context else ""
        question = f"""
<questions>
    <question>
        {state.original_question}
    </question>
</questions>
"""

        # Use reference project's HYPOTHESIS_GENERATION_PROMPT
        hypothesis_prompt = ImmunityPrompts.HYPOTHESIS_GENERATION_PROMPT.format(
            research_findings=state.research_summary, context=context, question=question
        )

        # Use JSON parser (not using with_structured_output, as dict type is not supported)
        from langchain_core.output_parsers import JsonOutputParser

        output_parser = JsonOutputParser()

        # ZhipuAI requires at least one user message, so send prompt as user message
        messages = [HumanMessage(content=hypothesis_prompt)]

        llm_info = {
            "type": type(llm).__name__,
            "model": getattr(llm, "model", getattr(llm, "model_name", None)),
            "temperature": getattr(llm, "temperature", None),
            "timeout": getattr(llm, "timeout", None),
            "max_retries": getattr(llm, "max_retries", None),
        }
        prompt_stats = {
            "original_question_len": len(state.original_question or ""),
            "research_summary_len": len(state.research_summary or ""),
            "context_len": len(context),
            "question_block_len": len(question),
        }
        _progress_logger.log_info(f"Prompt 统计: {prompt_stats}")

        # 记录 LLM 调用
        _progress_logger.log_llm_start(llm_info, len(hypothesis_prompt))

        # Directly invoke LLM, then parse JSON
        start_time = time.perf_counter()
        try:
            response = llm.invoke(messages)
            elapsed = time.perf_counter() - start_time
            _progress_logger.log_llm_end(
                elapsed, len(response.content) if hasattr(response, "content") else 0
            )
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            _progress_logger.log_llm_end(elapsed, 0, success=False)
            raise
        response_content = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Use JsonOutputParser to parse response
        try:
            hypothesis_data = output_parser.parse(response_content)
            if not isinstance(hypothesis_data, dict):
                hypothesis_data = {}
        except Exception as e:
            _progress_logger.log_warning(f"JSON 解析失败，尝试直接解析: {e}")
            # Try extracting JSON from response
            import json
            import re

            # Try extracting JSON code block or direct parsing
            json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
            if json_match:
                try:
                    hypothesis_data = json.loads(json_match.group())
                except:
                    hypothesis_data = {}
            else:
                hypothesis_data = {}

        if hypothesis_data and hypothesis_data.get("statement"):
            state.hypothesis = hypothesis_data
            state.hypothesis_confidence = float(
                hypothesis_data.get("confidence_score", 70.0)
            )

            # Extract testable predictions
            predictions = hypothesis_data.get("testable_predictions", [])
            state.testable_predictions = []
            for pred in predictions:
                if isinstance(pred, dict):
                    state.testable_predictions.append(pred.get("prediction", ""))
                else:
                    state.testable_predictions.append(str(pred))

            # Generate hypothesis summary
            hypothesis_summary = f"""
Hypothesis Statement: {hypothesis_data.get("statement", "Not specified")}

Confidence Score: {hypothesis_data.get("confidence_score", 0)}%
Innovation Level: {hypothesis_data.get("innovation_level", "moderate")}

Testable Predictions:
{chr(10).join([f"- {pred.get('prediction', '')} (Timeline: {pred.get('timeline', 'TBD')})" for pred in predictions[:5]])}

Validation Methods:
{chr(10).join([f"- {pred.get('validation_method', '')}" for pred in predictions[:5]])}

Expected Outcomes:
{chr(10).join([f"- {pred.get('expected_outcome', '')}" for pred in predictions[:5]])}

Falsification Criteria:
{chr(10).join([f"- {criteria}" for criteria in hypothesis_data.get("falsification_criteria", [])[:5]])}

Evidence Basis:
{chr(10).join([f"- {evidence}" for evidence in hypothesis_data.get("evidence_basis", [])[:5]])}

Expected Information Gain:
{hypothesis_data.get("expected_information_gain", "To be determined")}

Scientific Rationale:
{hypothesis_data.get("rationale", "Not provided")}
"""

            state.hypothesis_summary = f"""
<hypothesis_findings>
    <hypothesis_finding>
        {hypothesis_summary}
    </hypothesis_finding>
</hypothesis_findings>
"""

            _progress_logger.log_info(f"假设生成完成:")
            _progress_logger.log_info(
                f"  - 假设: {hypothesis_data.get('statement', 'Not specified')[:100]}..."
            )
            _progress_logger.log_info(f"  - 置信度: {state.hypothesis_confidence:.1f}%")
            _progress_logger.log_info(
                f"  - 创新水平: {hypothesis_data.get('innovation_level', 'moderate')}"
            )

            # Save hypothesis report
            report_path = _save_report(
                state.hypothesis_summary,
                "hypothesis",
                state.sandbox_dir,
                state.local_sandbox_dir,
                opensandbox_id=_get_opensandbox_id(state),
                progress_callback=state.progress_callback,
                session_id=state.session_id,
            )

            _progress_logger.end_stage(
                "hypothesis_generation",
                success=True,
                details=f"置信度: {state.hypothesis_confidence:.0f}%, 预测: {len(predictions)}",
            )
        else:
            _progress_logger.log_warning("无法解析假设结果")
            _progress_logger.end_stage(
                "hypothesis_generation", success=False, details="JSON 解析失败"
            )

    except Exception as e:
        _progress_logger.log_error("假设生成失败", e)
        import traceback

        traceback.print_exc()
        _progress_logger.end_stage(
            "hypothesis_generation", success=False, details=str(e)[:50]
        )

    return state


# ===================== Stage 5: Plan Generation Node ⭐ =====================


def _execute_tool_call(tool_name: str, tool_args: dict) -> str:
    """
    Execute a tool call and return the result.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments for the tool

    Returns:
        Tool execution result as string
    """
    try:
        if tool_name == "web_search":
            return web_search.invoke(
                tool_args.get("query", ""), tool_args.get("max_results", 5)
            )
        elif tool_name == "knowledge_search":
            return knowledge_search.invoke(
                tool_args.get("query", ""),
                tool_args.get("k", 5),
                tool_args.get("collections", None),
            )
        elif tool_name == "read_webpage":
            return read_webpage.invoke(
                tool_args.get("url", ""), tool_args.get("max_chars", 10000)
            )
        else:
            return f"[Error] Unknown tool: {tool_name}"
    except Exception as e:
        return f"[Error] Tool {tool_name} execution failed: {str(e)}"


def planning_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 5: Planning Node ⭐

    Generate executable experimental plan based on research results and hypotheses.
    This node has search tools bound to help LLM find latest experimental methods.
    """
    _progress_logger.start_stage("planning", "基于研究结果和假设生成可执行实验计划")

    llm = _get_llm_with_callback(state, "bioinformatics")
    if not llm:
        _progress_logger.log_warning("LLM 不可用，使用简单计划生成")
        _progress_logger.end_stage("planning", success=True, details="降级使用简单计划")
        return _generate_simple_plan(state)

    try:
        from langchain_core.messages import (
            SystemMessage,
            HumanMessage,
            AIMessage,
            ToolMessage,
        )

        tools_info = _load_tools_json()

        # Format optimized queries
        format_queries = []
        for i, query in enumerate(state.optimized_questions, 1):
            format_queries.append(f"""
<sub_questions>
    <q{i}>
        {query}
    </q{i}>
</sub_questions>
""")
        optimized_questions_text = (
            "\n\n".join(format_queries) if format_queries else "None"
        )

        # Get citations (from retrieval node)
        if state.citations:
            citations_json = json.dumps(state.citations, ensure_ascii=False, indent=2)
        else:
            citations_json = "[]"
        context = state.context if state.context else ""

        # Use reference project's IMMUNITY_PLANNING_PROMPT
        planning_prompt = ImmunityPrompts.IMMUNITY_PLANNING_PROMPT.format(
            original_question=state.original_question,
            optimized_questions=optimized_questions_text,
            hypothesis_findings=state.hypothesis_summary,
            tools_info=tools_info,
            research_findings=state.research_summary,
            context=context,
            citations_json=citations_json,
        )

        # ZhipuAI requires at least one user message, so send prompt as user message
        messages = [HumanMessage(content=planning_prompt)]

        # Bind search tools to LLM for planning (helps find latest methods)
        llm_with_tools = llm.bind_tools([web_search, knowledge_search, read_webpage])

        llm_info = {
            "type": type(llm).__name__,
            "model": getattr(llm, "model", getattr(llm, "model_name", None)),
            "temperature": getattr(llm, "temperature", None),
            "timeout": getattr(llm, "timeout", None),
            "max_retries": getattr(llm, "max_retries", None),
            "tools_bound": ["web_search", "knowledge_search", "read_webpage"],
        }
        prompt_stats = {
            "original_question_len": len(state.original_question or ""),
            "optimized_questions_count": len(state.optimized_questions or []),
            "optimized_questions_len": len(optimized_questions_text),
            "hypothesis_summary_len": len(state.hypothesis_summary or ""),
            "research_summary_len": len(state.research_summary or ""),
            "context_len": len(context),
            "citations_count": len(state.citations or []),
            "citations_json_len": len(citations_json),
            "tools_info_len": len(tools_info),
        }
        _progress_logger.log_info(f"Prompt 统计: {prompt_stats}")

        # 记录 LLM 调用
        _progress_logger.log_llm_start(llm_info, len(planning_prompt))

        start_time = time.perf_counter()

        # Invoke LLM with tools, handle tool calls if any
        max_tool_iterations = 3  # Limit tool call iterations
        tool_iterations = 0

        while tool_iterations < max_tool_iterations:
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                _progress_logger.log_llm_end(elapsed, 0, success=False)
                raise

            # Check if LLM made tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_iterations += 1
                _progress_logger.log_info(
                    f"  🔧 LLM 请求工具调用 (迭代 {tool_iterations}/{max_tool_iterations})"
                )

                # Add AI message with tool calls to conversation
                messages.append(response)

                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "")

                    _progress_logger.log_info(
                        f"    - 执行工具: {tool_name}({tool_args})"
                    )

                    # Execute tool
                    tool_result = _execute_tool_call(tool_name, tool_args)

                    # Add tool result to messages
                    messages.append(
                        ToolMessage(content=tool_result, tool_call_id=tool_id)
                    )

                # Continue loop to get final response
                continue

            # No tool calls, we have the final response
            break

        elapsed = time.perf_counter() - start_time
        _progress_logger.log_llm_end(
            elapsed, len(response.content) if hasattr(response, "content") else 0
        )

        if tool_iterations > 0:
            _progress_logger.log_info(f"  📊 工具调用总次数: {tool_iterations}")

        plan_content = (
            response.content.strip() if hasattr(response, "content") else str(response)
        )

        state.final_enhanced_plan = plan_content
        state.research_informed_plan = plan_content
        state.generated_plan = plan_content

        _progress_logger.log_info(f"计划生成完成:")
        _progress_logger.log_info(f"  - 计划长度: {len(plan_content)} 字符")

        # Save plan report
        report_path = _save_report(
            plan_content,
            "planning",
            state.sandbox_dir,
            state.local_sandbox_dir,
            opensandbox_id=_get_opensandbox_id(state),
            progress_callback=state.progress_callback,
            session_id=state.session_id,
        )

        _progress_logger.end_stage(
            "planning",
            success=True,
            details=f"计划长度: {len(plan_content)} 字符, 耗时: {elapsed:.1f}s",
        )

    except Exception as e:
        _progress_logger.log_error("计划生成失败", e)
        import traceback

        traceback.print_exc()
        # Fallback solution
        _progress_logger.end_stage(
            "planning", success=False, details="降级使用简单计划"
        )
        return _generate_simple_plan(state)

    return state


def _generate_simple_plan(state: ImmunityState) -> ImmunityState:
    """Generate simple plan (fallback solution)"""
    plan_lines = ["# Experimental Plan\n"]
    plan_lines.append(f"## Overview\n")
    plan_lines.append(f"Based on research question: {state.original_question}\n\n")

    if state.hypothesis_summary:
        plan_lines.append(f"## Hypothesis\n")
        plan_lines.append(f"{state.hypothesis_summary}\n\n")

    if state.research_summary:
        plan_lines.append(f"## Research Background\n")
        plan_lines.append(f"{state.research_summary[:500]}...\n\n")

    plan_lines.append(f"## Experimental Steps\n")
    plan_lines.append(f"(Detailed plan needs to be generated by LLM)\n")

    state.final_enhanced_plan = "".join(plan_lines)
    return state


# ===================== Stage 6: Evaluation Node =====================


def evaluation_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 6: Evaluation Node

    Evaluate the scientific validity and feasibility of the experimental plan
    """
    _progress_logger.start_stage("evaluation", "评估实验计划的科学性和可行性")

    if not state.final_enhanced_plan:
        _progress_logger.log_warning("没有计划可评估")
        state.final_evaluation = "No plan generated, cannot evaluate"
        _progress_logger.end_stage("evaluation", success=True, details="跳过（无计划）")
        return state

    # If user-provided plan, skip evaluation
    if state.is_user_provided_plan:
        _progress_logger.log_info("用户提供的计划，跳过自动评估")
        state.final_evaluation = f"""User-provided execution plan:

{state.final_enhanced_plan}

---
Evaluation Note: This plan was directly provided by the user, automatic evaluation step has been skipped.
"""
        report_path = _save_report(
            state.final_evaluation,
            "evaluation",
            state.sandbox_dir,
            state.local_sandbox_dir,
            opensandbox_id=_get_opensandbox_id(state),
            progress_callback=state.progress_callback,
            session_id=state.session_id,
        )
        _progress_logger.end_stage(
            "evaluation", success=True, details="跳过（用户计划）"
        )
        return state

    llm = _get_llm_with_callback(state, "bioinformatics")
    if not llm:
        _progress_logger.log_warning("LLM 不可用，跳过评估")
        state.final_evaluation = "LLM unavailable, cannot perform evaluation"
        _progress_logger.end_stage("evaluation", success=True, details="跳过（无 LLM）")
        return state

    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        # Use reference project's EVALUATE_PLANNING_PROMPT
        evaluation_prompt = ImmunityPrompts.EVALUATE_PLANNING_PROMPT.format(
            plan=state.final_enhanced_plan
        )

        # ZhipuAI requires at least one user message, so send prompt as user message
        messages = [HumanMessage(content=evaluation_prompt)]

        llm_info = {
            "model": getattr(llm, "model", getattr(llm, "model_name", None)),
            "temperature": getattr(llm, "temperature", None),
            "timeout": getattr(llm, "timeout", None),
        }
        _progress_logger.log_llm_start(llm_info, len(evaluation_prompt))

        start_time = time.perf_counter()
        response = llm.invoke(messages)
        elapsed = time.perf_counter() - start_time

        evaluation_content = (
            response.content.strip() if hasattr(response, "content") else str(response)
        )

        _progress_logger.log_llm_end(elapsed, len(evaluation_content))

        state.final_evaluation = evaluation_content

        _progress_logger.log_info(f"评估完成:")
        _progress_logger.log_info(f"  - 评估报告长度: {len(evaluation_content)} 字符")

        # Save evaluation report
        full_evaluation = evaluation_content + "\n\n" + state.original_question
        report_path = _save_report(
            full_evaluation,
            "evaluation",
            state.sandbox_dir,
            state.local_sandbox_dir,
            opensandbox_id=_get_opensandbox_id(state),
            progress_callback=state.progress_callback,
            session_id=state.session_id,
        )

        _progress_logger.end_stage(
            "evaluation",
            success=True,
            details=f"报告长度: {len(evaluation_content)} 字符, 耗时: {elapsed:.1f}s",
        )

    except Exception as e:
        _progress_logger.log_error("评估失败", e)
        import traceback

        traceback.print_exc()
        state.final_evaluation = f"Evaluation process error: {str(e)}"
        _progress_logger.end_stage("evaluation", success=False, details=str(e)[:50])

    return state


# ===================== Input/Output Mapping =====================


def immunity_input_mapper(global_state: GlobalState) -> ImmunityState:
    """
    Map main graph state to Immunity subgraph state

    Args:
        global_state: Main graph global state

    Returns:
        Immunity subgraph state
    """
    # sandbox_data_dir 是沙盒服务器路径（如 /data/sessions/{session_id}）
    # sandbox_dir 是本地沙盒目录（如 D:/path/to/sandbox）
    sandbox_data_dir = global_state.sandbox_data_dir or ""
    local_sandbox_dir = global_state.sandbox_dir or ""

    immunity_state = ImmunityState(
        original_question=global_state.user_input,
        subtasks=global_state.subtasks,
        parallel_task_groups=global_state.parallel_task_groups,
        sandbox_dir=sandbox_data_dir,  # 主路径：沙盒服务器路径
        local_sandbox_dir=local_sandbox_dir,  # 回退路径：本地路径
        parent_state=global_state,
        # 🔥 传递progress_callback和session_id，确保SSE消息能推送到前端
        progress_callback=getattr(global_state, "progress_callback", None),
        session_id=getattr(global_state, "session_id", None),
        # Mem0 缓存相关字段初始化
        cache_hit=False,
        skip_immunity_stages=False,
        cache_input_hash="",
    )

    return immunity_state


def immunity_output_mapper(
    immunity_state: ImmunityState, global_state: GlobalState
) -> GlobalState:
    """
    Map Immunity subgraph state back to main graph state

    Args:
        immunity_state: Immunity subgraph state (can be ImmunityState object or dict)
        global_state: Main graph global state

    Returns:
        Updated main graph state
    """
    # Handle case where immunity_state might be a dict (from LangGraph invoke)
    if isinstance(immunity_state, dict):
        original_question = immunity_state.get("original_question", "")
        optimized_questions = immunity_state.get("optimized_questions", [])
        research_summary = immunity_state.get("research_summary", "")
        hypothesis_summary = immunity_state.get("hypothesis_summary", "")
        final_enhanced_plan = immunity_state.get("final_enhanced_plan", "")
        plan_steps = immunity_state.get("plan_steps", [])
        plan_summary = immunity_state.get("plan_summary", "")
        final_evaluation = immunity_state.get("final_evaluation", "")
        executable_plan = immunity_state.get("executable_plan", {})
        generated_plan = immunity_state.get("generated_plan", "")
        skip_planning = immunity_state.get("skip_planning", False)
        research_confidence = immunity_state.get("research_confidence", 0.0)
        hypothesis_confidence = immunity_state.get("hypothesis_confidence", 0.0)
    else:
        original_question = getattr(immunity_state, "original_question", "")
        optimized_questions = getattr(immunity_state, "optimized_questions", []) or []
        research_summary = getattr(immunity_state, "research_summary", "")
        hypothesis_summary = getattr(immunity_state, "hypothesis_summary", "")
        final_enhanced_plan = getattr(immunity_state, "final_enhanced_plan", "")
        plan_steps = getattr(immunity_state, "plan_steps", []) or []
        plan_summary = getattr(immunity_state, "plan_summary", "")
        final_evaluation = getattr(immunity_state, "final_evaluation", "")
        executable_plan = getattr(immunity_state, "executable_plan", {}) or {}
        generated_plan = getattr(immunity_state, "generated_plan", "")
        skip_planning = getattr(immunity_state, "skip_planning", False)
        research_confidence = getattr(immunity_state, "research_confidence", 0.0)
        hypothesis_confidence = getattr(immunity_state, "hypothesis_confidence", 0.0)

    # Store complete experimental plan to merged_result
    if not global_state.merged_result:
        global_state.merged_result = {}

    global_state.merged_result["immunity_plan"] = {
        "original_question": original_question,
        "optimized_questions": optimized_questions,
        "research_summary": research_summary,
        "hypothesis_summary": hypothesis_summary,
        "experimental_plan": final_enhanced_plan,
        "final_enhanced_plan": final_enhanced_plan,
        "plan_steps": plan_steps,
        "plan_summary": plan_summary,
        "evaluation": final_evaluation,
        "executable_plan": executable_plan,
    }

    # Persist execution plan to global state for downstream logging/usage
    execution_plan = None
    if final_enhanced_plan:
        execution_plan = final_enhanced_plan
    elif generated_plan:
        execution_plan = generated_plan
    elif plan_summary:
        execution_plan = plan_summary

    if not execution_plan:
        if skip_planning:
            execution_plan = "PLAN_NOT_GENERATED: skip_planning is true"
        elif not original_question:
            execution_plan = "PLAN_NOT_GENERATED: original_question is empty"
        else:
            execution_plan = "PLAN_NOT_GENERATED: planning output is empty"

    global_state.execution_plan = execution_plan

    # 使用进度记录器输出完成信息
    _progress_logger.log_info(f"Immunity 子图完成:")
    _progress_logger.log_info(f"  - 优化查询数: {len(optimized_questions)}")
    _progress_logger.log_info(f"  - 研究置信度: {research_confidence:.1f}%")
    _progress_logger.log_info(f"  - 假设置信度: {hypothesis_confidence:.1f}%")
    _progress_logger.log_info(
        f"  - 计划文档长度: {len(final_enhanced_plan or '')} 字符"
    )

    # 🔥 推送生成的文件内容到前端（通过SSE）
    if hasattr(global_state, "progress_callback") and global_state.progress_callback:
        try:
            # 获取session_id和沙盒目录
            session_id = getattr(global_state, "session_id", None)
            sandbox_dir = getattr(global_state, "sandbox_dir", None)

            if session_id and sandbox_dir:
                # 推送执行计划
                if final_enhanced_plan or execution_plan:
                    plan_content = final_enhanced_plan or execution_plan
                    global_state.progress_callback(
                        event_type="file_content",
                        message=f"📄 实验计划生成完成",
                        details={
                            "file_type": "execution_plan",
                            "file_name": "planning_report.md",
                            "content": plan_content,
                            "content_length": len(plan_content),
                            "node": "immunity",
                        },
                    )
                    _progress_logger.log_info(
                        f"  ✅ 已推送实验计划到前端 ({len(plan_content)} 字符)"
                    )

                # 推送研究报告（如果有）
                if research_summary:
                    global_state.progress_callback(
                        event_type="file_content",
                        message=f"📚 研究报告已生成",
                        details={
                            "file_type": "research_report",
                            "file_name": "research_report.md",
                            "content": research_summary,
                            "content_length": len(research_summary),
                            "node": "immunity",
                        },
                    )
                    _progress_logger.log_info(
                        f"  ✅ 已推送研究报告到前端 ({len(research_summary)} 字符)"
                    )

                # 推送假设生成报告（如果有）
                if hypothesis_summary:
                    global_state.progress_callback(
                        event_type="file_content",
                        message=f"🧬 假设生成报告已生成",
                        details={
                            "file_type": "hypothesis_report",
                            "file_name": "hypothesis_report.md",
                            "content": hypothesis_summary,
                            "content_length": len(hypothesis_summary),
                            "node": "immunity",
                        },
                    )
                    _progress_logger.log_info(
                        f"  ✅ 已推送假设报告到前端 ({len(hypothesis_summary)} 字符)"
                    )

                # 推送评估报告（如果有）
                if final_evaluation:
                    global_state.progress_callback(
                        event_type="file_content",
                        message=f"📊 评估报告已生成",
                        details={
                            "file_type": "evaluation_report",
                            "file_name": "evaluation_report.md",
                            "content": final_evaluation,
                            "content_length": len(final_evaluation),
                            "node": "immunity",
                        },
                    )
                    _progress_logger.log_info(
                        f"  ✅ 已推送评估报告到前端 ({len(final_evaluation)} 字符)"
                    )

        except Exception as e:
            _progress_logger.log_warning(f"推送文件内容到前端失败: {e}")
            import traceback

            traceback.print_exc()

    # 结束工作流
    _progress_logger.end_workflow(success=True)

    return global_state


# ===================== Build Immunity Subgraph =====================


def _should_skip_stages(state: ImmunityState) -> str:
    """
    条件路由：判断是否跳过后续阶段

    Returns:
        "skip_to_planning": 跳过中间阶段，直接到 planning（用于缓存命中）
        "continue_retrieval": 继续执行 retrieval 阶段
    """
    if state.skip_immunity_stages or state.cache_hit:
        return "skip_to_planning"
    return "continue_retrieval"


def _should_skip_to_evaluation(state: ImmunityState) -> str:
    """
    条件路由：判断是否跳过到 evaluation（缓存命中时跳过 planning）

    Returns:
        "skip_to_evaluation": 跳过 planning，直接到 evaluation
        "continue_planning": 继续执行 planning 阶段
    """
    if state.skip_immunity_stages or state.cache_hit:
        return "skip_to_evaluation"
    return "continue_planning"


def build_immunity_subgraph():
    """
    Build Immunity Agent subgraph

    Complete workflow:
    Cache Check → Query Decomposition → Retrieval → Deep Research → Hypothesis Generation → Planning ⭐ → Evaluation

    如果缓存命中，则跳过中间阶段直接使用缓存结果

    Returns:
        Compiled subgraph
    """
    graph = StateGraph(ImmunityState)

    # Add all nodes
    graph.add_node("cache_check", cache_check_node)  # Stage 0: Cache Check (NEW!)
    graph.add_node("query_decomposition", query_decomposition_node)  # Stage 1
    graph.add_node("retrieval", retrieval_node)  # Stage 2: Retrieval node
    graph.add_node("deep_research", deep_research_node)  # Stage 3
    graph.add_node("hypothesis_generation", hypothesis_generation_node)  # Stage 4
    graph.add_node("planning", planning_node)  # Stage 5 ⭐
    graph.add_node("evaluation", evaluation_node)  # Stage 6

    # Define flow rules
    graph.add_edge(START, "cache_check")  # 首先检查缓存
    graph.add_edge(
        "cache_check", "query_decomposition"
    )  # 无论缓存是否命中，都执行 query_decomposition（用于记录）

    # 条件路由：query_decomposition 后决定是否跳过中间阶段
    graph.add_conditional_edges(
        "query_decomposition",
        _should_skip_stages,
        {
            "skip_to_planning": "planning",  # 缓存命中，跳过中间阶段
            "continue_retrieval": "retrieval",  # 正常流程
        },
    )

    # 正常流程边
    graph.add_edge("retrieval", "deep_research")
    graph.add_edge("deep_research", "hypothesis_generation")
    graph.add_edge("hypothesis_generation", "planning")

    # 条件路由：planning 后决定是否跳过到 evaluation
    # 注意：即使缓存命中，我们也执行 planning 节点（用于生成最终计划）
    graph.add_edge("planning", "evaluation")
    graph.add_edge("evaluation", END)

    return graph.compile()
