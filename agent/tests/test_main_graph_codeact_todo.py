"""
Main Graph CodeAct Todo 模式测试

测试流程: supervisor => task_decomposition => executor (CodeAct Todo 模式)
跳过 immunity 子图（因为耗时较长）

测试用例: Q13 - MART-1 癌症表位 TCR 结合预测

运行方法:
    cd D:\\projects-开保智药\\bio-agent\\agent
    python -m pytest tests/test_main_graph_codeact_todo.py -v -s --tb=short -k "test_codeact_todo_flow"
    # 或者直接运行
    python tests/test_main_graph_codeact_todo.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
import traceback

# 确保在 agent 目录
agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

# 设置环境变量
os.environ["OPENSANDBOX_ENABLED"] = "true"
os.environ["CODEACT_SANDBOX_PROVIDER"] = "opensandbox"


# =============================================================================
# 测试日志记录器
# =============================================================================

class TestLogger:
    """
    简洁的测试日志记录器
    
    只记录关键信息，生成人类可读的报告
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.log_dir = agent_dir / "logs" / "test_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_file = self.log_dir / f"report_{session_id}_{timestamp}.md"
        
        # 收集关键信息
        self.sections = {}
        self.errors = []
        self.current_section = None
        
        # 写入报告头部
        self._write_header()
    
    def _write_header(self):
        """写入报告头部"""
        header = f"""# CodeAct Todo 测试报告

**Session ID**: `{self.session_id}`  
**时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

"""
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def section(self, name: str):
        """开始一个新章节"""
        self.current_section = name
        self.sections[name] = {"content": [], "data": {}}
        self._append(f"\n## {name}\n")
    
    def _append(self, text: str):
        """追加文本到报告"""
        with open(self.report_file, 'a', encoding='utf-8') as f:
            f.write(text)
    
    def key_value(self, key: str, value: Any):
        """记录键值对"""
        if isinstance(value, str) and len(value) > 100:
            value = value[:100] + "..."
        self._append(f"- **{key}**: `{value}`\n")
    
    def key_values(self, data: Dict[str, Any]):
        """批量记录键值对"""
        for k, v in data.items():
            self.key_value(k, v)
    
    def code_block(self, title: str, content: str, language: str = ""):
        """记录代码块"""
        self._append(f"\n### {title}\n\n```{language}\n{content}\n```\n")
    
    def table(self, title: str, headers: List[str], rows: List[List[str]]):
        """记录表格"""
        self._append(f"\n### {title}\n\n")
        self._append("| " + " | ".join(headers) + " |\n")
        self._append("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            # 截断过长的单元格
            row = [str(c)[:50] + "..." if len(str(c)) > 50 else str(c) for c in row]
            self._append("| " + " | ".join(row) + " |\n")
    
    def success(self, msg: str):
        """记录成功"""
        self._append(f"\n✅ {msg}\n")
    
    def warning(self, msg: str):
        """记录警告"""
        self._append(f"\n⚠️ {msg}\n")
    
    def error(self, msg: str, details: str = None):
        """记录错误"""
        self._append(f"\n❌ **错误**: {msg}\n")
        if details:
            self._append(f"\n```\n{details}\n```\n")
        self.errors.append({"msg": msg, "details": details})
    
    def divider(self):
        """分隔线"""
        self._append("\n---\n")
    
    def log_parameter_table(self, param_table: Any, title: str = "参数表"):
        """记录参数表 - 聚焦关键信息"""
        if not param_table:
            self.warning(f"{title}: 无数据")
            return
        
        rows = []
        if hasattr(param_table, 'files') and param_table.files:
            for f in param_table.files:
                path = getattr(f, 'path', str(f))
                desc = getattr(f, 'description', '')[:30]
                rows.append([path, desc])
            self.table(title, ["文件路径", "描述"], rows)
        elif isinstance(param_table, dict):
            self.key_values(param_table)
    
    def log_todo_list(self, todo_list: Any, title: str = "Todo List"):
        """记录 todo-list - 聚焦任务列表"""
        if not todo_list:
            self.warning(f"{title}: 无任务")
            return
        
        tasks = []
        if hasattr(todo_list, 'tasks'):
            for task in todo_list.tasks:
                task_id = getattr(task, 'id', '?')
                task_type = getattr(task, 'type', 'unknown')
                if hasattr(task_type, 'value'):
                    task_type = task_type.value
                status = getattr(task, 'status', 'unknown')
                if hasattr(status, 'value'):
                    status = status.value
                desc = getattr(task, 'description', '')[:40]
                tasks.append([task_id, task_type, status, desc])
        
        if tasks:
            self.table(title, ["ID", "类型", "状态", "描述"], tasks)
        else:
            self.warning(f"{title}: 无任务")
    
    def log_codeact_execution(self, task_desc: str, code: str, result: Any, error: str = None):
        """记录 CodeAct 执行 - 核心调试信息"""
        self._append(f"\n### CodeAct 执行\n\n")
        
        # 任务描述
        self._append(f"**任务**: {task_desc[:100]}...\n\n")
        
        # 执行的代码（关键！）
        if code:
            self._append("**执行代码**:\n")
            # 只显示关键代码，截断过长的
            code_preview = code[:1500] if len(code) > 1500 else code
            self._append(f"```python\n{code_preview}\n```\n")
        
        # 执行结果
        if result:
            self._append("\n**执行结果**:\n")
            if hasattr(result, 'is_success'):
                status = "✅ 成功" if result.is_success() else "❌ 失败"
                self._append(f"- 状态: {status}\n")
                if hasattr(result, 'output') and result.output:
                    output = result.output[:500]
                    self._append(f"- 输出: `{output}`\n")
                if hasattr(result, 'error') and result.error:
                    self._append(f"- 错误: `{result.error}`\n")
                if hasattr(result, 'returncode') and result.returncode is not None:
                    self._append(f"- 返回码: `{result.returncode}`\n")
            elif isinstance(result, dict):
                self.key_values({k: v for k, v in result.items() if k in ['status', 'error', 'output', 'returncode']})
        
        # 错误详情
        if error:
            self.error(error)
    
    def finalize(self):
        """完成报告，添加总结"""
        self._append("\n---\n\n## 总结\n\n")
        
        if self.errors:
            self._append(f"### 错误列表 ({len(self.errors)} 个)\n\n")
            for i, err in enumerate(self.errors, 1):
                self._append(f"{i}. {err['msg']}\n")
        else:
            self._append("✅ **测试通过，无错误**\n")
        
        self._append(f"\n---\n*报告生成时间: {datetime.now().isoformat()}*\n")
        
        print(f"\n📄 测试报告已保存: {self.report_file}")


# 全局测试日志记录器
test_logger: Optional[TestLogger] = None


def get_test_logger() -> TestLogger:
    """获取测试日志记录器"""
    global test_logger
    if test_logger is None:
        test_logger = TestLogger(session_id="default")
    return test_logger


# =============================================================================
# Q13 测试用例配置
# =============================================================================

Q13_PROMPT = """Given 2080 T cell receptors with paired CDR3 alpha (CDR3a) and CDR3 beta (CDR3b) sequences, predict which TCRs bind the MART-1 cancer epitope (peptide: ELAGIGILTV, presented by HLA-A*02:01). MART-1 is a melanoma-associated antigen. For each TCR (identified by `main_name`), output a binary prediction: True = binder, False = non-binder.

What to use:
- CDR3a and CDR3b amino acid sequences for each TCR
- Target peptide: ELAGIGILTV
- HLA restriction: A*02:01
- TCR V/J gene usage annotations
- TCR-epitope binding prediction tools (e.g., NetTCR-2.0)

Expected output format:
- CSV with columns: `main_name`, `prediction` (True or False), optionally `probability` (0.0-1.0)
- 2080 rows (one per TCR)

Ground truth summary:
- 2080 TCRs tested
- 60 positive (2.9%), 2020 negative
- Highly imbalanced -- only ~3% are MART-1 binders

Primary metric: F1

Input files:
- meta_csv_file: /data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv
- meta_rds_file: /data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_benchmark.rds
"""

Q13_FILE_PATHS = {
    "meta_csv_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv",
    "meta_rds_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_benchmark.rds",
}


# =============================================================================
# 辅助函数
# =============================================================================

def create_test_state(skip_immunity: bool = True):
    """
    创建测试用例的 GlobalState
    
    Args:
        skip_immunity: 如果为 True，设置 task_type 为 EXECUTE_PLAN 跳过 immunity
    """
    from state import GlobalState, UserTaskType
    
    # 生成 session ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{timestamp}_codeact_todo_test"
    
    # 创建沙盒目录
    sandbox_dir = f"/data/sessions/{session_id}"
    sandbox_data_dir = f"/data/sessions/{session_id}"
    
    # 设置 task_type: EXECUTE_PLAN 会跳过 immunity，直接到 task_decomposition
    task_type = UserTaskType.EXECUTE_PLAN if skip_immunity else UserTaskType.IMMUNOLOGY_TASK
    
    # 创建 GlobalState
    state = GlobalState(
        user_input=Q13_PROMPT,
        user_task_type=task_type,
        sandbox_dir=sandbox_dir,
        sandbox_data_dir=sandbox_data_dir,
        session_id=session_id,
        file_paths=Q13_FILE_PATHS.copy()
    )
    
    return state, session_id


def print_section(title: str):
    """打印分节标题"""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title: str):
    """打印子分节标题"""
    print()
    print("-" * 60)
    print(f"  {title}")
    print("-" * 60)


# =============================================================================
# 测试函数
# =============================================================================

def test_imports():
    """测试所有必要的导入"""
    print_section("Step 0: 测试导入")
    
    errors = []
    
    # 测试主图导入
    try:
        from main_graph import (
            build_main_graph,
            supervisor_node,
            task_decomposition_node,
            executor_node,
            _codeact_input_mapper,
            _codeact_output_mapper,
            CODEACT_SUBGRAPH_AVAILABLE,
            TODOLIST_GENERATOR_AVAILABLE,
        )
        print("  ✅ main_graph 导入成功")
        print(f"     - CODEACT_SUBGRAPH_AVAILABLE: {CODEACT_SUBGRAPH_AVAILABLE}")
        print(f"     - TODOLIST_GENERATOR_AVAILABLE: {TODOLIST_GENERATOR_AVAILABLE}")
    except ImportError as e:
        errors.append(f"main_graph 导入失败: {e}")
        print(f"  ❌ main_graph 导入失败: {e}")
    
    # 测试 CodeAct 子图导入
    try:
        from nodes.subagents.code_act import (
            build_codeact_subgraph,
            CodeActState,
            CodeActExecutionMode,
            TodoListManager,
            TodoTask,
            TodoTaskStatus,
        )
        print("  ✅ code_act 子图导入成功")
    except ImportError as e:
        errors.append(f"code_act 导入失败: {e}")
        print(f"  ❌ code_act 导入失败: {e}")
    
    # 测试 TodoList Generator 导入
    try:
        from nodes.subagents.executor.todolist_generator import (
            convert_subtasks_to_todolist,
            generate_and_save_todolist_from_state,
        )
        print("  ✅ todolist_generator 导入成功")
    except ImportError as e:
        errors.append(f"todolist_generator 导入失败: {e}")
        print(f"  ❌ todolist_generator 导入失败: {e}")
    
    # 测试 GlobalState
    try:
        from state import GlobalState, UserTaskType, SubTask
        print("  ✅ state 导入成功")
    except ImportError as e:
        errors.append(f"state 导入失败: {e}")
        print(f"  ❌ state 导入失败: {e}")
    
    if errors:
        print()
        print("导入错误:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    return True


def test_supervisor_node(state):
    """测试 supervisor 节点"""
    print_section("Step 1: Supervisor 节点")
    logger = get_test_logger()
    
    from main_graph import supervisor_node
    
    logger.section("1. Supervisor")
    logger.key_values({
        "输入 session_id": state.session_id,
        "输入 task_type": str(state.user_task_type)
    })
    
    try:
        result = supervisor_node(state)
        
        logger.key_values({
            "输出 session_id": result.session_id,
            "sandbox_dir": result.sandbox_dir,
            "opensandbox_id": result.opensandbox_id or "N/A"
        })
        logger.success("Supervisor 节点完成")
        
        return result, True
        
    except Exception as e:
        logger.error(f"Supervisor 节点失败: {e}", traceback.format_exc())
        return state, False


def test_task_decomposition_node(state):
    """测试 task_decomposition 节点"""
    print_section("Step 2: Task Decomposition 节点")
    logger = get_test_logger()
    
    from main_graph import task_decomposition_node
    
    logger.section("2. Task Decomposition")
    
    try:
        result = task_decomposition_node(state)
        
        num_subtasks = len(result.subtasks) if result.subtasks else 0
        num_groups = len(result.parallel_task_groups) if result.parallel_task_groups else 0
        
        logger.key_values({
            "子任务数量": num_subtasks,
            "并行任务组数量": num_groups
        })
        
        # 记录任务列表
        if result.subtasks:
            tasks_table = []
            for i, task in enumerate(result.subtasks[:5]):
                tasks_table.append([
                    task.task_id,
                    task.content[:50],
                    task.parameters.get("tool_name", "") if hasattr(task, 'parameters') else ""
                ])
            logger.table("子任务列表", ["Task ID", "内容", "工具"], tasks_table)
        
        logger.success("Task Decomposition 节点完成")
        return result, True
        
    except Exception as e:
        logger.error(f"Task Decomposition 节点失败: {e}", traceback.format_exc())
        return state, False


def test_todolist_generation(state):
    """测试 todo-list.md 生成"""
    print_subsection("Step 2.5: 生成 todo-list.md")
    logger = get_test_logger()
    
    from nodes.subagents.executor.todolist_generator import generate_and_save_todolist_from_state
    
    logger.section("2.5 Todo List 生成")
    
    if not state.subtasks:
        logger.warning("没有子任务，跳过 todo-list 生成")
        return None, False
    
    try:
        opensandbox_id = state.opensandbox_id
        if not opensandbox_id and state.merged_result:
            opensandbox_id = state.merged_result.get('opensandbox_id')
        
        logger.key_value("opensandbox_id", opensandbox_id or "N/A")
        
        todo_list = generate_and_save_todolist_from_state(
            global_state=state,
            opensandbox_id=opensandbox_id
        )
        
        if todo_list:
            logger.log_todo_list(todo_list, "生成的 Todo List")
            logger.success(f"Todo list 生成成功: {len(todo_list.tasks)} 个任务")
            return todo_list, True
        else:
            logger.error("Todo list 生成失败")
            return None, False
            
    except Exception as e:
        logger.error(f"Todo list 生成失败: {e}", traceback.format_exc())
        return None, False


def test_codeact_todo_node(state):
    """测试 executor 节点 (CodeAct Todo 模式)"""
    print_section("Step 3: Executor 节点 (CodeAct Todo 模式)")
    logger = get_test_logger()
    
    from main_graph import executor_node
    
    logger.section("3. Executor (CodeAct Todo)")
    logger.key_values({
        "输入 session_id": state.session_id,
        "subtasks 数量": len(state.subtasks) if state.subtasks else 0
    })
    
    try:
        result = executor_node(state)
        
        # 记录执行结果
        exec_results = {}
        if result.merged_result and "executor_results" in result.merged_result:
            exec_results = result.merged_result["executor_results"]
        
        logger.key_values({
            "总任务数": exec_results.get('total_tasks', 0),
            "已完成": exec_results.get('completed_count', 0),
            "失败": exec_results.get('failed_count', 0),
            "todo_list_path": exec_results.get('todo_list_path', 'N/A')
        })
        
        logger.success("Executor 节点完成")
        return result, True
        
    except Exception as e:
        logger.error(f"Executor 节点失败: {e}", traceback.format_exc())
        return state, False


def test_mappers():
    """测试 mapper 函数"""
    print_subsection("测试 Mapper 函数")
    logger = get_test_logger()
    
    from main_graph import _codeact_input_mapper, _codeact_output_mapper
    from state import GlobalState, UserTaskType, SubTask
    from nodes.subagents.code_act import CodeActState, CodeActExecutionMode, TodoList, TodoTask, TodoTaskStatus, TodoListSession
    
    logger.section("0. Mapper 函数测试")
    
    # 创建测试状态
    test_state = GlobalState(
        user_input="测试输入",
        user_task_type=UserTaskType.EXECUTE_PLAN,
        session_id="test_session_123",
        sandbox_dir="/data/sessions/test_session_123",
        subtasks=[
            SubTask(
                task_id="task_1",
                task_type=UserTaskType.EXECUTE_PLAN,
                content="测试任务",
                dependencies=[],
                parallel_group_id=None
            )
        ]
    )
    
    # 测试 input mapper
    print("  测试 _codeact_input_mapper...")
    try:
        codeact_state = _codeact_input_mapper(test_state)
        sandbox_dir = codeact_state.parameters.get("sandbox_dir", "N/A")
        logger.key_values({
            "todo_list_path": codeact_state.todo_list_path,
            "execution_mode": str(codeact_state.execution_mode),
            "sandbox_dir": sandbox_dir
        })
        print(f"    ✅ 输入映射成功")
    except Exception as e:
        logger.error(f"输入映射失败: {e}", traceback.format_exc())
        return False
    
    # 测试 output mapper
    print("  测试 _codeact_output_mapper...")
    try:
        # 模拟 CodeAct 执行结果
        codeact_state.todo_list = TodoList(
            session=TodoListSession(
                session_id="test_session_123",
                created_at=datetime.now().isoformat(),
                sandbox_dir="/data/sessions/test_session_123"
            ),
            tasks=[
                TodoTask(
                    id="task_1",
                    type="general",
                    status=TodoTaskStatus.COMPLETED,
                    priority=5,
                    description="测试任务",
                    parameters={},
                    result={"status": "success", "output": "test output"}
                )
            ]
        )
        
        result_state = _codeact_output_mapper(codeact_state, test_state)
        logger.key_values({
            "completed_tasks": len(result_state.completed_tasks),
        })
        print(f"    ✅ 输出映射成功")
    except Exception as e:
        logger.error(f"输出映射失败: {e}", traceback.format_exc())
        return False
    
    logger.success("Mapper 函数测试通过")
    return True


# =============================================================================
# 主测试流程
# =============================================================================

def test_codeact_todo_flow():
    """
    测试完整流程: supervisor => task_decomposition => executor (CodeAct Todo)
    """
    global test_logger
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "Main Graph CodeAct Todo 模式测试" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    print("测试流程: supervisor => task_decomposition => executor (CodeAct Todo)")
    print("测试用例: Q13 - MART-1 癌症表位 TCR 结合预测")
    print()
    
    # Step 0: 测试导入
    if not test_imports():
        print()
        print("❌ 导入测试失败，终止测试")
        return False
    
    # 测试 mapper 函数
    if not test_mappers():
        print()
        print("❌ Mapper 测试失败，终止测试")
        return False
    
    # 创建测试状态
    print_section("创建测试状态")
    state, session_id = create_test_state(skip_immunity=True)
    print(f"  Session ID: {session_id}")
    print(f"  Task Type: {state.user_task_type} (跳过 immunity)")
    print(f"  Sandbox Dir: {state.sandbox_dir}")
    
    # 初始化测试日志记录器
    test_logger = TestLogger(session_id=session_id)
    test_logger.key_values({
        "session_id": session_id,
        "task_type": str(state.user_task_type),
        "sandbox_dir": state.sandbox_dir
    })
    test_logger.divider()
    
    # Step 1: Supervisor
    state, success = test_supervisor_node(state)
    if not success:
        print()
        print("❌ Supervisor 测试失败，终止测试")
        test_logger.error("测试终止: Supervisor 失败")
        test_logger.finalize()
        return False
    
    # Step 2: Task Decomposition
    state, success = test_task_decomposition_node(state)
    if not success:
        print()
        print("❌ Task Decomposition 测试失败，终止测试")
        test_logger.error("测试终止: Task Decomposition 失败")
        test_logger.finalize()
        return False
    
    # Step 2.5: 生成 todo-list.md
    todo_list, success = test_todolist_generation(state)
    
    # Step 3: Executor (CodeAct Todo 模式)
    state, success = test_codeact_todo_node(state)
    if not success:
        print()
        print("❌ Executor 测试失败")
    
    # 总结
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 25 + "测试总结" + " " * 35 + "║")
    print("╠" + "=" * 68 + "╣")
    print(f"║  Session ID: {session_id:<54}║")
    print(f"║  Sandbox: {state.sandbox_dir:<55}║")
    
    if state.merged_result and "executor_results" in state.merged_result:
        exec_results = state.merged_result["executor_results"]
        print(f"║  总任务数: {exec_results.get('total_tasks', 0):<52}║")
        print(f"║  已完成: {exec_results.get('completed_count', 0):<55}║")
        print(f"║  失败: {exec_results.get('failed_count', 0):<57}║")
    
    print("╚" + "=" * 68 + "╝")
    print()
    
    # 完成日志
    test_logger.finalize()
    
    return True


def test_executor_standalone():
    """
    单独测试 executor 节点 - 使用模拟数据，详细记录执行过程
    
    记录内容：
    1. todo-list.md 内容
    2. 每个任务的描述
    3. 生成的代码
    4. 执行结果或报错
    """
    global test_logger
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{timestamp}_executor_only"
    
    # 初始化日志
    test_logger = TestLogger(session_id=session_id)
    test_logger.section("Executor 节点独立测试")
    test_logger.key_value("模式", "使用模拟数据，详细记录执行过程")
    test_logger.divider()
    
    # 导入必要模块
    print("导入模块...")
    from main_graph import executor_node, _codeact_input_mapper, _codeact_output_mapper, _get_codeact_subgraph
    from state import GlobalState, UserTaskType, SubTask, ensure_global_state_rebuilt
    from nodes.subagents.executor.todolist_generator import generate_and_save_todolist_from_state
    
    # 确保GlobalState模型已重建以支持todo_list字段
    ensure_global_state_rebuilt()
    
    # 构造模拟的 GlobalState
    print("\n构造模拟 GlobalState...")
    
    # 模拟 subtasks
    mock_subtasks = [
        SubTask(
            task_id="test_task_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="读取 /data 目录下的文件列表，输出前5个文件名",
            dependencies=[],
            parallel_group_id=None,
            parameters={"tool_name": "codeact"}
        )
    ]
    
    # 获取 sandbox
    opensandbox_id = os.environ.get("TEST_OPENSANDBOX_ID")
    
    state = GlobalState(
        user_input="测试 executor 节点",
        user_task_type=UserTaskType.EXECUTE_PLAN,
        session_id=session_id,
        sandbox_dir=f"/data/sessions/{session_id}",
        sandbox_data_dir=f"/data/sessions/{session_id}",
        subtasks=mock_subtasks,
        opensandbox_id=opensandbox_id,
        merged_result={}
    )
    
    test_logger.key_values({
        "session_id": session_id,
        "sandbox_dir": state.sandbox_dir,
        "subtasks 数量": len(mock_subtasks),
        "opensandbox_id": opensandbox_id or "无（将创建新 sandbox）"
    })
    test_logger.divider()
    
    # ========== Step 1: 生成 todo-list.md ==========
    test_logger.section("Step 1: 生成 todo-list.md")
    
    try:
        todo_list = generate_and_save_todolist_from_state(
            global_state=state,
            opensandbox_id=opensandbox_id
        )
        
        if todo_list:
            test_logger.success(f"todo-list.md 生成成功: {len(todo_list.tasks)} 个任务")
            
            # 记录完整的 todo-list 内容
            test_logger.log_todo_list(todo_list, "Todo List 内容")
            
            # 更新 state
            state.todo_list = todo_list
        else:
            test_logger.warning("todo-list.md 生成失败")
    except Exception as e:
        test_logger.error(f"生成 todo-list.md 失败: {e}", traceback.format_exc())
    
    test_logger.divider()
    
    # ========== Step 2: 映射到 CodeAct 状态 ==========
    test_logger.section("Step 2: 映射到 CodeAct 状态")
    
    try:
        codeact_state = _codeact_input_mapper(state)
        test_logger.key_values({
            "todo_list_path": codeact_state.todo_list_path or "N/A",
            "execution_mode": str(codeact_state.execution_mode)
        })
    except Exception as e:
        test_logger.error(f"映射失败: {e}", traceback.format_exc())
        test_logger.finalize()
        return False
    
    test_logger.divider()
    
    # ========== Step 3: 执行 CodeAct 子图 ==========
    test_logger.section("Step 3: 执行 CodeAct 子图")
    
    subgraph = _get_codeact_subgraph()
    if subgraph is None:
        test_logger.error("CodeAct 子图不可用")
        test_logger.finalize()
        return False
    
    print("\n执行 CodeAct 子图...")
    test_logger.key_value("流程", "read_todo → select_next_task → infer_params → generate_code → execute_code → update_todo")
    
    try:
        # 使用 stream 模式获取中间状态
        # 这样可以捕获每个节点执行后的状态
        print("\n[开始执行子图...]")
        
        result_state = None
        step_count = 0
        
        # 使用 stream 获取每个节点的输出
        for event in subgraph.stream(codeact_state):
            step_count += 1
            node_name = list(event.keys())[0] if event else "unknown"
            node_output = event.get(node_name, {}) if event else {}
            
            print(f"\n  [Step {step_count}] 节点: {node_name}")
            
            # 记录关键信息
            if node_output:
                # 处理 dict 和对象两种情况
                def get_value(obj, key, default=None):
                    """Helper to get value from dict or object"""
                    if isinstance(obj, dict):
                        return obj.get(key, default)
                    return getattr(obj, key, default)
                
                # 记录生成的代码
                generated_code = get_value(node_output, 'generated_code')
                if generated_code:
                    test_logger.code_block(
                        f"Step {step_count} - {node_name}: 生成的代码",
                        generated_code,
                        "python"
                    )
                
                # 记录执行结果
                execution_result = get_value(node_output, 'execution_result')
                if execution_result:
                    if isinstance(execution_result, dict):
                        test_logger.key_values({
                            f"Step {step_count} 执行状态": execution_result.get('status', 'unknown'),
                            f"Step {step_count} 输出": str(execution_result.get('output', ''))[:200]
                        })
                        if execution_result.get('error'):
                            test_logger.error(f"Step {step_count} 执行错误: {execution_result.get('error')}")
                    else:
                        test_logger.key_value(f"Step {step_count} 执行结果", str(execution_result)[:200])
                
                # 记录当前任务
                current_todo_task = get_value(node_output, 'current_todo_task')
                if current_todo_task:
                    task_id = get_value(current_todo_task, 'id', 'unknown')
                    task_desc = get_value(current_todo_task, 'description', '')
                    test_logger.key_values({
                        f"Step {step_count} 当前任务 ID": task_id,
                        f"Step {step_count} 任务描述": task_desc[:100] if task_desc else "N/A"
                    })
                
                # 记录 todo list 进度
                todo_list = get_value(node_output, 'todo_list')
                if todo_list:
                    tasks = get_value(todo_list, 'tasks', [])
                    if tasks:
                        completed = sum(1 for t in tasks if str(get_value(t, 'status', '')).endswith('COMPLETED'))
                        total = len(tasks)
                        test_logger.key_value(f"Step {step_count} 任务进度", f"{completed}/{total}")
                
                # 记录 revision 信息
                revision_plan = get_value(node_output, 'revision_plan')
                if revision_plan:
                    strategy = get_value(revision_plan, 'strategy', 'unknown')
                    root_cause = get_value(revision_plan, 'root_cause', '')[:100]
                    test_logger.key_values({
                        f"Step {step_count} Revision 策略": strategy,
                        f"Step {step_count} 根本原因": root_cause + "..."
                    })
                
                # 记录 revision iteration
                revision_iter = get_value(node_output, 'revision_iteration')
                if revision_iter and revision_iter > 0:
                    test_logger.key_value(f"Step {step_count} Revision 迭代", revision_iter)
            
            result_state = node_output
        
        print("\n[子图执行完成]")
        
    except Exception as e:
        test_logger.error(f"CodeAct 子图执行失败: {e}", traceback.format_exc())
        result_state = None
    
    test_logger.divider()
    
    # ========== Step 4: 映射结果回全局状态 ==========
    test_logger.section("Step 4: 执行结果")
    
    if result_state:
        try:
            state = _codeact_output_mapper(result_state, state)
            
            # 记录最终结果
            if state.merged_result and "executor_results" in state.merged_result:
                exec_results = state.merged_result["executor_results"]
                test_logger.key_values({
                    "总任务数": exec_results.get('total_tasks', 0),
                    "已完成": exec_results.get('completed_count', 0),
                    "失败": exec_results.get('failed_count', 0),
                    "todo_list_path": exec_results.get('todo_list_path', 'N/A')
                })
            
            # 记录完成的任务
            if state.completed_tasks:
                test_logger.section("已完成的任务")
                for task in state.completed_tasks:
                    if hasattr(task, 'task_id'):
                        test_logger.key_values({
                            "任务 ID": task.task_id,
                            "结果": str(getattr(task, 'result', 'N/A'))[:200]
                        })
            
            test_logger.success("Executor 节点执行完成")
            
        except Exception as e:
            test_logger.error(f"结果映射失败: {e}", traceback.format_exc())
    else:
        test_logger.error("CodeAct 子图返回空结果")
    
    test_logger.finalize()
    return True


def main():
    """主入口"""
    global test_logger
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Main Graph CodeAct Todo 模式测试")
    parser.add_argument("--imports", action="store_true", help="仅测试导入")
    parser.add_argument("--mappers", action="store_true", help="仅测试 mapper 函数")
    parser.add_argument("--todolist", action="store_true", help="仅测试 todo-list 生成")
    parser.add_argument("--executor", action="store_true", help="单独测试 executor 节点（使用模拟数据）")
    args = parser.parse_args()
    
    if args.executor:
        test_executor_standalone()
    elif args.imports:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_logger = TestLogger(session_id=f"imports_{timestamp}")
        test_imports()
        test_logger.finalize()
    elif args.mappers:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_logger = TestLogger(session_id=f"mappers_{timestamp}")
        test_imports()
        test_mappers()
        test_logger.finalize()
    elif args.todolist:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_logger = TestLogger(session_id=f"todolist_{timestamp}")
        test_imports()
        state, _ = create_test_state()
        state, _ = test_task_decomposition_node(state)
        test_todolist_generation(state)
        test_logger.finalize()
    else:
        test_codeact_todo_flow()


if __name__ == "__main__":
    main()

