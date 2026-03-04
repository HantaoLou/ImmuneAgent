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
    详细测试日志记录器
    
    完整记录所有信息，无长度限制，生成人类可读的报告
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
        
        # 任务迭代计数器
        self.task_iteration_counts = {}
        
        # 写入报告头部
        self._write_header()
    
    def _write_header(self):
        """写入报告头部"""
        header = f"""# CodeAct Todo 测试报告（详细版）

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
    
    def subsection(self, name: str):
        """开始一个子章节"""
        self._append(f"\n### {name}\n")
    
    def _append(self, text: str):
        """追加文本到报告"""
        with open(self.report_file, 'a', encoding='utf-8') as f:
            f.write(text)
    
    def key_value(self, key: str, value: Any, no_truncate: bool = True):
        """记录键值对 - 默认不截断"""
        if no_truncate:
            # 完整记录，不截断
            if isinstance(value, str):
                self._append(f"- **{key}**:\n  `{value}`\n")
            else:
                self._append(f"- **{key}**: `{value}`\n")
        else:
            # 旧版本行为：截断
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            self._append(f"- **{key}**: `{value}`\n")
    
    def key_values(self, data: Dict[str, Any], no_truncate: bool = True):
        """批量记录键值对"""
        for k, v in data.items():
            self.key_value(k, v, no_truncate=no_truncate)
    
    def code_block(self, title: str, content: str, language: str = ""):
        """记录代码块 - 完整记录，不截断"""
        self._append(f"\n### {title}\n\n```{language}\n{content}\n```\n")
    
    def json_block(self, title: str, data: Any):
        """记录 JSON 数据块 - 格式化输出"""
        try:
            if isinstance(data, (dict, list)):
                content = json.dumps(data, indent=2, ensure_ascii=False, default=str)
            else:
                content = str(data)
            self.code_block(title, content, "json")
        except Exception as e:
            self._append(f"\n### {title}\n\n```\n{data}\n```\n")
    
    def table(self, title: str, headers: List[str], rows: List[List[str]], no_truncate: bool = True):
        """记录表格 - 默认不截断"""
        self._append(f"\n### {title}\n\n")
        self._append("| " + " | ".join(headers) + " |\n")
        self._append("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            if no_truncate:
                # 不截断
                row = [str(c).replace("\n", "<br>") for c in row]
            else:
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
        """记录错误 - 完整记录错误详情"""
        self._append(f"\n❌ **错误**: {msg}\n")
        if details:
            self._append(f"\n<details>\n<summary>错误详情</summary>\n\n```\n{details}\n```\n</details>\n")
        self.errors.append({"msg": msg, "details": details})
    
    def divider(self):
        """分隔线"""
        self._append("\n---\n")
    
    def log_parameter_table_full(self, param_table: Any, title: str = "参数表"):
        """完整记录参数表 - 包含所有字段，不截断"""
        if not param_table:
            self.warning(f"{title}: 无数据")
            return
        
        self.subsection(title)
        
        # 尝试获取完整属性
        if hasattr(param_table, '__dict__'):
            self._append("\n**参数表完整内容**:\n")
            for attr, value in vars(param_table).items():
                if attr.startswith('_'):
                    continue
                self._append(f"\n#### {attr}\n")
                if attr == 'files' and value:
                    for i, f in enumerate(value, 1):
                        self._append(f"\n**文件 {i}**:\n")
                        if hasattr(f, '__dict__'):
                            for f_attr, f_val in vars(f).items():
                                if not f_attr.startswith('_'):
                                    self.key_value(f"  {f_attr}", f_val)
                        else:
                            self._append(f"  {f}\n")
                elif isinstance(value, (dict, list)):
                    self.json_block(f"{attr} 详情", value)
                else:
                    self.key_value(attr, value)
        
        elif isinstance(param_table, dict):
            self.json_block("参数表内容 (dict)", param_table)
        
        else:
            self._append(f"\n```\n{param_table}\n```\n")
    
    def log_file_analysis_full(self, file_analysis: Any, title: str = "文件分析结果"):
        """完整记录文件分析结果"""
        if not file_analysis:
            self.warning(f"{title}: 无数据")
            return
        
        self.subsection(title)
        
        if hasattr(file_analysis, '__dict__'):
            self._append("\n**文件分析完整内容**:\n")
            for attr, value in vars(file_analysis).items():
                if attr.startswith('_'):
                    continue
                self._append(f"\n#### {attr}\n")
                if isinstance(value, (dict, list)):
                    self.json_block(f"{attr}", value)
                elif hasattr(value, '__dict__'):
                    self.json_block(f"{attr}", vars(value))
                else:
                    self.key_value(attr, value)
        elif isinstance(file_analysis, dict):
            self.json_block("文件分析结果 (dict)", file_analysis)
        else:
            self.code_block("文件分析结果", str(file_analysis))
    
    def log_todo_list_full(self, todo_list: Any, title: str = "Todo List 完整内容"):
        """完整记录 todo-list - 包含每个任务的所有参数"""
        if not todo_list:
            self.warning(f"{title}: 无任务")
            return
        
        self.subsection(title)
        
        # 记录 session 信息
        if hasattr(todo_list, 'session'):
            session = todo_list.session
            self._append("\n**Session 信息**:\n")
            if hasattr(session, '__dict__'):
                for attr, value in vars(session).items():
                    if not attr.startswith('_'):
                        self.key_value(attr, value)
        
        # 记录每个任务的完整信息
        if hasattr(todo_list, 'tasks'):
            self._append(f"\n**任务数量**: {len(todo_list.tasks)}\n")
            
            for i, task in enumerate(todo_list.tasks, 1):
                self._append(f"\n---\n\n#### 任务 {i}: `{getattr(task, 'id', '?')}`\n")
                
                # 基本信息
                task_type = getattr(task, 'type', 'unknown')
                if hasattr(task_type, 'value'):
                    task_type = task_type.value
                status = getattr(task, 'status', 'unknown')
                if hasattr(status, 'value'):
                    status = status.value
                
                self.key_value("类型", task_type)
                self.key_value("状态", status)
                self.key_value("优先级", getattr(task, 'priority', 'N/A'))
                
                # 完整描述
                description = getattr(task, 'description', '')
                self._append("\n**完整描述**:\n")
                self._append(f"{description}\n")
                
                # 完整参数
                parameters = getattr(task, 'parameters', {})
                if parameters:
                    self.json_block("任务参数", parameters)
                
                # 依赖关系
                dependencies = getattr(task, 'dependencies', [])
                if dependencies:
                    self._append(f"\n**依赖**: {dependencies}\n")
                
                # 执行结果（如果有）
                result = getattr(task, 'result', None)
                if result:
                    self.json_block("执行结果", result)
        
        self._append("\n---\n")
    
    def log_task_iteration(self, task_id: str, iteration: int, data: Dict[str, Any]):
        """
        记录任务的每次迭代 - 完整记录入参、代码、错误、结果
        
        Args:
            task_id: 任务 ID
            iteration: 迭代次数
            data: 包含 parameters, generated_code, execution_result, error_analysis 等
        """
        self._append(f"\n\n{'='*80}\n")
        self._append(f"## 任务 `{task_id}` - 迭代 {iteration}\n")
        self._append(f"{'='*80}\n\n")
        
        # 1. 入参
        parameters = data.get('parameters', {})
        if parameters:
            self.json_block("📥 输入参数", parameters)
        
        # 2. 生成的代码（完整）
        generated_code = data.get('generated_code', '')
        if generated_code:
            self.code_block("💻 生成的代码", generated_code, "python")
        
        # 3. 执行结果（完整）
        execution_result = data.get('execution_result')
        if execution_result:
            self._append("\n### 📤 执行结果\n")
            if isinstance(execution_result, dict):
                self.key_value("状态", execution_result.get('status', 'unknown'))
                
                output = execution_result.get('output', '')
                if output:
                    self.code_block("标准输出", str(output), "")
                
                error = execution_result.get('error', '')
                if error:
                    self.code_block("错误输出", str(error), "")
                
                returncode = execution_result.get('returncode')
                if returncode is not None:
                    self.key_value("返回码", returncode)
            elif hasattr(execution_result, '__dict__'):
                self.json_block("执行结果对象", vars(execution_result))
            else:
                self.code_block("执行结果", str(execution_result), "")
        
        # 4. 错误分析
        error_analysis = data.get('error_analysis')
        if error_analysis:
            self._append("\n### 🔍 错误分析\n")
            if isinstance(error_analysis, dict):
                self.key_value("错误类型", error_analysis.get('error_type', 'unknown'))
                self.key_value("根本原因", error_analysis.get('root_cause', ''))
                self.key_value("修复建议", error_analysis.get('fix_suggestion', ''))
            else:
                self._append(str(error_analysis))
        
        # 5. Revision 信息
        revision_plan = data.get('revision_plan')
        if revision_plan:
            self._append("\n### 🔄 Revision 计划\n")
            if isinstance(revision_plan, dict):
                self.key_value("策略", revision_plan.get('strategy', ''))
                self.key_value("原因", revision_plan.get('root_cause', ''))
                if revision_plan.get('plan'):
                    self._append(f"\n**修复计划**:\n{revision_plan.get('plan')}\n")
            else:
                self._append(str(revision_plan))
        
        self._append("\n")
    
    def log_supervisor_full(self, state_before: Any, state_after: Any):
        """完整记录 Supervisor 节点的输入输出"""
        self.section("1. Supervisor 详细记录")
        
        # 输入状态
        self.subsection("1.1 输入状态")
        self.key_value("session_id", getattr(state_before, 'session_id', 'N/A'))
        self.key_value("user_task_type", str(getattr(state_before, 'user_task_type', 'N/A')))
        self.key_value("user_input", getattr(state_before, 'user_input', 'N/A'))
        
        # 输出状态
        self.subsection("1.2 输出状态")
        self.key_value("session_id", getattr(state_after, 'session_id', 'N/A'))
        self.key_value("sandbox_dir", getattr(state_after, 'sandbox_dir', 'N/A'))
        self.key_value("opensandbox_id", getattr(state_after, 'opensandbox_id', 'N/A'))
        
        # 参数表
        param_table = getattr(state_after, 'parameter_table', None)
        if param_table:
            self.log_parameter_table_full(param_table, "1.3 提取的参数表")
        
        # 文件分析结果
        file_analysis = getattr(state_after, 'file_analysis_result', None)
        if file_analysis:
            self.log_file_analysis_full(file_analysis, "1.4 文件分析结果")
        
        # merged_result
        merged_result = getattr(state_after, 'merged_result', None)
        if merged_result:
            self.json_block("1.5 Merged Result", merged_result)
    
    def log_codeact_execution(self, task_desc: str, code: str, result: Any, error: str = None):
        """记录 CodeAct 执行 - 完整记录，不截断"""
        self._append(f"\n### CodeAct 执行\n\n")
        
        # 任务描述（完整）
        self._append(f"**任务描述**:\n{task_desc}\n\n")
        
        # 执行的代码（完整）
        if code:
            self._append("**执行代码**:\n")
            self._append(f"```python\n{code}\n```\n")
        
        # 执行结果（完整）
        if result:
            self._append("\n**执行结果**:\n")
            if hasattr(result, 'is_success'):
                status = "✅ 成功" if result.is_success() else "❌ 失败"
                self._append(f"- 状态: {status}\n")
                if hasattr(result, 'output') and result.output:
                    self._append(f"- 输出:\n```\n{result.output}\n```\n")
                if hasattr(result, 'error') and result.error:
                    self._append(f"- 错误:\n```\n{result.error}\n```\n")
                if hasattr(result, 'returncode') and result.returncode is not None:
                    self._append(f"- 返回码: `{result.returncode}`\n")
            elif isinstance(result, dict):
                # 完整记录所有字段
                for k, v in result.items():
                    if isinstance(v, str) and len(v) > 200:
                        self._append(f"- **{k}**:\n```\n{v}\n```\n")
                    else:
                        self.key_value(k, v)
        
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
                if err['details']:
                    self._append(f"   <details><summary>详情</summary>\n\n   ```\n   {err['details'][:500]}...\n   ```\n   </details>\n")
        else:
            self._append("✅ **测试通过，无错误**\n")
        
        self._append(f"\n---\n*报告生成时间: {datetime.now().isoformat()}*\n")
        
        print(f"\n📄 详细测试报告已保存: {self.report_file}")


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
# Q13 预定义 Todo List (从 report_20260304_154127 提取)
# =============================================================================

Q13_TODO_TASKS = [
    {
        "id": "task_001",
        "type": "mcp_tool",
        "status": "pending",
        "priority": 1,
        "description": "[check_peptide_support] Task name: Check MART-1 Peptide Support\n"
                       "Task description: Check if MART-1 peptide (ELAGIGILTV) has a pre-trained NetTCR-2.2 model or will use pan-specific prediction\n"
                       "Tools: check_peptide_support",
        "parameters": {
            "peptides": "ELAGIGILTV",
            "output_file": "/data/sessions/{session_id}/output/peptide_support_status.json",
            "tool_name": "check_peptide_support"
        }
    },
    {
        "id": "task_002",
        "type": "mcp_tool",
        "status": "pending",
        "priority": 1,
        "description": "[validate_tcr_input] Task name: Validate TCR Input Data\n"
                       "Task description: Validate TCR input data format for NetTCR-2.2 prediction. Checks sequence lengths, amino acid composition, and V gene resolution\n"
                       "Tools: validate_tcr_input",
        "parameters": {
            "test_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv",
            "output_file": "/data/sessions/{session_id}/output/validation_report.json",
            "tool_name": "validate_tcr_input"
        }
    },
    {
        "id": "task_003",
        "type": "mcp_tool",
        "status": "pending",
        "priority": 2,
        "description": "[predict_tcr_binding_complete] Task name: Predict TCR-MART-1 Binding\n"
                       "Task description: Run complete NetTCR-2.2 prediction pipeline for TCR-MART-1 binding. Includes input validation, CDR1/CDR2 resolution, prediction, and result aggregation\n"
                       "Tools: predict_tcr_binding_complete",
        "parameters": {
            "test_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv",
            "output_dir": "/data/sessions/{session_id}/output",
            "peptides": "ELAGIGILTV",
            "rank_threshold": 0.5,
            "tool_name": "predict_tcr_binding_complete"
        }
    },
    {
        "id": "task_004",
        "type": "general",
        "status": "pending",
        "priority": 3,
        "description": "[codeact] Task name: Integrate TCR Prediction Results\n"
                       "Task description: Read prediction results and generate final output CSV with columns: main_name, prediction, probability\n"
                       "This is a codeact task that requires Python code execution.",
        "parameters": {
            "input_file": "/data/sessions/{session_id}/output/predictions.csv",
            "output_file": "/data/sessions/{session_id}/output/final_predictions.csv"
        }
    },
    {
        "id": "task_005",
        "type": "general",
        "status": "pending",
        "priority": 4,
        "description": "[codeact] Task name: Evaluate TCR Binding Predictions\n"
                       "Task description: Calculate evaluation metrics (F1 score, precision, recall) for TCR binding predictions against ground truth\n"
                       "This is a codeact task that requires Python code execution.",
        "parameters": {
            "predictions_file": "/data/sessions/{session_id}/output/final_predictions.csv",
            "ground_truth_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv",
            "output_file": "/data/sessions/{session_id}/output/evaluation_metrics.json"
        }
    }
]


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
    """测试 supervisor 节点 - 完整记录参数表和文件分析结果"""
    print_section("Step 1: Supervisor 节点")
    logger = get_test_logger()
    
    from main_graph import supervisor_node
    
    # 记录输入状态
    logger.section("1. Supervisor 节点")
    logger.subsection("1.1 输入状态")
    logger.key_value("session_id", state.session_id)
    logger.key_value("user_task_type", str(state.user_task_type))
    logger.key_value("user_input", state.user_input)
    
    # 记录输入文件路径
    if hasattr(state, 'file_paths') and state.file_paths:
        logger.json_block("1.2 输入文件路径", state.file_paths)
    
    try:
        result = supervisor_node(state)
        
        # 记录输出状态基本信息
        logger.subsection("1.3 输出状态")
        logger.key_value("session_id", result.session_id)
        logger.key_value("sandbox_dir", result.sandbox_dir)
        logger.key_value("sandbox_data_dir", getattr(result, 'sandbox_data_dir', 'N/A'))
        logger.key_value("opensandbox_id", result.opensandbox_id or "N/A")
        
        # ========== 完整记录参数表 ==========
        param_table = getattr(result, 'parameter_table', None)
        if param_table:
            logger.log_parameter_table_full(param_table, "1.4 提取的参数表")
        else:
            logger.warning("1.4 参数表: 无数据")
        
        # ========== 完整记录文件分析结果 ==========
        file_analysis = getattr(result, 'file_analysis_result', None)
        if file_analysis:
            logger.log_file_analysis_full(file_analysis, "1.5 文件分析结果")
        else:
            logger.warning("1.5 文件分析结果: 无数据")
        
        # 记录 merged_result
        merged_result = getattr(result, 'merged_result', None)
        if merged_result:
            logger.json_block("1.6 Merged Result", merged_result)
        
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
    """测试 todo-list.md 生成 - 完整记录 todo-list 内容"""
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
        
        # 记录 subtasks 信息
        if state.subtasks:
            logger._append(f"\n**子任务数量**: {len(state.subtasks)}\n")
            subtasks_data = []
            for task in state.subtasks:
                task_info = {
                    "task_id": getattr(task, 'task_id', 'unknown'),
                    "content": getattr(task, 'content', '')[:200],
                    "task_type": str(getattr(task, 'task_type', 'unknown')),
                    "dependencies": getattr(task, 'dependencies', []),
                    "parallel_group_id": getattr(task, 'parallel_group_id', None),
                    "parameters": getattr(task, 'parameters', {})
                }
                subtasks_data.append(task_info)
            logger.json_block("Subtasks 详情", subtasks_data)
        
        todo_list = generate_and_save_todolist_from_state(
            global_state=state,
            opensandbox_id=opensandbox_id
        )
        
        if todo_list:
            # 完整记录 todo-list 内容
            logger.log_todo_list_full(todo_list, "生成的 Todo List")
            logger.success(f"Todo list 生成成功: {len(todo_list.tasks)} 个任务")
            return todo_list, True
        else:
            logger.error("Todo list 生成失败")
            return None, False
            
    except Exception as e:
        logger.error(f"Todo list 生成失败: {e}", traceback.format_exc())
        return None, False


def test_codeact_todo_node(state):
    """
    测试 executor 节点 (CodeAct Todo 模式)
    
    完整记录：
    - 每个任务的每次迭代
    - 入参、生成的代码、错误分析、运行结果
    - Revision 迭代过程
    """
    print_section("Step 3: Executor 节点 (CodeAct Todo 模式)")
    logger = get_test_logger()
    
    from main_graph import _codeact_input_mapper, _codeact_output_mapper, _get_codeact_subgraph
    
    logger.section("3. Executor (CodeAct Todo) 详细记录")
    
    # 输入状态
    logger.subsection("3.1 输入状态")
    logger.key_value("session_id", state.session_id)
    logger.key_value("sandbox_dir", state.sandbox_dir)
    logger.key_value("subtasks 数量", len(state.subtasks) if state.subtasks else 0)
    logger.key_value("opensandbox_id", state.opensandbox_id or "N/A")
    
    # 记录 todo_list
    if hasattr(state, 'todo_list') and state.todo_list:
        logger.log_todo_list_full(state.todo_list, "3.2 输入 Todo List")
    
    try:
        # 映射到 CodeAct 状态
        logger.subsection("3.3 映射到 CodeAct 状态")
        codeact_state = _codeact_input_mapper(state)
        logger.key_value("todo_list_path", codeact_state.todo_list_path or "N/A")
        logger.key_value("execution_mode", str(codeact_state.execution_mode))
        
        # 获取子图
        subgraph = _get_codeact_subgraph()
        if subgraph is None:
            logger.error("CodeAct 子图不可用")
            return state, False
        
        # ========== 执行 CodeAct 子图，完整记录每次迭代 ==========
        logger.section("3.4 CodeAct 执行过程（完整记录）")
        
        # 追踪任务迭代
        task_iterations = {}  # task_id -> list of iterations
        current_task_id = None
        iteration_count = 0
        
        print("\n执行 CodeAct 子图...")
        
        result_state = None
        step_count = 0
        
        for event in subgraph.stream(codeact_state):
            step_count += 1
            node_name = list(event.keys())[0] if event else "unknown"
            node_output = event.get(node_name, {}) if event else {}
            
            print(f"\n  [Step {step_count}] 节点: {node_name}")
            
            if not node_output:
                continue
            
            # Helper 函数
            def get_value(obj, key, default=None):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)
            
            # 获取当前任务信息
            current_task = get_value(node_output, 'current_todo_task')
            if current_task:
                current_task_id = get_value(current_task, 'id', 'unknown')
                
                # 初始化任务迭代记录
                if current_task_id not in task_iterations:
                    task_iterations[current_task_id] = []
            
            # ========== 节点级别的详细记录 ==========
            
            # select_next_task 节点 - 记录任务选择
            if node_name == 'select_next_task' and current_task:
                logger._append(f"\n\n### Step {step_count}: 选择任务 `{current_task_id}`\n")
                logger.key_value("任务描述", get_value(current_task, 'description', ''))
                logger.key_value("任务类型", get_value(current_task, 'type', ''))
                logger.key_value("任务优先级", get_value(current_task, 'priority', ''))
                
                params = get_value(current_task, 'parameters', {})
                if params:
                    logger.json_block("任务参数", params)
            
            # infer_parameters 节点 - 记录参数推断
            elif node_name == 'infer_parameters':
                inferred_params = get_value(node_output, 'inferred_parameters', {})
                file_param_table = get_value(node_output, 'file_parameter_table')
                
                logger._append(f"\n\n### Step {step_count}: 参数推断\n")
                
                # 记录文件参数表
                if file_param_table and hasattr(file_param_table, 'files'):
                    files = file_param_table.files
                    if files:
                        logger.key_value("📁 File Parameter Table", f"{len(files)} files")
                        for key, fp in files.items():
                            logger.key_value(f"  - {key}", getattr(fp, 'path', 'N/A'))
                
                if inferred_params:
                    logger.json_block("推断的参数", inferred_params)
                else:
                    logger.key_value("推断结果", "无新参数推断（使用任务参数）")
            
            # generate_code 节点 - 记录生成的代码
            elif node_name == 'generate_code':
                generated_code = get_value(node_output, 'generated_code', '')
                if generated_code:
                    logger._append(f"\n\n### Step {step_count}: 代码生成\n")
                    logger.code_block(f"生成的代码 (任务: {current_task_id})", generated_code, "python")
                    
                    # 记录迭代信息
                    if current_task_id:
                        iteration_count = len(task_iterations.get(current_task_id, [])) + 1
                        iteration_data = {
                            "step": step_count,
                            "generated_code": generated_code
                        }
                        task_iterations.setdefault(current_task_id, []).append(iteration_data)
            
            # execute_code 节点 - 记录执行结果
            elif node_name == 'execute_code':
                execution_result = get_value(node_output, 'execution_result')
                if execution_result:
                    logger._append(f"\n\n### Step {step_count}: 代码执行\n")
                    
                    if isinstance(execution_result, dict):
                        status = execution_result.get('status', 'unknown')
                        logger.key_value("执行状态", status)
                        
                        output = execution_result.get('output', '')
                        if output:
                            logger.code_block("标准输出", str(output), "")
                        
                        error = execution_result.get('error', '')
                        if error:
                            logger.code_block("错误输出", str(error), "")
                        
                        returncode = execution_result.get('returncode')
                        if returncode is not None:
                            logger.key_value("返回码", returncode)
                    else:
                        logger.code_block("执行结果", str(execution_result), "")
                    
                    # 更新迭代记录
                    if current_task_id and task_iterations.get(current_task_id):
                        task_iterations[current_task_id][-1]['execution_result'] = execution_result
            
            # update_todo 节点 - 记录任务状态更新
            elif node_name == 'update_todo':
                todo_list = get_value(node_output, 'todo_list')
                if todo_list:
                    tasks = get_value(todo_list, 'tasks', [])
                    if tasks:
                        logger._append(f"\n\n### Step {step_count}: 更新 Todo 状态\n")
                        
                        completed = []
                        failed = []
                        pending = []
                        
                        for t in tasks:
                            t_id = get_value(t, 'id', '?')
                            t_status = str(get_value(t, 'status', 'unknown'))
                            
                            if 'COMPLETED' in t_status:
                                completed.append(t_id)
                            elif 'FAILED' in t_status:
                                failed.append(t_id)
                            else:
                                pending.append(t_id)
                        
                        logger.key_value("已完成", completed)
                        logger.key_value("失败", failed)
                        logger.key_value("待执行", pending)
            
            # analyze_error 节点 - 记录错误分析
            elif node_name == 'analyze_error':
                error_analysis = get_value(node_output, 'error_analysis')
                if error_analysis:
                    logger._append(f"\n\n### Step {step_count}: 错误分析\n")
                    
                    if isinstance(error_analysis, dict):
                        logger.key_value("错误类型", error_analysis.get('error_type', 'unknown'))
                        logger.key_value("根本原因", error_analysis.get('root_cause', ''))
                        logger.key_value("修复建议", error_analysis.get('fix_suggestion', ''))
                    else:
                        logger.code_block("错误分析结果", str(error_analysis), "")
                    
                    # 更新迭代记录
                    if current_task_id and task_iterations.get(current_task_id):
                        task_iterations[current_task_id][-1]['error_analysis'] = error_analysis
            
            # plan_revision 节点 - 记录 Revision 计划
            elif node_name == 'plan_revision':
                revision_plan = get_value(node_output, 'revision_plan')
                revision_iter = get_value(node_output, 'revision_iteration', 0)
                
                if revision_plan:
                    logger._append(f"\n\n### Step {step_count}: Revision 计划 (迭代 {revision_iter})\n")
                    
                    if isinstance(revision_plan, dict):
                        logger.key_value("策略", revision_plan.get('strategy', ''))
                        logger.key_value("根本原因", revision_plan.get('root_cause', ''))
                        if revision_plan.get('plan'):
                            logger._append(f"\n**修复计划**:\n{revision_plan.get('plan')}\n")
                    else:
                        logger.code_block("Revision 计划", str(revision_plan), "")
                    
                    # 更新迭代记录
                    if current_task_id and task_iterations.get(current_task_id):
                        task_iterations[current_task_id][-1]['revision_plan'] = revision_plan
                        task_iterations[current_task_id][-1]['revision_iteration'] = revision_iter
            
            result_state = node_output
        
        print(f"\n[子图执行完成，共 {step_count} 步]")
        
        # ========== 汇总所有任务的迭代记录 ==========
        logger.section("3.5 任务迭代汇总")
        
        for task_id, iterations in task_iterations.items():
            logger._append(f"\n### 任务 `{task_id}` - 共 {len(iterations)} 次迭代\n")
            
            for i, iter_data in enumerate(iterations, 1):
                logger.log_task_iteration(
                    task_id=task_id,
                    iteration=i,
                    data=iter_data
                )
        
        # ========== 映射结果回全局状态 ==========
        logger.section("3.6 执行结果汇总")
        
        if result_state:
            state = _codeact_output_mapper(result_state, state)
            
            # 记录 merged_result
            if state.merged_result:
                logger.json_block("Merged Result", state.merged_result)
            
            # 记录 executor_results
            if state.merged_result and "executor_results" in state.merged_result:
                exec_results = state.merged_result["executor_results"]
                logger._append("\n**Executor 结果**:\n")
                logger.key_value("总任务数", exec_results.get('total_tasks', 0))
                logger.key_value("已完成", exec_results.get('completed_count', 0))
                logger.key_value("失败", exec_results.get('failed_count', 0))
                logger.key_value("todo_list_path", exec_results.get('todo_list_path', 'N/A'))
            
            # 记录完成的任务
            if hasattr(state, 'completed_tasks') and state.completed_tasks:
                logger._append("\n**已完成的任务**:\n")
                for task in state.completed_tasks:
                    task_id = getattr(task, 'task_id', 'unknown')
                    result = getattr(task, 'result', None)
                    logger._append(f"\n#### 任务 {task_id}\n")
                    if result:
                        logger.json_block("结果", result)
            
            logger.success("Executor 节点完成")
            return state, True
        else:
            logger.error("CodeAct 子图返回空结果")
            return state, False
        
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


def test_executor_standalone_simple():
    """
    单独测试 executor 节点 - 使用简单的模拟数据（原始版本）
    
    使用单一简单任务测试基本流程
    """
    global test_logger
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{timestamp}_executor_simple"
    
    # 初始化日志
    test_logger = TestLogger(session_id=session_id)
    test_logger.section("Executor 节点独立测试 (简单模式)")
    test_logger.key_value("模式", "使用简单模拟数据")
    test_logger.divider()
    
    # 导入必要模块
    print("导入模块...")
    from main_graph import _codeact_input_mapper, _codeact_output_mapper, _get_codeact_subgraph
    from state import GlobalState, UserTaskType, SubTask, ensure_global_state_rebuilt
    from nodes.subagents.code_act import TodoList, TodoTask, TodoTaskStatus, TodoListSession
    from nodes.subagents.executor.todolist_generator import save_todo_list_to_sandbox
    
    # 确保GlobalState模型已重建以支持todo_list字段
    ensure_global_state_rebuilt()
    
    # 构造简单的 GlobalState
    print("\n构造简单 GlobalState...")
    
    # 获取 sandbox
    opensandbox_id = os.environ.get("TEST_OPENSANDBOX_ID")
    sandbox_dir = f"/data/sessions/{session_id}"
    
    # 创建简单的 todo-list (1 个任务)
    simple_task = TodoTask(
        id="simple_task_001",
        type="general",
        status=TodoTaskStatus.PENDING,
        priority=1,
        description="读取 /data 目录下的文件列表，输出前5个文件名",
        parameters={}
    )
    
    todo_list = TodoList(
        session=TodoListSession(
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            sandbox_dir=sandbox_dir
        ),
        tasks=[simple_task]
    )
    
    # 创建模拟 subtask
    mock_subtasks = [
        SubTask(
            task_id="simple_task_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="读取 /data 目录下的文件列表，输出前5个文件名",
            dependencies=[],
            parallel_group_id=None,
            parameters={"tool_name": "codeact"}
        )
    ]
    
    state = GlobalState(
        user_input="测试 executor 节点（简单模式）",
        user_task_type=UserTaskType.EXECUTE_PLAN,
        session_id=session_id,
        sandbox_dir=sandbox_dir,
        sandbox_data_dir=sandbox_dir,
        subtasks=mock_subtasks,
        opensandbox_id=opensandbox_id,
        merged_result={},
        todo_list=todo_list
    )
    
    test_logger.key_values({
        "session_id": session_id,
        "sandbox_dir": state.sandbox_dir,
        "todo 任务数量": 1,
        "opensandbox_id": opensandbox_id or "无（将创建新 sandbox）"
    })
    test_logger.divider()
    
    # ========== Step 1: 保存 todo-list.md ==========
    test_logger.section("Step 1: 保存 todo-list.md")
    
    try:
        save_result = save_todo_list_to_sandbox(
            todo_list=todo_list,
            opensandbox_id=opensandbox_id,
            sandbox_dir=sandbox_dir
        )
        
        if save_result:
            test_logger.success(f"todo-list.md 保存成功: 1 个任务")
        
        test_logger.log_todo_list(todo_list, "Todo List 内容")
        
    except Exception as e:
        test_logger.error(f"保存 todo-list.md 失败: {e}", traceback.format_exc())
    
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
        print("\n[开始执行子图...]")
        
        result_state = None
        step_count = 0
        
        for event in subgraph.stream(codeact_state):
            step_count += 1
            node_name = list(event.keys())[0] if event else "unknown"
            node_output = event.get(node_name, {}) if event else {}
            
            print(f"\n  [Step {step_count}] 节点: {node_name}")
            
            if node_output:
                def get_value(obj, key, default=None):
                    if isinstance(obj, dict):
                        return obj.get(key, default)
                    return getattr(obj, key, default)
                
                generated_code = get_value(node_output, 'generated_code')
                if generated_code:
                    test_logger.code_block(
                        f"Step {step_count} - {node_name}: 生成的代码",
                        generated_code[:1500] if len(generated_code) > 1500 else generated_code,
                        "python"
                    )
                
                execution_result = get_value(node_output, 'execution_result')
                if execution_result:
                    if isinstance(execution_result, dict):
                        test_logger.key_values({
                            f"Step {step_count} 执行状态": execution_result.get('status', 'unknown'),
                            f"Step {step_count} 输出": str(execution_result.get('output', ''))[:200]
                        })
                        if execution_result.get('error'):
                            test_logger.error(f"Step {step_count} 执行错误: {execution_result.get('error')}")
                
                current_todo_task = get_value(node_output, 'current_todo_task')
                if current_todo_task:
                    task_id = get_value(current_todo_task, 'id', 'unknown')
                    task_desc = get_value(current_todo_task, 'description', '')
                    test_logger.key_values({
                        f"Step {step_count} 当前任务 ID": task_id,
                        f"Step {step_count} 任务描述": task_desc[:100] if task_desc else "N/A"
                    })
                
                todo_list = get_value(node_output, 'todo_list')
                if todo_list:
                    tasks = get_value(todo_list, 'tasks', [])
                    if tasks:
                        completed = sum(1 for t in tasks if str(get_value(t, 'status', '')).endswith('COMPLETED'))
                        total = len(tasks)
                        test_logger.key_value(f"Step {step_count} 任务进度", f"{completed}/{total}")
            
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
            
            if state.merged_result and "executor_results" in state.merged_result:
                exec_results = state.merged_result["executor_results"]
                test_logger.key_values({
                    "总任务数": exec_results.get('total_tasks', 0),
                    "已完成": exec_results.get('completed_count', 0),
                    "失败": exec_results.get('failed_count', 0),
                    "todo_list_path": exec_results.get('todo_list_path', 'N/A')
                })
            
            test_logger.success("Executor 节点执行完成")
            
        except Exception as e:
            test_logger.error(f"结果映射失败: {e}", traceback.format_exc())
    else:
        test_logger.error("CodeAct 子图返回空结果")
    
    test_logger.finalize()
    return True


def test_executor_standalone():
    """
    单独测试 executor 节点 - 使用 Q13 预定义的复杂 todo-list
    
    这个测试使用从 report_20260304_154127 提取的 5 个任务：
    1. task_001 - mcp_tool: check_peptide_support
    2. task_002 - mcp_tool: validate_tcr_input
    3. task_003 - mcp_tool: predict_tcr_binding_complete
    4. task_004 - general (codeact): Integrate TCR Prediction Results
    5. task_005 - general (codeact): Evaluate TCR Binding Predictions
    
    记录内容：
    1. todo-list.md 内容
    2. 每个任务的描述和参数
    3. 生成的代码
    4. 执行结果或报错
    5. Revision 迭代过程
    """
    global test_logger
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{timestamp}_executor_q13"
    
    # 初始化日志
    test_logger = TestLogger(session_id=session_id)
    test_logger.section("Executor 节点独立测试 (Q13 Todo-List)")
    test_logger.key_value("模式", "使用 Q13 预定义的复杂 todo-list（5个任务）")
    test_logger.key_value("来源", "report_20260304_154127_codeact_todo_test")
    test_logger.divider()
    
    # 导入必要模块
    print("导入模块...")
    from main_graph import executor_node, _codeact_input_mapper, _codeact_output_mapper, _get_codeact_subgraph
    from state import GlobalState, UserTaskType, SubTask, ensure_global_state_rebuilt
    from nodes.subagents.code_act import TodoList, TodoTask, TodoTaskStatus, TodoListSession
    from nodes.subagents.executor.todolist_generator import save_todo_list_to_sandbox
    
    # 确保GlobalState模型已重建以支持todo_list字段
    ensure_global_state_rebuilt()
    
    # 构造 GlobalState
    print("\n构造 GlobalState (使用 Q13 预定义 todo-list)...")
    
    # 获取 sandbox
    opensandbox_id = os.environ.get("TEST_OPENSANDBOX_ID")
    sandbox_dir = f"/data/sessions/{session_id}"
    
    # 构造 Q13 预定义的 todo-list
    print(f"\n构造 Q13 todo-list ({len(Q13_TODO_TASKS)} 个任务)...")
    
    todo_tasks = []
    for task_def in Q13_TODO_TASKS:
        # 替换参数中的 session_id 占位符
        parameters = {}
        for k, v in task_def["parameters"].items():
            if isinstance(v, str):
                parameters[k] = v.replace("{session_id}", session_id)
            else:
                parameters[k] = v
        
        todo_task = TodoTask(
            id=task_def["id"],
            type=task_def["type"],
            status=TodoTaskStatus.PENDING,
            priority=task_def["priority"],
            description=task_def["description"],
            parameters=parameters
        )
        todo_tasks.append(todo_task)
    
    # 创建 TodoList
    todo_list = TodoList(
        session=TodoListSession(
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            sandbox_dir=sandbox_dir
        ),
        tasks=todo_tasks
    )
    
    # 创建模拟 subtasks (用于 GlobalState)
    mock_subtasks = [
        SubTask(
            task_id="task_q13_main",
            task_type=UserTaskType.EXECUTE_PLAN,
            content=Q13_PROMPT[:200],
            dependencies=[],
            parallel_group_id=None,
            parameters={"tool_name": "codeact"}
        )
    ]
    
    state = GlobalState(
        user_input=Q13_PROMPT,
        user_task_type=UserTaskType.EXECUTE_PLAN,
        session_id=session_id,
        sandbox_dir=sandbox_dir,
        sandbox_data_dir=sandbox_dir,
        subtasks=mock_subtasks,
        opensandbox_id=opensandbox_id,
        merged_result={},
        file_paths=Q13_FILE_PATHS.copy(),
        todo_list=todo_list
    )
    
    test_logger.key_values({
        "session_id": session_id,
        "sandbox_dir": state.sandbox_dir,
        "todo 任务数量": len(todo_tasks),
        "opensandbox_id": opensandbox_id or "无（将创建新 sandbox）"
    })
    test_logger.divider()
    
    # ========== Step 1: 保存 todo-list.md ==========
    test_logger.section("Step 1: 保存预定义的 todo-list.md")
    
    try:
        # 保存 todo-list.md 到沙盒
        save_result = save_todo_list_to_sandbox(
            todo_list=todo_list,
            opensandbox_id=opensandbox_id,
            sandbox_dir=sandbox_dir
        )
        
        if save_result:
            test_logger.success(f"todo-list.md 保存成功: {len(todo_list.tasks)} 个任务")
        else:
            test_logger.warning("todo-list.md 保存返回空结果，继续使用内存中的 todo_list")
        
        # 记录完整的 todo-list 内容
        test_logger.log_todo_list(todo_list, "Q13 Todo List 内容")
        
        # 记录每个任务的详细参数
        test_logger.section("任务参数详情")
        for task in todo_list.tasks:
            params_str = ", ".join([f"{k}={v}" for k, v in task.parameters.items()])
            test_logger.key_value(f"{task.id} 参数", params_str[:100] + "..." if len(params_str) > 100 else params_str)
            
    except Exception as e:
        test_logger.error(f"保存 todo-list.md 失败: {e}", traceback.format_exc())
        test_logger.warning("继续使用内存中的 todo_list")
    
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
    parser.add_argument("--executor", action="store_true", help="单独测试 executor 节点（使用 Q13 预定义 todo-list）")
    parser.add_argument("--executor-simple", action="store_true", help="单独测试 executor 节点（使用简单模拟数据）")
    args = parser.parse_args()
    
    if args.executor:
        test_executor_standalone()
    elif args.executor_simple:
        test_executor_standalone_simple()
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

