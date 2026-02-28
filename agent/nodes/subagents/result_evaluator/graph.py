"""
Result Evaluator Subgraph

用于总结执行结果并生成最终报告的子图

工作流：
1. 结果收集：收集所有任务的执行结果和输出文件
2. 结果分析：分析成功/失败情况，提取关键发现
3. 报告生成：使用 LLM 生成最终总结报告
"""

import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from langgraph.graph import StateGraph, START, END

# 添加 agent 目录到路径
import sys
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import GlobalState
from utils.llm_factory import create_bioinformatics_llm, create_reasoning_llm
from .state import ResultEvaluatorState, TaskResultSummary


# ===================== 辅助函数 =====================

def _save_report(content: str, report_type: str, sandbox_dir: str) -> str:
    """
    保存报告到文件

    Args:
        content: 报告内容
        report_type: 报告类型
        sandbox_dir: 沙盒目录

    Returns:
        保存的文件路径
    """
    try:
        reports_dir = Path(sandbox_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"{report_type}_{timestamp}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"📄 {report_type} 报告已保存到: {report_file}")
        return str(report_file)
    except Exception as e:
        print(f"⚠️ 保存报告失败: {e}")
        return ""


def _extract_output_files(task_result: Any) -> List[str]:
    """从任务结果中提取输出文件路径"""
    output_files = []

    if task_result is None:
        return output_files

    # 尝试从 output 中提取文件路径
    output = None
    if hasattr(task_result, 'output'):
        output = task_result.output
    elif isinstance(task_result, dict):
        output = task_result.get('output')

    if output is None:
        return output_files

    # 如果 output 是字符串，尝试解析 JSON
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            # 如果不是 JSON，检查是否是文件路径
            if '/' in output or '\\' in output:
                output_files.append(output)
            return output_files

    if not isinstance(output, dict):
        return output_files

    # 常见的文件路径字段名
    file_fields = [
        'output_file', 'output_path', 'file_path', 'result_file',
        'output_files', 'files', 'result_files', 'saved_files',
        'csv_file', 'json_file', 'report_file', 'log_file'
    ]

    for field in file_fields:
        if field in output:
            value = output[field]
            if isinstance(value, str):
                output_files.append(value)
            elif isinstance(value, list):
                output_files.extend([f for f in value if isinstance(f, str)])

    # 检查 final_result 中的文件
    if 'final_result' in output:
        final_result = output['final_result']
        if isinstance(final_result, dict):
            for field in file_fields:
                if field in final_result:
                    value = final_result[field]
                    if isinstance(value, str):
                        output_files.append(value)
                    elif isinstance(value, list):
                        output_files.extend([f for f in value if isinstance(f, str)])

    return output_files


def _truncate_text(text: str, max_length: int = 500) -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


# ===================== 节点 1: 结果收集 =====================

def collect_results_node(state: ResultEvaluatorState) -> ResultEvaluatorState:
    """
    节点 1: 结果收集

    收集所有任务的执行结果和输出文件
    """
    print("\n" + "=" * 60)
    print("📊 阶段 1: 收集执行结果")
    print("=" * 60)

    # 统计任务状态
    total = len(state.all_tasks)
    completed = sum(1 for t in state.all_tasks if t.status.upper() == "COMPLETED")
    failed = sum(1 for t in state.all_tasks if t.status.upper() == "FAILED")

    state.total_tasks = total
    state.completed_tasks = completed
    state.failed_tasks = failed
    state.success_rate = (completed / total * 100) if total > 0 else 0.0

    # 收集所有输出文件
    all_output_files = []
    for task in state.all_tasks:
        task_files = _extract_output_files(task)
        all_output_files.extend(task_files)

    # 去重
    state.output_files = list(set(all_output_files))

    # 生成错误摘要
    errors = []
    for task in state.all_tasks:
        if task.error:
            errors.append(f"- 任务 {task.task_id}: {_truncate_text(task.error, 200)}")

    if errors:
        state.error_summary = "\n".join(errors[:10])  # 最多显示10个错误
        if len(errors) > 10:
            state.error_summary += f"\n... 还有 {len(errors) - 10} 个错误"
    else:
        state.error_summary = "无错误"

    print(f"✅ 结果收集完成")
    print(f"  - 总任务数: {total}")
    print(f"  - 完成数: {completed}")
    print(f"  - 失败数: {failed}")
    print(f"  - 成功率: {state.success_rate:.1f}%")
    print(f"  - 输出文件数: {len(state.output_files)}")

    return state


# ===================== 节点 2: 结果分析 =====================

def analyze_results_node(state: ResultEvaluatorState) -> ResultEvaluatorState:
    """
    节点 2: 结果分析

    使用 LLM 分析执行结果
    """
    print("\n" + "=" * 60)
    print("🔍 阶段 2: 分析执行结果")
    print("=" * 60)

    llm = create_bioinformatics_llm()
    if not llm:
        print("⚠️ LLM 不可用，使用简单分析")
        state.key_findings = ["LLM 不可用，无法进行深度分析"]
        state.recommendations = ["请检查 LLM 配置"]
        return state

    try:
        from langchain_core.messages import HumanMessage

        # 构建分析提示词
        task_summaries = []
        for task in state.all_tasks[:20]:  # 最多显示20个任务
            task_summary = f"""
任务ID: {task.task_id}
类型: {task.task_type}
状态: {task.status}
内容: {_truncate_text(task.content, 200)}
"""
            if task.error:
                task_summary += f"错误: {_truncate_text(task.error, 200)}\n"
            if task.output:
                output_str = str(task.output)
                task_summary += f"输出: {_truncate_text(output_str, 300)}\n"

            task_summaries.append(task_summary)

        tasks_text = "\n---\n".join(task_summaries)

        # 输出文件信息
        files_text = ""
        if state.output_files:
            files_text = "输出文件:\n" + "\n".join([f"- {f}" for f in state.output_files[:20]])

        analysis_prompt = f"""
你是一个专业的数据分析助手。请分析以下任务执行结果并生成总结。

## 用户原始输入
{state.user_input}

## 执行计划
{state.execution_plan[:2000] if state.execution_plan else "无执行计划"}

## 执行统计
- 总任务数: {state.total_tasks}
- 完成数: {state.completed_tasks}
- 失败数: {state.failed_tasks}
- 成功率: {state.success_rate:.1f}%

## 任务执行详情
{tasks_text}

## {files_text}

## 错误摘要
{state.error_summary}

请提供：
1. 关键发现 (key_findings): 列出3-5个关键发现，每个发现一句话
2. 建议 (recommendations): 列出2-3条改进建议

以 JSON 格式返回：
{{
    "key_findings": ["发现1", "发现2", "发现3"],
    "recommendations": ["建议1", "建议2"]
}}
"""

        messages = [HumanMessage(content=analysis_prompt)]

        response = llm.invoke(messages)
        response_content = response.content if hasattr(response, 'content') else str(response)

        # 尝试解析 JSON
        try:
            # 尝试提取 JSON 块
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                analysis_result = json.loads(json_match.group())
            else:
                analysis_result = json.loads(response_content)

            state.key_findings = analysis_result.get('key_findings', [])
            state.recommendations = analysis_result.get('recommendations', [])

        except (json.JSONDecodeError, TypeError) as e:
            print(f"⚠️ JSON 解析失败: {e}")
            # 使用默认值
            state.key_findings = ["任务执行完成，请查看详细报告"]
            state.recommendations = ["建议检查输出文件以获取更多信息"]

        print(f"✅ 结果分析完成")
        print(f"  - 关键发现数: {len(state.key_findings)}")
        print(f"  - 建议数: {len(state.recommendations)}")

    except Exception as e:
        print(f"⚠️ 结果分析失败: {e}")
        import traceback
        traceback.print_exc()
        state.key_findings = [f"分析过程出错: {str(e)}"]
        state.recommendations = ["请检查系统配置"]

    return state


# ===================== 节点 3: 报告生成 =====================

def generate_report_node(state: ResultEvaluatorState) -> ResultEvaluatorState:
    """
    节点 3: 报告生成

    生成最终总结报告
    """
    print("\n" + "=" * 60)
    print("📝 阶段 3: 生成总结报告")
    print("=" * 60)

    llm = create_bioinformatics_llm()

    try:
        # 构建任务结果摘要
        task_results_text = ""
        for task in state.all_tasks:
            status_icon = "✅" if task.status.upper() == "COMPLETED" else "❌" if task.status.upper() == "FAILED" else "⏳"
            task_results_text += f"\n{status_icon} **{task.task_id}** ({task.status})\n"
            task_results_text += f"   - 类型: {task.task_type}\n"
            task_results_text += f"   - 内容: {_truncate_text(task.content, 100)}\n"
            if task.error:
                task_results_text += f"   - 错误: {_truncate_text(task.error, 100)}\n"

        # 关键发现
        findings_text = "\n".join([f"- {f}" for f in state.key_findings])

        # 建议
        recommendations_text = "\n".join([f"- {r}" for r in state.recommendations])

        # 输出文件
        files_text = ""
        if state.output_files:
            files_text = "\n### 输出文件\n\n" + "\n".join([f"- `{f}`" for f in state.output_files])

        # 使用 LLM 生成报告摘要
        summary_text = ""
        if llm:
            try:
                from langchain_core.messages import HumanMessage

                summary_prompt = f"""
请为以下任务执行结果生成一个简洁的总结段落（200字以内）：

用户问题: {state.user_input[:500]}

执行统计:
- 总任务数: {state.total_tasks}
- 完成数: {state.completed_tasks}
- 失败数: {state.failed_tasks}
- 成功率: {state.success_rate:.1f}%

关键发现:
{findings_text}

请直接输出总结段落，不要包含其他内容。
"""

                response = llm.invoke([HumanMessage(content=summary_prompt)])
                summary_text = response.content if hasattr(response, 'content') else str(response)

            except Exception as e:
                print(f"⚠️ 生成摘要失败: {e}")
                summary_text = f"任务执行完成率 {state.success_rate:.1f}%，共完成 {state.completed_tasks}/{state.total_tasks} 个任务。"

        # 生成详细报告
        detailed_report = f"""# 任务执行总结报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**会话ID**: {state.session_id or 'N/A'}

---

## 摘要

{summary_text}

---

## 执行统计

| 指标 | 数值 |
|------|------|
| 总任务数 | {state.total_tasks} |
| 完成数 | {state.completed_tasks} |
| 失败数 | {state.failed_tasks} |
| 成功率 | {state.success_rate:.1f}% |

---

## 用户原始输入

```
{state.user_input[:1000]}
```

---

## 执行计划

```
{state.execution_plan[:2000] if state.execution_plan else "无执行计划"}
```

---

## 任务执行详情

{task_results_text}

---

## 关键发现

{findings_text}

---

## 建议

{recommendations_text}

{files_text}

---

## 错误摘要

```
{state.error_summary}
```

---

*报告由 Bio-Agent 自动生成*
"""

        state.detailed_report = detailed_report
        state.summary_report = summary_text

        # 保存报告
        report_path = _save_report(detailed_report, "result_evaluation", state.sandbox_dir)
        state.report_path = report_path

        print(f"✅ 报告生成完成")
        print(f"  - 报告路径: {report_path}")

    except Exception as e:
        print(f"⚠️ 报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        state.detailed_report = f"报告生成失败: {str(e)}"
        state.summary_report = "无法生成报告"

    return state


# ===================== 输入/输出映射 =====================

def result_evaluator_input_mapper(global_state: GlobalState) -> ResultEvaluatorState:
    """
    将主图状态映射到 Result Evaluator 子图状态

    Args:
        global_state: 主图全局状态

    Returns:
        Result Evaluator 子图状态
    """
    # 从 global_state 中收集任务结果
    task_results = {}
    all_tasks = []

    # 从 executor 结果中获取任务执行情况
    executor_results = global_state.merged_result.get("executor_results", {})
    task_results_dict = executor_results.get("task_results", {})

    # 获取所有任务（从 subtasks 和 parallel_task_groups）
    all_subtasks = list(global_state.subtasks)
    seen_task_ids = {task.task_id for task in all_subtasks}

    for group in global_state.parallel_task_groups.values():
        if hasattr(group, 'subtasks'):
            for task in group.subtasks:
                if task.task_id not in seen_task_ids:
                    all_subtasks.append(task)
                    seen_task_ids.add(task.task_id)

    # 构建任务摘要
    for task in all_subtasks:
        task_id = task.task_id

        # 获取执行结果
        exec_result = task_results_dict.get(task_id, {})

        # 提取状态
        status = "PENDING"
        if exec_result:
            if hasattr(exec_result, 'status'):
                status = exec_result.status.value if hasattr(exec_result.status, 'value') else str(exec_result.status)
            elif isinstance(exec_result, dict):
                status = exec_result.get('status', 'PENDING')

        # 提取错误
        error = None
        if exec_result:
            if hasattr(exec_result, 'error'):
                error = exec_result.error
            elif isinstance(exec_result, dict):
                error = exec_result.get('error')

        # 提取输出
        output = None
        if exec_result:
            if hasattr(exec_result, 'output'):
                output = exec_result.output
            elif isinstance(exec_result, dict):
                output = exec_result.get('output')

        # 提取执行时间
        execution_time = None
        if exec_result:
            if hasattr(exec_result, 'execution_time'):
                execution_time = exec_result.execution_time
            elif isinstance(exec_result, dict):
                execution_time = exec_result.get('execution_time')

        task_summary = TaskResultSummary(
            task_id=task_id,
            task_type=task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type),
            status=status,
            content=task.content,
            error=error,
            output=output,
            output_files=[],  # 将在 collect_results_node 中填充
            execution_time=execution_time
        )

        task_results[task_id] = task_summary
        all_tasks.append(task_summary)

    # 获取执行计划
    execution_plan = global_state.execution_plan or ""

    # 获取沙盒目录
    sandbox_dir = global_state.sandbox_dir or ""

    return ResultEvaluatorState(
        user_input=global_state.user_input,
        execution_plan=execution_plan,
        task_results=task_results,
        all_tasks=all_tasks,
        sandbox_dir=sandbox_dir,
        session_id=global_state.session_id,
        parent_state=global_state
    )


def result_evaluator_output_mapper(evaluator_state: ResultEvaluatorState, global_state: GlobalState) -> GlobalState:
    """
    将 Result Evaluator 子图状态映射回主图状态

    Args:
        evaluator_state: Result Evaluator 子图状态
        global_state: 主图全局状态

    Returns:
        更新后的主图全局状态
    """
    if not global_state.merged_result:
        global_state.merged_result = {}

    # 存储评估结果
    global_state.merged_result["result_evaluation"] = {
        "summary_report": evaluator_state.summary_report,
        "detailed_report": evaluator_state.detailed_report,
        "report_path": evaluator_state.report_path,
        "key_findings": evaluator_state.key_findings,
        "recommendations": evaluator_state.recommendations,
        "output_files": evaluator_state.output_files,
        "statistics": {
            "total_tasks": evaluator_state.total_tasks,
            "completed_tasks": evaluator_state.completed_tasks,
            "failed_tasks": evaluator_state.failed_tasks,
            "success_rate": evaluator_state.success_rate
        }
    }

    # 存储输出文件路径
    if evaluator_state.output_files:
        if "output_files" not in global_state.file_paths:
            global_state.file_paths["output_files"] = []
        global_state.file_paths["output_files"].extend(evaluator_state.output_files)

    print(f"✅ Result Evaluator 子图完成")
    print(f"  - 总结报告长度: {len(evaluator_state.summary_report)} 字符")
    print(f"  - 详细报告长度: {len(evaluator_state.detailed_report)} 字符")
    print(f"  - 输出文件数: {len(evaluator_state.output_files)}")

    return global_state


# ===================== 构建 Result Evaluator 子图 =====================

def build_result_evaluator_subgraph():
    """
    构建 Result Evaluator 子图

    工作流：
    1. 结果收集 → 2. 结果分析 → 3. 报告生成

    Returns:
        编译后的子图
    """
    graph = StateGraph(ResultEvaluatorState)

    # 添加所有节点
    graph.add_node("collect_results", collect_results_node)  # 阶段 1
    graph.add_node("analyze_results", analyze_results_node)  # 阶段 2
    graph.add_node("generate_report", generate_report_node)  # 阶段 3

    # 定义流程
    graph.add_edge(START, "collect_results")
    graph.add_edge("collect_results", "analyze_results")
    graph.add_edge("analyze_results", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()

