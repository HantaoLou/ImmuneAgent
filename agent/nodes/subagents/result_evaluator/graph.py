"""
Result Evaluator Subgraph

用于总结执行结果并生成最终报告的子图

工作流：
1. 结果收集：收集所有任务的执行结果和输出文件
2. 结果分析：分析成功/失败情况，提取关键发现
3. 报告生成：使用 LLM 生成最终总结报告

增强功能：
- 收集完整的分析流程信息（deep research, hypothesis, execution plan, task list）
- 从沙盒 output 目录读取所有工具输出文件
- 分析 task list 的推导依据
- 生成类似学术论文格式的 TXT 分析报告
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
from utils.llm_factory import (
    create_bioinformatics_llm,
    create_reasoning_llm,
    create_llm_with_thinking,
)
from .state import ResultEvaluatorState, TaskResultSummary, ToolOutputSummary


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

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"📄 {report_type} 报告已保存到: {report_file}")
        return str(report_file)
    except Exception as e:
        print(f"[WARN]️ 保存报告失败: {e}")
        return ""


def _save_txt_report(content: str, report_type: str, sandbox_dir: str) -> str:
    """
    保存 TXT 格式报告到文件

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
        report_file = reports_dir / f"{report_type}_{timestamp}.txt"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"📄 {report_type} TXT 报告已保存到: {report_file}")
        return str(report_file)
    except Exception as e:
        print(f"[WARN]️ 保存 TXT 报告失败: {e}")
        return ""


def _generate_txt_analysis_report(state: "ResultEvaluatorState", llm: Any) -> str:
    """
    生成类似学术论文格式的 TXT 分析报告

    报告格式参考 analysis_report.txt，包含：
    - ANALYSIS OVERVIEW: 分析概述
    - METHODOLOGY: 方法论
    - DEEP RESEARCH: 深度研究结果
    - HYPOTHESIS: 假设
    - TASK LIST DERIVATION: 任务列表推导依据
    - EXECUTION SUMMARY: 执行统计
    - RESULTS SUMMARY: 结果摘要
    - KEY FINDINGS: 关键发现
    - SCIENTIFIC RATIONALE: 科学依据
    - LIMITATIONS: 局限性
    - VALIDATION RECOMMENDATIONS: 验证建议
    - FILES GENERATED: 生成的文件
    - CONCLUSION: 结论

    Args:
        state: Result Evaluator 状态
        llm: LLM 实例

    Returns:
        TXT 格式的分析报告
    """
    # 构建任务列表文本
    task_list_text = ""
    for i, task in enumerate(state.all_tasks, 1):
        status_str = (
            "[OK]"
            if task.status.upper() == "COMPLETED"
            else "[FAIL]"
            if task.status.upper() == "FAILED"
            else "⏳"
        )
        task_list_text += f"\n{i}. [{status_str}] {task.task_id}: {task.task_type}\n"
        task_list_text += f"   内容: {_truncate_text(task.content, 150)}\n"
        if task.error:
            task_list_text += f"   错误: {_truncate_text(task.error, 100)}\n"

    # 构建 Deep Research 文本
    deep_research_text = ""
    if state.deep_research and state.deep_research.research_summary:
        deep_research_text = f"\n{state.deep_research.research_summary}\n"
        if state.deep_research.key_insights:
            deep_research_text += "\n关键洞察:\n"
            for insight in state.deep_research.key_insights[:5]:
                deep_research_text += f"- {insight}\n"
        if state.deep_research.evidence:
            deep_research_text += "\n证据:\n"
            for evidence in state.deep_research.evidence[:5]:
                deep_research_text += f"- {evidence}\n"

    # 构建 Hypothesis 文本
    hypothesis_text = ""
    if state.hypothesis and state.hypothesis.hypothesis_summary:
        hypothesis_text = f"\n{state.hypothesis.hypothesis_summary}\n"
        if state.hypothesis.testable_predictions:
            hypothesis_text += "\n可验证的预测:\n"
            for pred in state.hypothesis.testable_predictions[:5]:
                hypothesis_text += f"- {pred}\n"

    # 构建 Task List 推导依据文本
    derivation_text = ""
    if state.task_list_derivation and state.task_list_derivation.decomposition_summary:
        derivation_text = (
            f"\n{state.task_list_derivation.decomposition_summary}\n\n所需服务:\n"
        )
        for service in state.task_list_derivation.required_services[:10]:
            derivation_text += f"- {service}\n"
        derivation_text += (
            f"\n依赖关系依据:\n{state.task_list_derivation.dependency_rationale}\n"
        )
        if state.task_list_derivation.parallel_groups:
            derivation_text += "\n并行执行组:\n"
            for (
                group_id,
                group_info,
            ) in state.task_list_derivation.parallel_groups.items():
                derivation_text += (
                    f"- {group_id}: {group_info.get('subtask_count', 0)} 个子任务\n"
                )

    # 构建输出文件文本
    output_files_text = ""
    if state.output_files:
        output_files_text = "\n输出文件:\n"
        for f in state.output_files[:20]:
            output_files_text += f"  - {f}\n"
        if len(state.output_files) > 20:
            output_files_text += f"  ... 还有 {len(state.output_files) - 20} 个文件\n"

    # 构建工具输出摘要文本
    tool_outputs_summary = _analyze_tool_outputs_for_report(state.tool_output_summaries)

    # 使用 LLM 生成深度分析（方法、科学依据、局限性等）
    methodology = ""
    scientific_rationale = ""
    limitations = []
    validation_recommendations = []

    if llm:
        try:
            from langchain_core.messages import HumanMessage

            # 提取原始问题的关键信息（如样本数、预测目标等）
            user_question = state.user_input[:2000]

            analysis_prompt = f"""
你是一个专业的生物医药研究分析专家。请根据以下计算分析过程和结果，生成一份专业的分析报告摘要。

## 用户原始问题
{user_question}

## 执行计划摘要
{state.execution_plan[:2000] if state.execution_plan else "无"}

## Deep Research 发现
{deep_research_text[:1500] if deep_research_text else "无"}

## 假设
{hypothesis_text[:1500] if hypothesis_text else "无"}

## 执行统计
- 总任务数: {state.total_tasks}
- 完成数: {state.completed_tasks}
- 失败数: {state.failed_tasks}
- 成功率: {state.success_rate:.1f}%

## 任务列表
{task_list_text[:3000]}

## 工具输出摘要
{tool_outputs_summary[:2000]}

## 关键发现
{chr(10).join([f"- {f}" for f in state.key_findings[:10]]) if state.key_findings else "无"}

请根据以上信息，以 JSON 格式返回以下内容（用专业的学术语言）：
{{
    "methodology": "分析方法描述（2-3句话，描述使用了什么计算方法和工具来解决问题，例如：This computational method identifies... by integrating multiple biological features...）",
    "scientific_rationale": "科学依据（2-3句话，描述分析的科学基础和理论依据，例如：Broadly neutralizing antibodies typically exhibit...）",
    "limitations": ["局限性1（描述分析方法的局限性，如数据依赖性、假设限制等）", "局限性2", "局限性3"],
    "validation_recommendations": ["验证建议1（描述如何通过实验验证分析结果）", "验证建议2", "验证建议3"]
}}

注意：
1. 使用英文撰写，风格参考学术论文
2. methodology 应描述具体的分析方法（如特征提取、评分计算、加权组合等）
3. scientific_rationale 应描述分析的科学依据（如已知的生物学规律、相关性等）
4. limitations 应诚实地指出分析方法的局限性
5. validation_recommendations 应提供具体的实验验证建议
"""

            response = llm.invoke([HumanMessage(content=analysis_prompt)])
            response_content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # 解析 JSON
            import re

            json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
            if json_match:
                analysis_result = json.loads(json_match.group())
                methodology = analysis_result.get("methodology", "")
                scientific_rationale = analysis_result.get("scientific_rationale", "")
                limitations = analysis_result.get("limitations", [])
                validation_recommendations = analysis_result.get(
                    "validation_recommendations", []
                )

        except Exception as e:
            print(f"[WARN]️ 生成深度分析失败: {e}")
            methodology = "This computational method integrates multiple analytical tools and domain knowledge to address the research question through a multi-stage workflow approach."
            scientific_rationale = "The analysis is based on established biological principles and computational methods from the bioinformatics domain."
            limitations = [
                "Results require experimental validation before drawing final conclusions",
                "Analysis quality depends on input data completeness and accuracy",
                "Computational predictions may not capture all biological complexity",
            ]
            validation_recommendations = [
                "Perform experimental validation using appropriate assays",
                "Verify input data quality and completeness",
                "Cross-reference results with known literature",
            ]

    # 更新状态
    state.methodology = methodology
    state.scientific_rationale = scientific_rationale
    state.limitations = limitations
    state.validation_recommendations = validation_recommendations

    # 生成完整的 TXT 报告
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 构建关键发现文本
    key_findings_text = ""
    for i, finding in enumerate(state.key_findings, 1):
        key_findings_text += f"{i}. {finding}\n"

    # 构建局限性文本
    limitations_text = ""
    for i, limitation in enumerate(limitations, 1):
        limitations_text += f"{i}. {limitation}\n"

    # 构建验证建议文本
    validation_text = ""
    for i, rec in enumerate(validation_recommendations, 1):
        validation_text += f"{i}. {rec}\n"

    # 构建建议文本
    recommendations_text = ""
    for i, rec in enumerate(state.recommendations, 1):
        recommendations_text += f"{i}. {rec}\n"

    # 构建文件列表文本
    files_generated_text = ""
    if state.output_files:
        for i, f in enumerate(state.output_files[:20], 1):
            files_generated_text += f"{i}. {f}\n"
        if len(state.output_files) > 20:
            files_generated_text += (
                f"... and {len(state.output_files) - 20} more files\n"
            )
    else:
        files_generated_text = "No output files generated.\n"

    # ========== 生成报告标题 ==========
    # 从用户问题中提取关键主题
    title = "ANALYSIS REPORT"
    if "antibod" in state.user_input.lower():
        if "neutraliz" in state.user_input.lower():
            if "broad" in state.user_input.lower():
                title = (
                    "BROADLY NEUTRALIZING ANTIBODY (BNAB) PREDICTION ANALYSIS REPORT"
                )
            else:
                title = "ANTIBODY NEUTRALIZATION PREDICTION ANALYSIS REPORT"
        elif "bind" in state.user_input.lower():
            title = "ANTIBODY BINDING PREDICTION ANALYSIS REPORT"
        else:
            title = "ANTIBODY ANALYSIS REPORT"
    elif "tcr" in state.user_input.lower() or "t cell" in state.user_input.lower():
        if "bind" in state.user_input.lower():
            title = "TCR-EPITOPE BINDING PREDICTION ANALYSIS REPORT"
        else:
            title = "T CELL RECEPTOR ANALYSIS REPORT"
    elif "b cell" in state.user_input.lower():
        title = "B CELL ANALYSIS REPORT"

    # ========== 生成完整报告 ==========
    txt_report = f"""
{"=" * 70}
{title}
{"=" * 70}

ANALYSIS OVERVIEW
-----------------
This computational method identifies biological patterns and predictions by 
integrating multiple analytical features. The analysis was performed using 
AI-powered bioinformatics tools.

Original Question:
{state.user_input[:800]}

"""

    # ========== METHODOLOGY ==========
    txt_report += f"""METHODOLOGY
-----------
{methodology}
"""

    # ========== DEEP RESEARCH ==========
    if deep_research_text:
        txt_report += f"""
DEEP RESEARCH
-------------
{deep_research_text}
"""

    # ========== HYPOTHESIS ==========
    if hypothesis_text:
        txt_report += f"""
HYPOTHESIS
----------
{hypothesis_text}
"""

    # ========== TASK LIST DERIVATION ==========
    if derivation_text:
        txt_report += f"""
TASK LIST DERIVATION
--------------------
{derivation_text}
"""

    # ========== EXECUTION SUMMARY ==========
    txt_report += f"""
EXECUTION SUMMARY
-----------------
Total tasks: {state.total_tasks}
Completed: {state.completed_tasks}
Failed: {state.failed_tasks}
Success rate: {state.success_rate:.1f}%

Task List:
{task_list_text}
"""

    # ========== TOOL OUTPUTS ==========
    if tool_outputs_summary:
        txt_report += f"""
TOOL OUTPUTS SUMMARY
--------------------
{tool_outputs_summary}
"""

    # ========== RESULTS SUMMARY ==========
    txt_report += f"""
RESULTS SUMMARY
---------------
{state.summary_report if state.summary_report else "Analysis completed. See key findings below."}
"""

    # ========== KEY FINDINGS ==========
    if key_findings_text:
        txt_report += f"""
KEY FINDINGS
------------
{key_findings_text}"""

    # ========== SCIENTIFIC RATIONALE ==========
    txt_report += f"""
SCIENTIFIC RATIONALE
--------------------
{scientific_rationale}
"""

    # ========== LIMITATIONS ==========
    if limitations_text:
        txt_report += f"""LIMITATIONS
-----------
{limitations_text}
"""

    # ========== VALIDATION RECOMMENDATIONS ==========
    if validation_text:
        txt_report += f"""VALIDATION RECOMMENDATIONS
--------------------------
{validation_text}
"""

    # ========== RECOMMENDATIONS ==========
    if recommendations_text:
        txt_report += f"""RECOMMENDATIONS
---------------
{recommendations_text}
"""

    # ========== FILES GENERATED ==========
    txt_report += f"""FILES GENERATED
---------------
{files_generated_text}
"""

    # ========== ERROR SUMMARY ==========
    if state.error_summary and state.error_summary != "无错误":
        txt_report += f"""ERROR SUMMARY
-------------
{state.error_summary}

"""

    # ========== CONCLUSION ==========
    txt_report += f"""CONCLUSION
----------
This computational pipeline has successfully processed the research question
using a multi-stage workflow approach. The results should be validated
experimentally before drawing final conclusions.

Generated: {timestamp}
Session ID: {state.session_id or "N/A"}

{"=" * 70}
"""

    return txt_report


def _extract_output_files(task_result: Any) -> List[str]:
    """从任务结果中提取输出文件路径"""
    output_files = []

    if task_result is None:
        return output_files

    # 尝试从 output 中提取文件路径
    output = None
    if hasattr(task_result, "output"):
        output = task_result.output
    elif isinstance(task_result, dict):
        output = task_result.get("output")

    if output is None:
        return output_files

    # 如果 output 是字符串，尝试解析 JSON
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            # 如果不是 JSON，检查是否是文件路径
            if "/" in output or "\\" in output:
                output_files.append(output)
            return output_files

    if not isinstance(output, dict):
        return output_files

    # 常见的文件路径字段名
    file_fields = [
        "output_file",
        "output_path",
        "file_path",
        "result_file",
        "output_files",
        "files",
        "result_files",
        "saved_files",
        "csv_file",
        "json_file",
        "report_file",
        "log_file",
    ]

    for field in file_fields:
        if field in output:
            value = output[field]
            if isinstance(value, str):
                output_files.append(value)
            elif isinstance(value, list):
                output_files.extend([f for f in value if isinstance(f, str)])

    # 检查 final_result 中的文件
    if "final_result" in output:
        final_result = output["final_result"]
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


def _collect_output_files_from_sandbox(
    sandbox_dir: str, session_id: Optional[str] = None, parent_state: Any = None
) -> List[ToolOutputSummary]:
    """
    从沙盒 output 目录收集所有工具输出文件

    通过 CodeAct 子图统一执行代码，遵循架构原则：
    - 其他子图不直接调用 OpenSandbox
    - 所有沙盒操作通过 CodeAct 执行

    沙盒目录结构：
    - sandbox_dir 可能是 /data/sessions/{session_id} 或容器内路径 /data/sessions/{session_id}
    - 输出文件位于 {sandbox_dir}/output/ 目录下

    Args:
        sandbox_dir: 沙盒目录路径
        session_id: 会话ID
        parent_state: 父状态，用于获取 opensandbox_id

    Returns:
        工具输出摘要列表
    """
    tool_outputs = []

    if not sandbox_dir:
        return tool_outputs

    # 获取现有的 sandbox_id（如果有的话）
    existing_sandbox_id = None
    if parent_state:
        merged_result = getattr(parent_state, "merged_result", None) or {}
        existing_sandbox_id = merged_result.get("opensandbox_id")

    # 使用 CodeAct 统一接口执行代码（架构原则：唯一与 OpenSandbox 沟通的入口）
    from utils.codeact_executor import execute_code_via_codeact, is_codeact_available

    if not is_codeact_available():
        print(f"  [OutputCollector] CodeAct/OpenSandbox 未启用，无法读取远程文件")
        return tool_outputs

    print(f"  [OutputCollector] 通过 CodeAct 读取远程文件...")
    print(
        f"  [OutputCollector] sandbox_dir={sandbox_dir}, sandbox_id={existing_sandbox_id}"
    )

    # 构建在沙盒内执行的代码
    # 这段代码会扫描 output 目录，收集文件列表和内容预览
    collector_code = f'''
import os
import json
from pathlib import Path

sandbox_dir = "{sandbox_dir}"
output_dir = Path(sandbox_dir) / "output"

# 如果 output 目录不存在，尝试直接在 sandbox_dir 下查找
if not output_dir.exists():
    output_dir = Path(sandbox_dir)

results = []

if output_dir.exists():
    # 支持的文件类型
    supported_extensions = {{
        '.csv': 'CSV Data',
        '.json': 'JSON Data', 
        '.txt': 'Text Report',
        '.md': 'Markdown Report',
        '.tsv': 'TSV Data',
        '.fasta': 'FASTA Sequence',
        '.fa': 'FASTA Sequence',
        '.airr': 'AIRR TSV Data',
        '.h5ad': 'AnnData Object',
        '.rds': 'R Data Object',
        '.pdf': 'PDF Report',
        '.png': 'Image',
        '.jpg': 'Image',
    }}
    
    for file_path in output_dir.rglob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext in supported_extensions:
                try:
                    rel_path = file_path.relative_to(output_dir)
                except ValueError:
                    rel_path = file_path.name
                
                content_preview = ""
                key_results = []
                
                try:
                    if ext in ['.csv', '.tsv', '.txt', '.json', '.md', '.fasta', '.fa', '.airr']:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(2000)
                            content_preview = content[:500]
                            
                            if ext == '.csv':
                                lines = content.split('\\n')[:5]
                                if lines:
                                    key_results.append(f"列: {{lines[0][:200]}}")
                                    if len(lines) > 1:
                                        key_results.append(f"数据行示例: {{lines[1][:200]}}")
                            elif ext == '.json':
                                try:
                                    data = json.loads(content)
                                    if isinstance(data, dict):
                                        key_results = list(data.keys())[:5]
                                    elif isinstance(data, list):
                                        key_results.append(f"数组长度: {{len(data)}}")
                                except:
                                    pass
                except Exception as e:
                    content_preview = f"[无法读取: {{e}}]"
                
                results.append({{
                    "file_path": str(file_path),
                    "file_type": supported_extensions[ext],
                    "content_preview": content_preview,
                    "key_results": key_results
                }})

# 输出 JSON 结果
print("__OUTPUT_FILES_JSON_START__")
print(json.dumps(results, ensure_ascii=False))
print("__OUTPUT_FILES_JSON_END__")
'''

    try:
        # 通过 CodeAct 统一接口执行代码
        result = execute_code_via_codeact(
            task_description=f"收集沙盒 {sandbox_dir}/output 目录下的所有输出文件",
            code_template=collector_code,
            sandbox_id=existing_sandbox_id,
            timeout_seconds=60,
            keep_alive=True,
        )

        if not result.is_success():
            print(f"  [OutputCollector] 沙盒执行失败: {result.error}")
            return tool_outputs

        # 解析输出
        stdout = result.output

        # 提取 JSON 结果
        import re

        json_match = re.search(
            r"__OUTPUT_FILES_JSON_START__\s*(.*?)\s*__OUTPUT_FILES_JSON_END__",
            stdout,
            re.DOTALL,
        )

        if json_match:
            files_data = json.loads(json_match.group(1))
            for file_data in files_data:
                tool_outputs.append(
                    ToolOutputSummary(
                        file_path=file_data.get("file_path", ""),
                        file_type=file_data.get("file_type", "Unknown"),
                        content_preview=file_data.get("content_preview", ""),
                        key_results=file_data.get("key_results", []),
                    )
                )

            print(f"  [OutputCollector] 找到 {len(tool_outputs)} 个输出文件")
        else:
            # 尝试使用自动解析的结果
            if result.parsed_result and isinstance(result.parsed_result, dict):
                items = result.parsed_result.get("items", [])
                for file_data in items:
                    tool_outputs.append(
                        ToolOutputSummary(
                            file_path=file_data.get("file_path", ""),
                            file_type=file_data.get("file_type", "Unknown"),
                            content_preview=file_data.get("content_preview", ""),
                            key_results=file_data.get("key_results", []),
                        )
                    )
                print(
                    f"  [OutputCollector] 通过自动解析找到 {len(tool_outputs)} 个输出文件"
                )
            else:
                print(f"  [OutputCollector] 未找到输出文件 JSON 结果")

    except Exception as e:
        print(f"  [OutputCollector] 收集文件失败: {e}")
        import traceback

        traceback.print_exc()

    return tool_outputs


def _analyze_tool_outputs_for_report(tool_outputs: List[ToolOutputSummary]) -> str:
    """
    分析工具输出，生成用于报告的文本摘要

    Args:
        tool_outputs: 工具输出摘要列表

    Returns:
        用于报告的文本摘要
    """
    if not tool_outputs:
        return "No tool output files generated."

    # 按文件类型分组
    by_type: Dict[str, List[ToolOutputSummary]] = {}
    for output in tool_outputs:
        if output.file_type not in by_type:
            by_type[output.file_type] = []
        by_type[output.file_type].append(output)

    summary_lines = [f"Total output files: {len(tool_outputs)}", ""]

    for file_type, files in by_type.items():
        summary_lines.append(f"{file_type} ({len(files)} files):")
        for f in files[:5]:  # 每种类型最多显示5个
            summary_lines.append(f"  - {Path(f.file_path).name}")
        if len(files) > 5:
            summary_lines.append(f"  ... and {len(files) - 5} more")
        summary_lines.append("")

    return "\n".join(summary_lines)


# ===================== 节点 1: 结果收集 =====================


def collect_results_node(state: ResultEvaluatorState) -> ResultEvaluatorState:
    """
    节点 1: 结果收集

    收集所有任务的执行结果和输出文件
    增强功能：通过 OpenSandbox 从远程沙盒 output 目录读取所有工具输出文件
    """
    print("\n" + "=" * 60)
    print("[STAT] 阶段 1: 收集执行结果")
    print("=" * 60)

    # 统计任务状态
    total = len(state.all_tasks)
    completed = sum(1 for t in state.all_tasks if t.status.upper() == "COMPLETED")
    failed = sum(1 for t in state.all_tasks if t.status.upper() == "FAILED")

    state.total_tasks = total
    state.completed_tasks = completed
    state.failed_tasks = failed
    state.success_rate = (completed / total * 100) if total > 0 else 0.0

    # 收集所有输出文件（从任务结果中）
    all_output_files = []
    for task in state.all_tasks:
        task_files = _extract_output_files(task)
        all_output_files.extend(task_files)

    # 去重
    state.output_files = list(set(all_output_files))

    # ========== 增强：通过 OpenSandbox 从远程沙盒 output 目录收集工具输出文件 ==========
    print(f"  [OutputCollector] 从远程沙盒目录收集工具输出文件...")

    # 优先使用 sandbox_data_dir（会话目录格式：/data/sessions/{session_id}）
    # 然后回退到 sandbox_dir
    sandbox_dir = None
    if state.parent_state:
        sandbox_dir = getattr(state.parent_state, "sandbox_data_dir", None)
    if not sandbox_dir:
        sandbox_dir = state.sandbox_dir

    # 传递 parent_state 以获取现有的 sandbox_id
    tool_outputs = _collect_output_files_from_sandbox(
        sandbox_dir=sandbox_dir,
        session_id=state.session_id,
        parent_state=state.parent_state,
    )
    state.tool_output_summaries = tool_outputs

    # 将工具输出文件路径也添加到 output_files
    for tool_output in tool_outputs:
        if tool_output.file_path not in state.output_files:
            state.output_files.append(tool_output.file_path)

    print(f"  [OutputCollector] 收集到 {len(tool_outputs)} 个工具输出文件")

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

    print(f"[SUCCESS] 结果收集完成")
    print(f"  - 总任务数: {total}")
    print(f"  - 完成数: {completed}")
    print(f"  - 失败数: {failed}")
    print(f"  - 成功率: {state.success_rate:.1f}%")
    print(f"  - 输出文件数: {len(state.output_files)}")
    print(f"  - 工具输出摘要数: {len(state.tool_output_summaries)}")

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

    llm = (
        create_llm_with_thinking(
            purpose="bioinformatics",
            progress_callback=getattr(state, "progress_callback", None),
            session_id=getattr(state, "session_id", None),
            node_name="result_evaluator",
        )
        or create_bioinformatics_llm()
    )
    if not llm:
        print("[WARN]️ LLM 不可用，使用简单分析")
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
            files_text = "输出文件:\n" + "\n".join(
                [f"- {f}" for f in state.output_files[:20]]
            )

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
        response_content = (
            response.content if hasattr(response, "content") else str(response)
        )

        # 尝试解析 JSON
        try:
            # 尝试提取 JSON 块
            import re

            json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
            if json_match:
                analysis_result = json.loads(json_match.group())
            else:
                analysis_result = json.loads(response_content)

            state.key_findings = analysis_result.get("key_findings", [])
            state.recommendations = analysis_result.get("recommendations", [])

        except (json.JSONDecodeError, TypeError) as e:
            print(f"[WARN]️ JSON 解析失败: {e}")
            # 使用默认值
            state.key_findings = ["任务执行完成，请查看详细报告"]
            state.recommendations = ["建议检查输出文件以获取更多信息"]

        print(f"[SUCCESS] 结果分析完成")
        print(f"  - 关键发现数: {len(state.key_findings)}")
        print(f"  - 建议数: {len(state.recommendations)}")

    except Exception as e:
        print(f"[WARN]️ 结果分析失败: {e}")
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

    llm = (
        create_llm_with_thinking(
            purpose="bioinformatics",
            progress_callback=getattr(state, "progress_callback", None),
            session_id=getattr(state, "session_id", None),
            node_name="result_evaluator",
        )
        or create_bioinformatics_llm()
    )

    try:
        # 构建任务结果摘要
        task_results_text = ""
        for task in state.all_tasks:
            status_icon = (
                "[SUCCESS]"
                if task.status.upper() == "COMPLETED"
                else "[ERROR]"
                if task.status.upper() == "FAILED"
                else "⏳"
            )
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
            files_text = "\n### 输出文件\n\n" + "\n".join(
                [f"- `{f}`" for f in state.output_files]
            )

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
                summary_text = (
                    response.content if hasattr(response, "content") else str(response)
                )

            except Exception as e:
                print(f"[WARN]️ 生成摘要失败: {e}")
                summary_text = f"任务执行完成率 {state.success_rate:.1f}%，共完成 {state.completed_tasks}/{state.total_tasks} 个任务。"

        # 生成详细报告
        detailed_report = f"""# 任务执行总结报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**会话ID**: {state.session_id or "N/A"}

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

        # 保存 Markdown 报告
        report_path = _save_report(
            detailed_report, "result_evaluation", state.sandbox_dir
        )
        state.report_path = report_path

        # ========== 生成 TXT 格式分析报告 ==========
        txt_report = _generate_txt_analysis_report(state, llm)
        state.txt_report = txt_report

        # 保存 TXT 报告
        txt_report_path = _save_txt_report(
            txt_report, "analysis_report", state.sandbox_dir
        )
        state.txt_report_path = txt_report_path

        print(f"[SUCCESS] 报告生成完成")
        print(f"  - Markdown 报告路径: {report_path}")
        print(f"  - TXT 分析报告路径: {txt_report_path}")

    except Exception as e:
        print(f"[WARN]️ 报告生成失败: {e}")
        import traceback

        traceback.print_exc()
        state.detailed_report = f"报告生成失败: {str(e)}"
        state.summary_report = "无法生成报告"

    return state


# ===================== 输入/输出映射 =====================


def result_evaluator_input_mapper(global_state: GlobalState) -> ResultEvaluatorState:
    """
    将主图状态映射到 Result Evaluator 子图状态

    增强功能：
    - 收集完整的分析流程信息（deep research, hypothesis, execution plan, task list）
    - 收集 task list 的推导依据
    - 优先使用 sandbox_data_dir（会话目录）来读取工具输出

    Args:
        global_state: 主图全局状态

    Returns:
        Result Evaluator 子图状态
    """
    from .state import DeepResearchInfo, HypothesisInfo, TaskListDerivation

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
        if hasattr(group, "subtasks"):
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
            if hasattr(exec_result, "status"):
                status = (
                    exec_result.status.value
                    if hasattr(exec_result.status, "value")
                    else str(exec_result.status)
                )
            elif isinstance(exec_result, dict):
                status = exec_result.get("status", "PENDING")

        # 提取错误
        error = None
        if exec_result:
            if hasattr(exec_result, "error"):
                error = exec_result.error
            elif isinstance(exec_result, dict):
                error = exec_result.get("error")

        # 提取输出
        output = None
        if exec_result:
            if hasattr(exec_result, "output"):
                output = exec_result.output
            elif isinstance(exec_result, dict):
                output = exec_result.get("output")

        # 提取执行时间
        execution_time = None
        if exec_result:
            if hasattr(exec_result, "execution_time"):
                execution_time = exec_result.execution_time
            elif isinstance(exec_result, dict):
                execution_time = exec_result.get("execution_time")

        task_summary = TaskResultSummary(
            task_id=task_id,
            task_type=task.task_type.value
            if hasattr(task.task_type, "value")
            else str(task.task_type),
            status=status,
            content=task.content,
            error=error,
            output=output,
            output_files=[],  # 将在 collect_results_node 中填充
            execution_time=execution_time,
        )

        task_results[task_id] = task_summary
        all_tasks.append(task_summary)

    # 获取执行计划
    execution_plan = global_state.execution_plan or ""

    # 优先使用 sandbox_data_dir（会话目录），然后回退到 sandbox_dir
    # sandbox_data_dir 格式: /data/sessions/{session_id}
    sandbox_dir = global_state.sandbox_data_dir or global_state.sandbox_dir or ""

    # ========== 收集 Immunity 子图信息 ==========
    immunity_plan = global_state.merged_result.get("immunity_plan", {})

    # 收集 Deep Research 信息（增强：收集更完整的信息）
    deep_research = DeepResearchInfo(
        research_summary=immunity_plan.get("research_summary", ""),
        key_insights=immunity_plan.get("research_insights", []),
        evidence=immunity_plan.get("research_evidence", []),
        knowledge_gaps=immunity_plan.get("research_gaps", []),
        confidence=immunity_plan.get("research_confidence", 0.0),
    )

    # 收集 Hypothesis 信息（增强：收集更完整的信息）
    hypothesis = HypothesisInfo(
        hypothesis_summary=immunity_plan.get("hypothesis_summary", ""),
        testable_predictions=immunity_plan.get("testable_predictions", []),
        confidence=immunity_plan.get("hypothesis_confidence", 0.0),
    )

    # 收集 Task List 推导依据（增强：收集更完整的分解依据）
    task_decomp_results = global_state.merged_result.get("task_decomposition", {})
    required_service_ids = task_decomp_results.get("required_service_ids", [])
    raw_tasks = task_decomp_results.get("raw_tasks", [])

    # 构建更详细的任务分解摘要
    decomposition_details = []
    if required_service_ids:
        decomposition_details.append(
            f"所需服务: {', '.join(required_service_ids[:10])}"
        )
    if raw_tasks:
        decomposition_details.append(f"初始任务数: {len(raw_tasks)}")

    # 收集并行组信息
    parallel_group_info = {}
    for group_id, group in global_state.parallel_task_groups.items():
        if hasattr(group, "subtasks"):
            parallel_group_info[group_id] = {
                "subtask_count": len(group.subtasks),
                "task_types": list(
                    set(
                        t.task_type.value
                        if hasattr(t.task_type, "value")
                        else str(t.task_type)
                        for t in group.subtasks
                    )
                ),
            }

    task_list_derivation = TaskListDerivation(
        decomposition_summary=f"基于执行计划分解为 {len(all_subtasks)} 个子任务。"
        + (" ".join(decomposition_details) if decomposition_details else ""),
        required_services=required_service_ids
        if isinstance(required_service_ids, list)
        else [],
        dependency_rationale="任务按数据流和依赖关系排序执行，确保上游任务的输出可作为下游任务的输入。",
        parallel_groups=parallel_group_info,
    )

    return ResultEvaluatorState(
        user_input=global_state.user_input,
        execution_plan=execution_plan,
        deep_research=deep_research,
        hypothesis=hypothesis,
        task_list_derivation=task_list_derivation,
        task_results=task_results,
        all_tasks=all_tasks,
        sandbox_dir=sandbox_dir,
        session_id=global_state.session_id,
        parent_state=global_state,
    )


def result_evaluator_output_mapper(
    evaluator_state: ResultEvaluatorState, global_state: GlobalState
) -> GlobalState:
    """
    将 Result Evaluator 子图状态映射回主图状态

    Args:
        evaluator_state: Result Evaluator 子图状态 (可以是 ResultEvaluatorState 对象或 dict)
        global_state: 主图全局状态

    Returns:
        更新后的主图全局状态
    """
    if not global_state.merged_result:
        global_state.merged_result = {}

    # Handle case where evaluator_state might be a dict (from LangGraph invoke)
    if isinstance(evaluator_state, dict):
        summary_report = evaluator_state.get("summary_report", "")
        detailed_report = evaluator_state.get("detailed_report", "")
        txt_report = evaluator_state.get("txt_report", "")
        report_path = evaluator_state.get("report_path", "")
        txt_report_path = evaluator_state.get("txt_report_path", "")
        key_findings = evaluator_state.get("key_findings", [])
        recommendations = evaluator_state.get("recommendations", [])
        output_files = evaluator_state.get("output_files", [])
        methodology = evaluator_state.get("methodology", "")
        scientific_rationale = evaluator_state.get("scientific_rationale", "")
        limitations = evaluator_state.get("limitations", [])
        validation_recommendations = evaluator_state.get(
            "validation_recommendations", []
        )
        total_tasks = evaluator_state.get("total_tasks", 0)
        completed_tasks = evaluator_state.get("completed_tasks", 0)
        failed_tasks = evaluator_state.get("failed_tasks", 0)
        success_rate = evaluator_state.get("success_rate", 0.0)
    else:
        summary_report = getattr(evaluator_state, "summary_report", "")
        detailed_report = getattr(evaluator_state, "detailed_report", "")
        txt_report = getattr(evaluator_state, "txt_report", "")
        report_path = getattr(evaluator_state, "report_path", "")
        txt_report_path = getattr(evaluator_state, "txt_report_path", "")
        key_findings = getattr(evaluator_state, "key_findings", []) or []
        recommendations = getattr(evaluator_state, "recommendations", []) or []
        output_files = getattr(evaluator_state, "output_files", []) or []
        methodology = getattr(evaluator_state, "methodology", "")
        scientific_rationale = getattr(evaluator_state, "scientific_rationale", "")
        limitations = getattr(evaluator_state, "limitations", []) or []
        validation_recommendations = (
            getattr(evaluator_state, "validation_recommendations", []) or []
        )
        total_tasks = getattr(evaluator_state, "total_tasks", 0)
        completed_tasks = getattr(evaluator_state, "completed_tasks", 0)
        failed_tasks = getattr(evaluator_state, "failed_tasks", 0)
        success_rate = getattr(evaluator_state, "success_rate", 0.0)

    # 存储评估结果
    global_state.merged_result["result_evaluation"] = {
        "summary_report": summary_report,
        "detailed_report": detailed_report,
        "txt_report": txt_report,
        "report_path": report_path,
        "txt_report_path": txt_report_path,
        "key_findings": key_findings,
        "recommendations": recommendations,
        "output_files": output_files,
        "methodology": methodology,
        "scientific_rationale": scientific_rationale,
        "limitations": limitations,
        "validation_recommendations": validation_recommendations,
        "statistics": {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": success_rate,
        },
    }

    # 存储输出文件路径到专门的列表字段（file_paths 的值必须是 str 类型）
    if output_files:
        # 去重后添加到 completed_output_files
        existing = set(global_state.completed_output_files)
        for f in output_files:
            if f not in existing:
                global_state.completed_output_files.append(f)

    print(f"[SUCCESS] Result Evaluator 子图完成")
    print(f"  - 总结报告长度: {len(summary_report)} 字符")
    print(f"  - 详细报告长度: {len(detailed_report)} 字符")
    print(f"  - TXT 报告长度: {len(txt_report)} 字符")
    print(f"  - 输出文件数: {len(output_files)}")

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
