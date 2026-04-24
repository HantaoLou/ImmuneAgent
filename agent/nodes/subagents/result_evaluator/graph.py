"""
Result Evaluator Subgraph

Subgraph for summarizing execution results and generating final reports

Workflow:
1. Results collection: Collect execution results and output files from all tasks
2. Results analysis: Analyze success/failure status, extract key findings
3. Report generation: Use LLM to generate final summary report

Enhanced Features:
- Collect complete analysis pipeline information (deep research, hypothesis, execution plan, task list)
- Read all tool output files from sandbox output directory
- Analyze task list derivation basis
- Generate academic paper style TXT analysis report
"""

import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from langgraph.graph import StateGraph, START, END

# Add agent directory to path
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


# ===================== Helper Functions =====================


def _get_progress_callback_by_session(session_id: Optional[str]) -> Optional[Any]:
    """
    Get progress callback from global registry by session_id

    Args:
        session_id: Session ID to look up

    Returns:
        Progress callback function if found, None otherwise
    """
    if not session_id:
        return None

    try:
        from pathlib import Path

        backend_dir = Path(__file__).parent.parent.parent.parent / "backend"
        project_root = backend_dir.parent

        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from backend import progress_tracker as pt_module

        callback = pt_module.get_progress_callback(session_id)
        print(
            f"[ResultEvaluator] Got callback for session {session_id}: {callback is not None}"
        )
        return callback
    except (ImportError, AttributeError) as e:
        print(f"[ResultEvaluator] Failed to get callback: {e}")
        return None


def _save_report_to_opensandbox(
    content: str,
    report_type: str,
    opensandbox_id: Optional[str],
    sandbox_dir: str,
) -> str:
    """
    Save report to OpenSandbox using opensandbox_executor.

    Args:
        content: Report content
        report_type: Report type (e.g., "result_evaluation", "analysis_report")
        opensandbox_id: OpenSandbox instance ID to connect to (None to create new sandbox)
        sandbox_dir: Sandbox directory base path

    Returns:
        Remote file path if successful, empty string otherwise
    """
    from utils.opensandbox_executor import save_file_to_opensandbox_sync

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = "txt" if report_type == "analysis_report" else "md"
    remote_file_path = f"{sandbox_dir}/output/reports/{report_type}_{timestamp}.{ext}"

    result = save_file_to_opensandbox_sync(
        file_path=remote_file_path,
        content=content,
        existing_sandbox_id=opensandbox_id,
    )

    if result.get("success"):
        print(f"  [ReportSave] ✅ {report_type} saved to: {remote_file_path}")
        return remote_file_path
    else:
        print(f"  [ReportSave] ❌ Failed to save {report_type}: {result.get('error')}")
        return ""


def _generate_txt_analysis_report(state: "ResultEvaluatorState", llm: Any) -> str:
    """
    Generate academic paper style TXT analysis report

    Report format references analysis_report.txt, containing:
    - ANALYSIS OVERVIEW: Analysis overview
    - METHODOLOGY: Methodology
    - DEEP RESEARCH: Deep research findings
    - HYPOTHESIS: Hypothesis
    - TASK LIST DERIVATION: Task list derivation basis
    - EXECUTION SUMMARY: Execution statistics
    - RESULTS SUMMARY: Results summary
    - KEY FINDINGS: Key findings
    - SCIENTIFIC RATIONALE: Scientific rationale
    - LIMITATIONS: Limitations
    - VALIDATION RECOMMENDATIONS: Validation recommendations
    - FILES GENERATED: Generated files
    - CONCLUSION: Conclusion

    Args:
        state: Result Evaluator state
        llm: LLM instance

    Returns:
        TXT format analysis report
    """
    # Build task list text
    task_list_text = ""
    for i, task in enumerate(state.all_tasks, 1):
        status_str = (
            "[OK]"
            if task.status.upper() == "COMPLETED"
            else "[FAIL]"
            if task.status.upper() == "FAILED"
            else "[PENDING]"
        )
        task_list_text += f"\n{i}. [{status_str}] {task.task_id}: {task.task_type}\n"
        task_list_text += f"   Content: {_truncate_text(task.content, 150)}\n"
        if task.error:
            task_list_text += f"   Error: {_truncate_text(task.error, 100)}\n"

    # Build Deep Research text
    deep_research_text = ""
    if state.deep_research and state.deep_research.research_summary:
        deep_research_text = f"\n{state.deep_research.research_summary}\n"
        if state.deep_research.key_insights:
            deep_research_text += "\nKey Insights:\n"
            for insight in state.deep_research.key_insights[:5]:
                deep_research_text += f"- {insight}\n"
        if state.deep_research.evidence:
            deep_research_text += "\nEvidence:\n"
            for evidence in state.deep_research.evidence[:5]:
                deep_research_text += f"- {evidence}\n"

    # Build Hypothesis text
    hypothesis_text = ""
    if state.hypothesis and state.hypothesis.hypothesis_summary:
        hypothesis_text = f"\n{state.hypothesis.hypothesis_summary}\n"
        if state.hypothesis.testable_predictions:
            hypothesis_text += "\nTestable Predictions:\n"
            for pred in state.hypothesis.testable_predictions[:5]:
                hypothesis_text += f"- {pred}\n"

    # Build Task List derivation text
    derivation_text = ""
    if state.task_list_derivation and state.task_list_derivation.decomposition_summary:
        derivation_text = f"\n{state.task_list_derivation.decomposition_summary}\n\nRequired Services:\n"
        for service in state.task_list_derivation.required_services[:10]:
            derivation_text += f"- {service}\n"
        derivation_text += f"\nDependency Rationale:\n{state.task_list_derivation.dependency_rationale}\n"
        if state.task_list_derivation.parallel_groups:
            derivation_text += "\nParallel Execution Groups:\n"
            for (
                group_id,
                group_info,
            ) in state.task_list_derivation.parallel_groups.items():
                derivation_text += (
                    f"- {group_id}: {group_info.get('subtask_count', 0)} subtasks\n"
                )

    # Build output files text
    output_files_text = ""
    if state.output_files:
        output_files_text = "\nOutput Files:\n"
        for f in state.output_files[:20]:
            output_files_text += f"  - {f}\n"
        if len(state.output_files) > 20:
            output_files_text += (
                f"  ... and {len(state.output_files) - 20} more files\n"
            )

    # Build tool output summary text with actual content
    tool_outputs_summary = _analyze_tool_outputs_for_report(state.tool_output_summaries)

    # Build detailed file content section for LLM analysis
    file_content_for_analysis = ""
    if state.tool_output_summaries:
        file_content_lines = []
        for i, f in enumerate(state.tool_output_summaries[:10], 1):
            file_name = Path(f.file_path).name
            file_content_lines.append(
                f"\n--- File {i}: {file_name} ({f.file_type}) ---"
            )
            if f.row_count is not None:
                file_content_lines.append(f"Rows: {f.row_count}")
            if f.columns:
                file_content_lines.append(f"Columns: {', '.join(f.columns[:8])}")
            if f.key_results:
                file_content_lines.append("Key data:")
                for kr in f.key_results[:4]:
                    file_content_lines.append(f"  {kr}")
            if f.content_summary:
                file_content_lines.append(f"Summary: {f.content_summary}")
        file_content_for_analysis = "\n".join(file_content_lines)

    # Use LLM to generate deep analysis (methodology, scientific rationale, limitations, etc.)
    methodology = ""
    scientific_rationale = ""
    limitations = []
    validation_recommendations = []

    if llm:
        try:
            from langchain_core.messages import HumanMessage

            user_question = state.user_input[:2000]

            analysis_prompt = f"""
You are a professional biomedical research analyst. Generate a professional analysis report based on the following computational analysis and ACTUAL FILE CONTENTS.

## Original Research Question
{user_question}

## Execution Plan Summary
{state.execution_plan[:1500] if state.execution_plan else "None"}

## Deep Research Findings
{deep_research_text[:1000] if deep_research_text else "None"}

## Hypothesis
{hypothesis_text[:1000] if hypothesis_text else "None"}

## Execution Statistics
- Total tasks: {state.total_tasks}
- Completed: {state.completed_tasks}
- Failed: {state.failed_tasks}
- Success rate: {state.success_rate:.1f}%

## Task Summary
{task_list_text[:2000]}

## Output File Contents (ACTUAL DATA)
{file_content_for_analysis[:3000]}

## Key Findings from Analysis
{chr(10).join([f"- {f}" for f in state.key_findings[:8]]) if state.key_findings else "None"}

Based on the actual file contents and data above, return in JSON format:
{{
    "methodology": "2-3 sentences describing the computational methods used, referencing specific tools/data types from the output files",
    "scientific_rationale": "2-3 sentences on the scientific basis, citing specific patterns or values found in the data",
    "limitations": ["Specific limitation 1 based on data quality/completeness", "Limitation 2", "Limitation 3"],
    "validation_recommendations": ["Specific experimental validation 1", "Validation 2", "Validation 3"]
}}

Write in professional academic English. Reference actual data values where relevant.
"""

            response = llm.invoke([HumanMessage(content=analysis_prompt)])
            response_content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Parse JSON
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
            print(f"[WARN] Failed to generate deep analysis: {e}")
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

    # Update state
    state.methodology = methodology
    state.scientific_rationale = scientific_rationale
    state.limitations = limitations
    state.validation_recommendations = validation_recommendations

    # Generate complete TXT report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build key findings text
    key_findings_text = ""
    for i, finding in enumerate(state.key_findings, 1):
        key_findings_text += f"{i}. {finding}\n"

    # Build limitations text
    limitations_text = ""
    for i, limitation in enumerate(limitations, 1):
        limitations_text += f"{i}. {limitation}\n"

    # Build validation recommendations text
    validation_text = ""
    for i, rec in enumerate(validation_recommendations, 1):
        validation_text += f"{i}. {rec}\n"

    # Build recommendations text
    recommendations_text = ""
    for i, rec in enumerate(state.recommendations, 1):
        recommendations_text += f"{i}. {rec}\n"

    # Build file list text
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

    # ========== Generate report title ==========
    # Extract key theme from user question
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

    # ========== Generate complete report ==========
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
    if state.error_summary and state.error_summary != "No errors":
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
    """Extract output file paths from task result"""
    output_files = []

    if task_result is None:
        return output_files

    # Try to extract file paths from output
    output = None
    if hasattr(task_result, "output"):
        output = task_result.output
    elif isinstance(task_result, dict):
        output = task_result.get("output")

    if output is None:
        return output_files

    # If output is a string, try to parse JSON
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            # If not JSON, check if it's a file path
            if "/" in output or "\\" in output:
                output_files.append(output)
            return output_files

    if not isinstance(output, dict):
        return output_files

    # Common file path field names
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

    # Check files in final_result
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
    """Truncate text"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def _analyze_csv_content(content: str, max_rows: int = 10000) -> Dict[str, Any]:
    """
    Analyze CSV/TSV content and extract structured summary.

    Args:
        content: Full CSV content string
        max_rows: Maximum rows to analyze

    Returns:
        Dictionary with columns, row_count, statistics, and sample data
    """
    import io

    result = {
        "columns": [],
        "row_count": 0,
        "statistics": {},
        "sample_data": [],
    }

    try:
        lines = content.strip().split("\n")
        if not lines:
            return result

        header_line = lines[0]
        delimiter = "\t" if "\t" in header_line else ","
        columns = [col.strip().strip('"') for col in header_line.split(delimiter)]
        result["columns"] = columns

        data_lines = lines[1 : max_rows + 1] if len(lines) > 1 else []
        result["row_count"] = len(data_lines)
        total_rows = len(lines) - 1

        if data_lines:
            for i, line in enumerate(data_lines[:3]):
                values = [v.strip().strip('"') for v in line.split(delimiter)]
                sample = (
                    dict(zip(columns, values)) if len(values) == len(columns) else {}
                )
                if sample:
                    result["sample_data"].append(sample)

            numeric_cols = {}
            for col_idx, col in enumerate(columns):
                numeric_values = []
                for line in data_lines[:1000]:
                    values = line.split(delimiter)
                    if col_idx < len(values):
                        try:
                            val = values[col_idx].strip().strip('"')
                            numeric_values.append(float(val))
                        except (ValueError, IndexError):
                            pass
                if numeric_values:
                    numeric_cols[col] = {
                        "min": min(numeric_values),
                        "max": max(numeric_values),
                        "mean": sum(numeric_values) / len(numeric_values),
                        "count": len(numeric_values),
                    }
            result["statistics"] = numeric_cols

        if total_rows > max_rows:
            result["statistics"]["_truncated"] = (
                f"Analyzed {max_rows} of {total_rows} total rows"
            )

    except Exception as e:
        result["error"] = str(e)

    return result


def _summarize_file_content_with_llm(
    llm: Any,
    file_path: str,
    file_type: str,
    content: str,
    user_question: str,
    max_summary_length: int = 500,
) -> str:
    """
    Use LLM to summarize file content intelligently.

    Args:
        llm: LLM instance
        file_path: File path
        file_type: File type description
        content: File content (may be truncated)
        user_question: Original user question for context
        max_summary_length: Maximum summary length

    Returns:
        Summary string
    """
    if not llm or not content:
        return ""

    try:
        from langchain_core.messages import HumanMessage

        file_name = Path(file_path).name

        prompt = f"""Analyze this {file_type} file and provide a concise summary focused on insights relevant to this research question:

Research Question: {user_question[:500]}

File: {file_name}
Content (may be truncated):
```
{content[:3000]}
```

Provide a summary (under {max_summary_length} characters) that:
1. Describes what data/results this file contains
2. Highlights key findings or patterns
3. Notes any limitations (e.g., if data appears incomplete)

Be specific and factual. Do not speculate beyond what's in the data."""

        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content if hasattr(response, "content") else str(response)
        return summary[:max_summary_length]

    except Exception as e:
        print(f"  [FileSummarizer] LLM summarization failed for {file_path}: {e}")
        return f"[Summary unavailable: {e}]"


def _collect_output_files_from_sandbox(
    sandbox_dir: str,
    session_id: Optional[str] = None,
    parent_state: Any = None,
    llm: Optional[Any] = None,
    user_question: str = "",
) -> List[ToolOutputSummary]:
    """
    Collect all tool output files from sandbox output directory with intelligent summarization.

    Execute code through CodeAct subgraph following architecture principles:
    - Other subgraphs do not directly call OpenSandbox
    - All sandbox operations are executed through CodeAct
    - Files are summarized using LLM instead of truncated

    Sandbox directory structure:
    - sandbox_dir may be /data/sessions/{session_id} or container internal path /data/sessions/{session_id}
    - Output files are located in {sandbox_dir}/output/ directory

    Args:
        sandbox_dir: Sandbox directory path
        session_id: Session ID
        parent_state: Parent state, used to get opensandbox_id
        llm: LLM instance for summarization
        user_question: Original user question for context-aware summarization

    Returns:
        Tool output summary list with intelligent summaries
    """
    tool_outputs = []

    if not sandbox_dir:
        return tool_outputs

    existing_sandbox_id = None
    if parent_state:
        merged_result = getattr(parent_state, "merged_result", None) or {}
        existing_sandbox_id = merged_result.get("opensandbox_id")

    from utils.codeact_executor import execute_code_via_codeact, is_codeact_available

    if not is_codeact_available():
        print(
            f"  [OutputCollector] CodeAct/OpenSandbox not enabled, cannot read remote files"
        )
        return tool_outputs

    print(f"  [OutputCollector] Reading remote files via CodeAct...")
    print(
        f"  [OutputCollector] sandbox_dir={sandbox_dir}, sandbox_id={existing_sandbox_id}"
    )

    COLLECTOR_CODE_TEMPLATE = """
import os
import json
import csv
from pathlib import Path
from io import StringIO

sandbox_dir = "{sandbox_dir}"
output_dir = Path(sandbox_dir) / "output"

if not output_dir.exists():
    output_dir = Path(sandbox_dir)

results = []

if output_dir.exists():
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
    
    MAX_FILE_SIZE = 100000  # 100KB limit for full content reading
    
    for file_path in output_dir.rglob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext in supported_extensions:
                try:
                    rel_path = file_path.relative_to(output_dir)
                except ValueError:
                    rel_path = file_path.name
                
                file_size = file_path.stat().st_size
                content_preview = ""
                full_content = ""
                key_results = []
                columns = []
                row_count = None
                statistics = {{}}
                
                try:
                    if ext in ['.csv', '.tsv', '.txt', '.json', '.md', '.fasta', '.fa', '.airr']:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            if file_size <= MAX_FILE_SIZE:
                                full_content = f.read()
                                content_preview = full_content[:1000]
                            else:
                                content_preview = f.read(5000)
                                full_content = content_preview
                        
                        if ext in ['.csv', '.tsv', '.airr']:
                            lines = full_content.split('\\n') if full_content else content_preview.split('\\n')
                            if lines:
                                delimiter = '\\t' if ext in ['.tsv', '.airr'] or '\\t' in lines[0] else ','
                                header = [c.strip().strip('"') for c in lines[0].split(delimiter)]
                                columns = header
                                row_count = len([l for l in lines[1:] if l.strip()])
                                
                                key_results.append(f"Columns ({{len(columns)}}): {{', '.join(columns[:10])}}{{'...' if len(columns) > 10 else ''}}")
                                key_results.append(f"Total rows: {{row_count}}")
                                
                                numeric_stats = {{}}
                                for col_idx, col in enumerate(header[:20]):
                                    values = []
                                    for line in lines[1:min(101, len(lines))]:
                                        parts = line.split(delimiter)
                                        if col_idx < len(parts):
                                            try:
                                                values.append(float(parts[col_idx].strip().strip('"')))
                                            except:
                                                pass
                                    if values:
                                        numeric_stats[col] = {{"min": min(values), "max": max(values), "mean": round(sum(values)/len(values), 2)}}
                                if numeric_stats:
                                    statistics["numeric_columns"] = numeric_stats
                                    
                        elif ext == '.json':
                            try:
                                data = json.loads(full_content)
                                if isinstance(data, dict):
                                    key_results.append(f"Object with keys: {{list(data.keys())[:10]}}")
                                    statistics["type"] = "object"
                                    statistics["key_count"] = len(data.keys())
                                elif isinstance(data, list):
                                    key_results.append(f"Array with {{len(data)}} items")
                                    statistics["type"] = "array"
                                    statistics["length"] = len(data)
                                    if data and isinstance(data[0], dict):
                                        statistics["item_keys"] = list(data[0].keys())[:10]
                            except:
                                pass
                                
                except Exception as e:
                    content_preview = f"[Unable to read: {{e}}]"
                
                results.append({{
                    "file_path": str(file_path),
                    "file_type": supported_extensions[ext],
                    "content_preview": content_preview,
                    "full_content": full_content if file_size <= MAX_FILE_SIZE else "",
                    "key_results": key_results,
                    "file_size": file_size,
                    "columns": columns,
                    "row_count": row_count,
                    "statistics": statistics
                }})

print("__OUTPUT_FILES_JSON_START__")
print(json.dumps(results, ensure_ascii=False))
print("__OUTPUT_FILES_JSON_END__")
"""

    collector_code = COLLECTOR_CODE_TEMPLATE.format(sandbox_dir=sandbox_dir)

    try:
        result = execute_code_via_codeact(
            task_description=f"Collect all output files from sandbox {sandbox_dir}/output directory",
            code_template=collector_code,
            sandbox_id=existing_sandbox_id,
            timeout_seconds=120,
            keep_alive=True,
        )

        if not result.is_success():
            print(f"  [OutputCollector] Sandbox execution failed: {result.error}")
            return tool_outputs

        stdout = result.output
        import re

        json_match = re.search(
            r"__OUTPUT_FILES_JSON_START__\s*(.*?)\s*__OUTPUT_FILES_JSON_END__",
            stdout,
            re.DOTALL,
        )

        if json_match:
            files_data = json.loads(json_match.group(1))

            print(
                f"  [OutputCollector] Found {len(files_data)} files, summarizing with LLM..."
            )

            for i, file_data in enumerate(files_data):
                content_summary = ""
                full_content = file_data.get("full_content", "")
                file_size = file_data.get("file_size", 0)
                file_type = file_data.get("file_type", "Unknown")
                file_path = file_data.get("file_path", "")

                if llm and (file_size > 2000 or not full_content):
                    content_for_summary = (
                        full_content
                        if full_content
                        else file_data.get("content_preview", "")
                    )
                    if content_for_summary:
                        print(
                            f"    [{i + 1}/{len(files_data)}] Summarizing {Path(file_path).name}..."
                        )
                        content_summary = _summarize_file_content_with_llm(
                            llm=llm,
                            file_path=file_path,
                            file_type=file_type,
                            content=content_for_summary,
                            user_question=user_question,
                            max_summary_length=500,
                        )
                elif full_content and len(full_content) <= 500:
                    content_summary = full_content

                tool_outputs.append(
                    ToolOutputSummary(
                        file_path=file_path,
                        file_type=file_type,
                        content_preview=file_data.get("content_preview", ""),
                        key_results=file_data.get("key_results", []),
                        content_summary=content_summary,
                        file_size=file_size,
                        row_count=file_data.get("row_count"),
                        columns=file_data.get("columns", []),
                        statistics=file_data.get("statistics", {}),
                    )
                )

            summarized_count = sum(1 for t in tool_outputs if t.content_summary)
            print(
                f"  [OutputCollector] Collected {len(tool_outputs)} files, {summarized_count} with LLM summaries"
            )
        else:
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
                    f"  [OutputCollector] Found {len(tool_outputs)} output files via auto-parsing"
                )
            else:
                print(f"  [OutputCollector] No output file JSON results found")

    except Exception as e:
        print(f"  [OutputCollector] Failed to collect files: {e}")
        import traceback

        traceback.print_exc()

    return tool_outputs


def _analyze_tool_outputs_for_report(tool_outputs: List[ToolOutputSummary]) -> str:
    """
    Analyze tool outputs with actual content summaries, not just file names.

    Args:
        tool_outputs: Tool output summary list

    Returns:
        Text summary for report with actual file content analysis
    """
    if not tool_outputs:
        return "No tool output files generated."

    by_type: Dict[str, List[ToolOutputSummary]] = {}
    for output in tool_outputs:
        if output.file_type not in by_type:
            by_type[output.file_type] = []
        by_type[output.file_type].append(output)

    summary_lines = [f"Total output files: {len(tool_outputs)}", ""]

    for file_type, files in by_type.items():
        summary_lines.append(f"## {file_type} ({len(files)} files)")
        summary_lines.append("")

        for f in files[:10]:
            summary_lines.append(f"### {Path(f.file_path).name}")

            if f.row_count is not None:
                summary_lines.append(f"- Rows: {f.row_count}")
            if f.columns:
                cols_str = ", ".join(f.columns[:8])
                if len(f.columns) > 8:
                    cols_str += f" ... (+{len(f.columns) - 8} more)"
                summary_lines.append(f"- Columns: {cols_str}")

            if f.key_results:
                for kr in f.key_results[:5]:
                    summary_lines.append(f"- {kr}")

            if f.content_summary:
                summary_lines.append(f"- Summary: {f.content_summary}")
            elif f.content_preview and len(f.content_preview) > 50:
                preview = f.content_preview[:300].replace("\n", " ").strip()
                summary_lines.append(f"- Preview: {preview}...")

            summary_lines.append("")

        if len(files) > 10:
            summary_lines.append(f"... and {len(files) - 10} more {file_type} files")
            summary_lines.append("")

    return "\n".join(summary_lines)


# ===================== Node 1: Results Collection =====================


def collect_results_node(state: ResultEvaluatorState) -> ResultEvaluatorState:
    """
    Node 1: Results Collection

    Collect execution results and output files from all tasks
    Enhanced: Read all tool output files from remote sandbox output directory via OpenSandbox
    Enhanced: Use LLM to summarize large files instead of truncating
    """
    print("\n" + "=" * 60)
    print("[STAT] Phase 1: Collecting Execution Results")
    print("=" * 60)

    total = len(state.all_tasks)
    completed = sum(1 for t in state.all_tasks if t.status.upper() == "COMPLETED")
    failed = sum(1 for t in state.all_tasks if t.status.upper() == "FAILED")

    state.total_tasks = total
    state.completed_tasks = completed
    state.failed_tasks = failed
    state.success_rate = (completed / total * 100) if total > 0 else 0.0

    all_output_files = []
    for task in state.all_tasks:
        task_files = _extract_output_files(task)
        all_output_files.extend(task_files)

    state.output_files = list(set(all_output_files))

    print(
        f"  [OutputCollector] Collecting tool output files from remote sandbox directory..."
    )

    sandbox_dir = None
    if state.parent_state:
        sandbox_dir = getattr(state.parent_state, "sandbox_data_dir", None)
    if not sandbox_dir:
        sandbox_dir = state.sandbox_dir

    llm = state.get_llm(
        purpose="bioinformatics", node_name="result_evaluator_collector"
    )

    tool_outputs = _collect_output_files_from_sandbox(
        sandbox_dir=sandbox_dir,
        session_id=state.session_id,
        parent_state=state.parent_state,
        llm=llm,
        user_question=state.user_input,
    )
    state.tool_output_summaries = tool_outputs

    for tool_output in tool_outputs:
        if tool_output.file_path not in state.output_files:
            state.output_files.append(tool_output.file_path)

    print(f"  [OutputCollector] Collected {len(tool_outputs)} tool output files")

    errors = []
    for task in state.all_tasks:
        if task.error:
            errors.append(f"- Task {task.task_id}: {_truncate_text(task.error, 200)}")

    if errors:
        state.error_summary = "\n".join(errors[:10])
        if len(errors) > 10:
            state.error_summary += f"\n... and {len(errors) - 10} more errors"
    else:
        state.error_summary = "No errors"

    summarized_count = sum(1 for t in tool_outputs if t.content_summary)
    print(f"[SUCCESS] Results collection completed")
    print(f"  - Total tasks: {total}")
    print(f"  - Completed: {completed}")
    print(f"  - Failed: {failed}")
    print(f"  - Success rate: {state.success_rate:.1f}%")
    print(f"  - Output files count: {len(state.output_files)}")
    print(f"  - Files with LLM summaries: {summarized_count}")

    return state


# ===================== Node 2: Results Analysis =====================


def analyze_results_node(state: ResultEvaluatorState) -> ResultEvaluatorState:
    """
    Node 2: Results Analysis

    Use LLM to analyze execution results with actual file content.
    """
    print("\n" + "=" * 60)
    print("[ANALYSIS] Phase 2: Analyzing Execution Results")
    print("=" * 60)

    llm = state.get_llm(purpose="bioinformatics", node_name="result_evaluator")
    if not llm:
        print("[WARN] LLM not available, using simple analysis")
        state.key_findings = ["LLM not available, unable to perform deep analysis"]
        state.recommendations = ["Please check LLM configuration"]
        return state

    try:
        from langchain_core.messages import HumanMessage

        task_summaries = []
        for task in state.all_tasks[:15]:
            task_summary = f"""
Task: {task.task_id} ({task.task_type})
Status: {task.status}
Content: {_truncate_text(task.content, 150)}
"""
            if task.error:
                task_summary += f"Error: {_truncate_text(task.error, 150)}\n"
            task_summaries.append(task_summary)

        tasks_text = "\n".join(task_summaries)

        files_content_section = ""
        if state.tool_output_summaries:
            files_content_lines = ["## Output File Contents Analysis\n"]

            for i, f in enumerate(state.tool_output_summaries[:15]):
                file_name = Path(f.file_path).name
                files_content_lines.append(f"\n### File {i + 1}: {file_name}")
                files_content_lines.append(f"Type: {f.file_type}")

                if f.row_count is not None:
                    files_content_lines.append(f"Rows: {f.row_count}")
                if f.columns:
                    cols_preview = ", ".join(f.columns[:6])
                    if len(f.columns) > 6:
                        cols_preview += f" (+{len(f.columns) - 6} more)"
                    files_content_lines.append(f"Columns: {cols_preview}")

                if f.key_results:
                    files_content_lines.append("Key Results:")
                    for kr in f.key_results[:4]:
                        files_content_lines.append(f"  - {kr}")

                if f.content_summary:
                    files_content_lines.append(f"Content Summary: {f.content_summary}")
                elif f.content_preview and len(f.content_preview) > 30:
                    preview = f.content_preview[:200].replace("\n", " ").strip()
                    files_content_lines.append(f"Preview: {preview}")

            if len(state.tool_output_summaries) > 15:
                files_content_lines.append(
                    f"\n... and {len(state.tool_output_summaries) - 15} more files"
                )

            files_content_section = "\n".join(files_content_lines)

        analysis_prompt = f"""You are a professional biomedical data analyst. Analyze the following task execution results and output files to extract meaningful scientific findings.

## Original Research Question
{state.user_input[:1500]}

## Execution Plan
{state.execution_plan[:1500] if state.execution_plan else "No execution plan provided"}

## Execution Statistics
- Total tasks: {state.total_tasks}
- Completed: {state.completed_tasks}
- Failed: {state.failed_tasks}
- Success rate: {state.success_rate:.1f}%

## Task Summary
{tasks_text}

{files_content_section}

## Errors (if any)
{state.error_summary if state.error_summary != "No errors" else "No errors occurred"}

---

Based on the actual file contents and task results above, provide:

1. **Key Findings** (3-5 items): What are the main results or discoveries? Be specific about data patterns, scores, or values found in the output files. DO NOT make up information not present in the data.

2. **Recommendations** (2-3 items): What would you suggest for next steps or improvements?

Return in JSON format:
{{
    "key_findings": ["specific finding 1 with data details", "finding 2", ...],
    "recommendations": ["recommendation 1", "recommendation 2", ...]
}}
"""

        messages = [HumanMessage(content=analysis_prompt)]

        response = llm.invoke(messages)
        response_content = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Use robust JSON extraction utility
        from utils.json_extractor import extract_json_from_llm_response

        analysis_result = extract_json_from_llm_response(
            response_content,
            default={"key_findings": [], "recommendations": []},
            log_errors=True,
        )

        state.key_findings = analysis_result.get(
            "key_findings",
            ["Task execution completed. See detailed output files for results."],
        )
        state.recommendations = analysis_result.get(
            "recommendations", ["Review output files for detailed analysis"]
        )

        print(f"[SUCCESS] Results analysis completed")
        print(f"  - Key findings count: {len(state.key_findings)}")
        print(f"  - Recommendations count: {len(state.recommendations)}")

    except Exception as e:
        print(f"[WARN] Results analysis failed: {e}")
        import traceback

        traceback.print_exc()
        state.key_findings = [f"Analysis process error: {str(e)}"]
        state.recommendations = ["Please check system configuration"]

    return state


# ===================== Node 3: Report Generation =====================


def generate_report_node(state: ResultEvaluatorState) -> ResultEvaluatorState:
    """
    Node 3: Report Generation

    Generate final summary report
    """
    print("\n" + "=" * 60)
    print("[REPORT] Phase 3: Generating Summary Report")
    print("=" * 60)

    llm = state.get_llm(purpose="bioinformatics", node_name="result_evaluator")

    try:
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
            task_results_text += f"   - Type: {task.task_type}\n"
            task_results_text += f"   - Content: {_truncate_text(task.content, 100)}\n"
            if task.error:
                task_results_text += f"   - Error: {_truncate_text(task.error, 100)}\n"

        findings_text = "\n".join([f"- {f}" for f in state.key_findings])

        recommendations_text = "\n".join([f"- {r}" for r in state.recommendations])

        files_text = ""
        if state.output_files:
            files_text = "\n### Output Files\n\n" + "\n".join(
                [f"- `{f}`" for f in state.output_files]
            )

        summary_text = ""
        if llm:
            try:
                from langchain_core.messages import HumanMessage

                summary_prompt = f"""
Please generate a concise summary paragraph (under 200 words) for the following task execution results:

User Question: {state.user_input[:500]}

Execution Statistics:
- Total tasks: {state.total_tasks}
- Completed: {state.completed_tasks}
- Failed: {state.failed_tasks}
- Success rate: {state.success_rate:.1f}%

Key Findings:
{findings_text}

Please output the summary paragraph directly without any additional content.
"""

                response = llm.invoke([HumanMessage(content=summary_prompt)])
                summary_text = (
                    response.content if hasattr(response, "content") else str(response)
                )

            except Exception as e:
                print(f"[WARN] Failed to generate summary: {e}")
                summary_text = f"Task execution completed with {state.success_rate:.1f}% success rate. Completed {state.completed_tasks}/{state.total_tasks} tasks."

        detailed_report = f"""# Task Execution Summary Report

**Generated at**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Session ID**: {state.session_id or "N/A"}

---

## Summary

{summary_text}

---

## Execution Statistics

| Metric | Value |
|--------|-------|
| Total tasks | {state.total_tasks} |
| Completed | {state.completed_tasks} |
| Failed | {state.failed_tasks} |
| Success rate | {state.success_rate:.1f}% |

---

## Original User Input

```
{state.user_input[:1000]}
```

---

## Execution Plan

```
{state.execution_plan[:2000] if state.execution_plan else "No execution plan"}
```

---

## Task Execution Details

{task_results_text}

---

## Key Findings

{findings_text}

---

## Recommendations

{recommendations_text}

{files_text}

---

## Error Summary

```
{state.error_summary}
```

---

*Report generated by Bio-Agent*
"""

        state.detailed_report = detailed_report
        state.summary_report = summary_text

        # Get opensandbox_id for saving reports
        opensandbox_id = None
        if state.parent_state:
            merged_result = getattr(state.parent_state, "merged_result", None) or {}
            opensandbox_id = merged_result.get("opensandbox_id")

        remote_report_path = ""
        remote_txt_report_path = ""

        if state.session_id and state.sandbox_dir:
            print(f"  [ReportSave] Saving reports to OpenSandbox...")
            print(
                f"  [ReportSave] opensandbox_id: {opensandbox_id or 'will create new'}"
            )
            print(f"  [ReportSave] sandbox_dir: {state.sandbox_dir}")

            remote_report_path = _save_report_to_opensandbox(
                content=detailed_report,
                report_type="result_evaluation",
                opensandbox_id=opensandbox_id,
                sandbox_dir=state.sandbox_dir,
            )
            state.report_path = remote_report_path

            txt_report = _generate_txt_analysis_report(state, llm)
            state.txt_report = txt_report

            remote_txt_report_path = _save_report_to_opensandbox(
                content=txt_report,
                report_type="analysis_report",
                opensandbox_id=opensandbox_id,
                sandbox_dir=state.sandbox_dir,
            )
            state.txt_report_path = remote_txt_report_path

            if remote_report_path and remote_report_path not in state.output_files:
                state.output_files.append(remote_report_path)
            if (
                remote_txt_report_path
                and remote_txt_report_path not in state.output_files
            ):
                state.output_files.append(remote_txt_report_path)
        else:
            print(
                f"  [ReportSave] ⚠️ Missing session_id or sandbox_dir, reports not saved to remote"
            )
            print(
                f"  [ReportSave] session_id: {state.session_id}, sandbox_dir: {state.sandbox_dir}"
            )

            txt_report = _generate_txt_analysis_report(state, llm)
            state.txt_report = txt_report

        print(f"[SUCCESS] Report generation completed")
        if remote_report_path:
            print(f"  - MD report saved to: {remote_report_path}")
        if remote_txt_report_path:
            print(f"  - TXT report saved to: {remote_txt_report_path}")

    except Exception as e:
        print(f"[WARN] Report generation failed: {e}")
        import traceback

        traceback.print_exc()
        state.detailed_report = f"Report generation failed: {str(e)}"
        state.summary_report = "Unable to generate report"

    return state


# ===================== Input/Output Mapping =====================


def result_evaluator_input_mapper(global_state: GlobalState) -> ResultEvaluatorState:
    """
    Map main graph state to Result Evaluator subgraph state

    Enhanced features:
    - Collect complete analysis pipeline information (deep research, hypothesis, execution plan, task list)
    - Collect task list derivation basis
    - Prioritize using sandbox_data_dir (session directory) to read tool outputs

    Args:
        global_state: Main graph global state

    Returns:
        Result Evaluator subgraph state
    """
    from .state import DeepResearchInfo, HypothesisInfo, TaskListDerivation

    # Collect task results from global_state
    task_results = {}
    all_tasks = []

    # Get task execution status from executor results
    executor_results = global_state.merged_result.get("executor_results", {})
    task_results_dict = executor_results.get("task_results", {})

    # Get all tasks (from subtasks and parallel_task_groups)
    all_subtasks = list(global_state.subtasks)
    seen_task_ids = {task.task_id for task in all_subtasks}

    for group in global_state.parallel_task_groups.values():
        if hasattr(group, "subtasks"):
            for task in group.subtasks:
                if task.task_id not in seen_task_ids:
                    all_subtasks.append(task)
                    seen_task_ids.add(task.task_id)

    # Build task summaries
    for task in all_subtasks:
        task_id = task.task_id

        # Get execution result
        exec_result = task_results_dict.get(task_id, {})

        # Extract status
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

        # Extract error
        error = None
        if exec_result:
            if hasattr(exec_result, "error"):
                error = exec_result.error
            elif isinstance(exec_result, dict):
                error = exec_result.get("error")

        # Extract output
        output = None
        if exec_result:
            if hasattr(exec_result, "output"):
                output = exec_result.output
            elif isinstance(exec_result, dict):
                output = exec_result.get("output")

        # Extract execution time
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
            output_files=[],  # Will be populated in collect_results_node
            execution_time=execution_time,
        )

        task_results[task_id] = task_summary
        all_tasks.append(task_summary)

    # Get execution plan
    execution_plan = global_state.execution_plan or ""

    # Prefer sandbox_data_dir (session directory), fallback to sandbox_dir
    # sandbox_data_dir format: /data/sessions/{session_id}
    sandbox_dir = global_state.sandbox_data_dir or global_state.sandbox_dir or ""

    # ========== Collect Immunity subgraph information ==========
    immunity_plan = global_state.merged_result.get("immunity_plan", {})

    # Collect Deep Research information (enhanced: collect more complete information)
    deep_research = DeepResearchInfo(
        research_summary=immunity_plan.get("research_summary", ""),
        key_insights=immunity_plan.get("research_insights", []),
        evidence=immunity_plan.get("research_evidence", []),
        knowledge_gaps=immunity_plan.get("research_gaps", []),
        confidence=immunity_plan.get("research_confidence", 0.0),
    )

    # Collect Hypothesis information (enhanced: collect more complete information)
    hypothesis = HypothesisInfo(
        hypothesis_summary=immunity_plan.get("hypothesis_summary", ""),
        testable_predictions=immunity_plan.get("testable_predictions", []),
        confidence=immunity_plan.get("hypothesis_confidence", 0.0),
    )

    # Collect Task List derivation basis (enhanced: collect more complete decomposition basis)
    task_decomp_results = global_state.merged_result.get("task_decomposition", {})
    required_service_ids = task_decomp_results.get("required_service_ids", [])
    raw_tasks = task_decomp_results.get("raw_tasks", [])

    # Build more detailed task decomposition summary
    decomposition_details = []
    if required_service_ids:
        decomposition_details.append(
            f"Required services: {', '.join(required_service_ids[:10])}"
        )
    if raw_tasks:
        decomposition_details.append(f"Initial task count: {len(raw_tasks)}")

    # Collect parallel group information
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
        decomposition_summary=f"Decomposed into {len(all_subtasks)} subtasks based on execution plan."
        + (" ".join(decomposition_details) if decomposition_details else ""),
        required_services=required_service_ids
        if isinstance(required_service_ids, list)
        else [],
        dependency_rationale="Tasks are executed in order based on data flow and dependencies, ensuring upstream task outputs are available as downstream task inputs.",
        parallel_groups=parallel_group_info,
    )

    return ResultEvaluatorState(
        # [FIX] Do NOT pass progress_callback - it cannot be serialized by LangGraph.
        # The callback is retrieved dynamically from global registry via session_id in get_llm().
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
    Map Result Evaluator subgraph state back to main graph state

    Args:
        evaluator_state: Result Evaluator subgraph state (can be ResultEvaluatorState object or dict)
        global_state: Main graph global state

    Returns:
        Updated main graph global state
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

    # Store evaluation results
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

    # Store output file paths to dedicated list field (file_paths values must be str type)
    if output_files:
        # Deduplicate and add to completed_output_files
        existing = set(global_state.completed_output_files)
        for f in output_files:
            if f not in existing:
                global_state.completed_output_files.append(f)

    print(f"[SUCCESS] Result Evaluator subgraph completed")
    print(f"  - Summary report length: {len(summary_report)} characters")
    print(f"  - Detailed report length: {len(detailed_report)} characters")
    print(f"  - TXT report length: {len(txt_report)} characters")
    print(f"  - Output files count: {len(output_files)}")

    return global_state


# ===================== Build Result Evaluator Subgraph =====================


def build_result_evaluator_subgraph():
    """
    Build Result Evaluator subgraph

    Workflow:
    1. Results collection -> 2. Results analysis -> 3. Report generation

    Returns:
        Compiled subgraph
    """
    graph = StateGraph(ResultEvaluatorState)

    # Add all nodes
    graph.add_node("collect_results", collect_results_node)  # Phase 1
    graph.add_node("analyze_results", analyze_results_node)  # Phase 2
    graph.add_node("generate_report", generate_report_node)  # Phase 3

    # Define workflow
    graph.add_edge(START, "collect_results")
    graph.add_edge("collect_results", "analyze_results")
    graph.add_edge("analyze_results", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()
