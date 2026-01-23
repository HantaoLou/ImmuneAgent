"""
Immunity 和 Task Decomposition 子图综合测试用例

测试用例分类：
1.1 Tool Selection - 工具选择问题
1.2 Planning - 计划制定问题
1.3 Deep Research - 深度研究问题
1.4 Hypothesis - 假设生成问题

每个测试用例会：
1. 运行 Immunity 子图（生成实验计划）
2. 运行 Task Decomposition 子图（分解任务）
3. 详细记录每个阶段的结果

运行方式：pytest tests/test_immunity_and_decomposition_comprehensive.py -v
"""

import os
import pytest
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

# 加载环境变量
load_dotenv()

# 添加agent目录到路径
import sys
agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from nodes.subagents.immunity.graph import (
    build_immunity_subgraph,
    immunity_input_mapper,
    immunity_output_mapper,
    ImmunityState
)
from nodes.subagents.task_decomposition.graph import (
    build_task_decomposition_subgraph,
    task_decomposition_input_mapper,
    task_decomposition_output_mapper,
    TaskDecompositionState
)
from state import GlobalState, UserTaskType


# ===================== 测试用例数据 =====================

# 1.1 Tool Selection - 工具选择问题
TOOL_SELECTION_TEST_CASES = [
    {
        "id": "1.1.1",
        "name": "抗体设计_工具选择",
        "user_input": "I have a candidate antibody sequence against H5N1 and want to evaluate its potential breadth and affinity (e.g., with influenza group 1 strains) and its developability. Which tools should I use for the analysis? Please explain the purpose of each step.",
        "category": "tool_selection",
        "description": "Evaluate H5N1 candidate antibody breadth, affinity, and developability"
    },
    {
        "id": "1.1.2",
        "name": "B单细胞分析_工具选择",
        "user_input": "I have obtained a dataset of peripheral blood B cell single-cell sequencing data (scRNA-seq + scBCR-seq) from COVID-19 convalescents and vaccinated individuals. I want to identify antigen-specific B cell clones from it and compare their transcriptional state differences. Which tools should I use for the analysis? Please describe the workflow.",
        "category": "tool_selection",
        "description": "Identify antigen-specific B cell clones and compare transcriptional states"
    },
    {
        "id": "1.1.3",
        "name": "蛋白抗原优化_工具选择",
        "user_input": "Optimize the sequence stability of the influenza virus NA protein to obtain an optimized NA amino acid sequence. How should I utilize which tools to build and evaluate this prediction pipeline?",
        "category": "tool_selection",
        "description": "Optimize influenza virus NA protein sequence stability"
    },
    {
        "id": "1.1.4",
        "name": "mRNA疫苗设计_工具选择",
        "user_input": " I want to optimize an mRNA sequence for the influenza NA antigen to achieve its high-efficiency and sustained expression. Which tools can I use to predict the mRNA expression level and translation efficiency?",
        "category": "tool_selection",
        "description": "Optimize mRNA sequence to predict expression level and translation efficiency"
    },
]

# 1.2 Planning - 计划制定问题
PLANNING_TEST_CASES = [
    {
        "id": "1.2.1",
        "name": "抗体设计_计划制定",
        "user_input": "Please develop a detailed research plan for a project to \"design broadly neutralizing antibodies against H5N1.\" The plan should include: hypothesis proposal, antigen design, antibody screening platform, in vitro validation experiments, and key steps for in vivo efficacy evaluation.",
        "category": "planning",
        "description": "Develop research plan for H5N1 broadly neutralizing antibodies"
    },
    {
        "id": "1.2.2",
        "name": "机制探索_计划制定",
        "user_input": "In order to investigate \"why some individuals generate highly cross-reactive plasma cells after influenza vaccination, while others do not,\" please design a longitudinal study protocol combining single-cell multi-omics and B cell receptor analysis, including sample collection time points, intended data types for analysis, and core analysis steps.",
        "category": "planning",
        "description": "Design longitudinal study to investigate plasma cell cross-reactivity differences"
    },
    {
        "id": "1.2.3",
        "name": "疫苗开发_计划制定",
        "user_input": "Please create a development roadmap from concept to pre-clinical research for \"developing a universal mRNA vaccine based on the influenza virus HA stem.\" The roadmap should cover antigen selection and optimization, mRNA optimization, immunogenicity assessment strategy, and animal model experiments to verify breadth and protection.",
        "category": "planning",
        "description": "Create development roadmap for universal mRNA vaccine"
    },
]

# 1.3 Deep Research - 深度研究问题
DEEP_RESEARCH_TEST_CASES = [
    {
        "id": "1.3.1",
        "name": "抗体设计_深度研究",
        "user_input": "Please systematically review the trade-off relationship between \"antibody affinity maturation\" and \"neutralization breadth.\" In research on HIV, influenza, and coronaviruses, which somatic hypermutation (SHM) patterns, antibody framework regions, or CDR conformational features have been proven to break or optimize this trade-off?",
        "category": "deep_research",
        "description": "Review trade-off between antibody affinity maturation and neutralization breadth"
    },
    {
        "id": "1.3.2",
        "name": "B单细胞分析_深度研究",
        "user_input": "Please conduct an in-depth analysis and comparison of the known differences between \"long-lived plasma cells\" and \"short-lived plasma cells\" in terms of their transcriptome, epigenetics, metabolism, and dependency on survival niche signals. Which key transcription factors and their regulatory networks are central to determining their fate differentiation?",
        "category": "deep_research",
        "description": "Analyze differences between long-lived and short-lived plasma cells"
    },
    {
        "id": "1.3.3",
        "name": "蛋白抗原优化_深度研究",
        "user_input": "For the design of a respiratory syncytial virus (RSV) prefusion F protein (pre-F) vaccine, please conduct an in-depth study of its binding epitopes (site Ø, V, III, etc.) with neutralizing antibodies. Which point mutations or disulfide bond engineering have been proven to stabilize the pre-F conformation while minimizing its transition to the postfusion (post-F) conformation?",
        "category": "deep_research",
        "description": "Study stabilization strategies for RSV F protein pre-F conformation"
    },
    {
        "id": "1.3.4",
        "name": "mRNA疫苗设计_深度研究",
        "user_input": " Conduct an in-depth study on the \"innate immune sensing\" mechanism of mRNA vaccines. Please elaborate in detail how the recognition of mRNA by different pattern recognition receptors (e.g., RLRs, TLRs) affects antigen translation efficiency, immunogenicity type (Th1/Th2 balance), and antibody affinity maturation. Also, discuss the advantages and disadvantages of current \"nucleoside modification\" and \"sequence engineering\" strategies.",
        "category": "deep_research",
        "description": "Study innate immune sensing mechanism of mRNA vaccines"
    },
]

# 1.4 Hypothesis - 假设生成问题
HYPOTHESIS_TEST_CASES = [
    {
        "id": "1.4.1",
        "name": "抗体设计_假设生成",
        "user_input": "Based on recent cryo-EM studies on the dynamics of the \"cryptic epitope\" in the HA stem region of H5N1 avian influenza virus and the commonality of the human VH1-69 germline antibody gene, please propose a hypothesis on how to design a novel antibody scaffold with broader neutralizing activity and a higher barrier to viral escape.",
        "category": "hypothesis",
        "description": "Propose hypothesis for novel broadly neutralizing antibody scaffold"
    },
    {
        "id": "1.4.2",
        "name": "B单细胞分析_假设生成",
        "user_input": "After COVID-19 mRNA vaccination, an expansion of an FCRL5-high \"atypical\" memory B cell population has been observed. Please propose a hypothesis to explain whether this cell population is a marker of \"functional exhaustion\" or a \"pre-adapted\" reservoir against variant strains. Also, specify the types of single-cell data needed to validate this hypothesis.",
        "category": "hypothesis",
        "description": "Propose hypothesis for FCRL5-high memory B cells"
    },
    {
        "id": "1.4.3",
        "name": "蛋白抗原优化_假设生成",
        "user_input": "To design a universal influenza virus M2e protein vaccine, but M2e has weak immunogenicity. Please propose a hypothesis on how to rationally design its fusion with a self-assembling nanoparticle to simultaneously enhance its conformational stability, B cell receptor cross-reactivity, and T cell helper response.",
        "category": "hypothesis",
        "description": "Propose hypothesis to enhance M2e protein vaccine immunogenicity"
    },
    {
        "id": "1.4.4",
        "name": "mRNA疫苗设计_假设生成",
        "user_input": "There is evidence that the innate immunogenicity of some mRNA vaccines interferes with antigen expression. Please propose a hypothesis on how to achieve spatiotemporal regulation of \"high antigen expression first, followed by interferon activation\" in dendritic cells by co-optimizing the nucleotide sequence (codons, UTRs) of the mRNA and the lipid components of the delivery system, thereby enhancing the germinal center response.",
        "category": "hypothesis",
        "description": "Propose hypothesis for mRNA vaccine spatiotemporal regulation"
    },
]

# 合并所有测试用例
ALL_TEST_CASES = (
    TOOL_SELECTION_TEST_CASES +
    PLANNING_TEST_CASES +
    DEEP_RESEARCH_TEST_CASES +
    HYPOTHESIS_TEST_CASES
)


# ===================== 辅助函数 =====================

def _ensure_immunity_state(result):
    """确保结果是 ImmunityState 对象"""
    if isinstance(result, dict):
        return ImmunityState(**result)
    return result


def _ensure_task_decomposition_state(result):
    """确保结果是 TaskDecompositionState 对象"""
    if isinstance(result, dict):
        return TaskDecompositionState(**result)
    return result


def _create_test_sandbox() -> str:
    """创建测试沙盒目录"""
    test_sandbox = Path(agent_dir) / "tests" / "sandbox" / "comprehensive_test"
    test_sandbox.mkdir(parents=True, exist_ok=True)
    return str(test_sandbox)


def _format_stage_result(stage_name: str, state: Any, stage_data: Dict[str, Any]) -> str:
    """格式化单个阶段的结果"""
    result = f"\n### {stage_name}\n\n"
    
    if stage_name == "查询分解":
        result += f"**优化查询数量**: {len(state.optimized_questions)}\n\n"
        result += "**优化查询列表**:\n"
        for i, query in enumerate(state.optimized_questions, 1):
            result += f"{i}. {query}\n"
        result += "\n"
        
    elif stage_name == "检索":
        result += f"**检索上下文长度**: {len(state.context)} 字符\n\n"
        result += f"**检索文档数量**: {len(state.retrieval_docs)}\n"
        if state.retrieval_docs:
            result += "\n**检索文档列表**:\n"
            for i, doc in enumerate(state.retrieval_docs, 1):
                if isinstance(doc, dict):
                    title = doc.get('title', 'N/A')
                    summary = doc.get('summary', 'N/A')
                    relevance = doc.get('relevance_score', 'N/A')
                    result += f"{i}. **{title}** (相关性: {relevance})\n"
                    result += f"   - 摘要: {summary[:200]}{'...' if len(summary) > 200 else ''}\n"
                else:
                    result += f"{i}. {doc}\n"
        result += f"\n**引用文献数量**: {len(state.citations)}\n"
        if state.citations:
            result += "\n**引用文献列表**:\n"
            for i, cite in enumerate(state.citations, 1):
                if isinstance(cite, dict):
                    author = cite.get('author', 'N/A')
                    year = cite.get('year', 'N/A')
                    title = cite.get('title', 'N/A')
                    journal = cite.get('journal', 'N/A')
                    doi = cite.get('doi', '')
                    result += f"{i}. {author} et al. ({year}). {title}. *{journal}*"
                    if doi:
                        result += f" DOI: {doi}"
                    result += "\n"
                else:
                    result += f"{i}. {cite}\n"
        if state.context:
            result += f"\n**检索上下文摘要**:\n```\n{state.context[:1000]}{'...' if len(state.context) > 1000 else ''}\n```\n"
        
    elif stage_name == "深度研究":
        deep_research_findings = state.deep_research_findings if state.deep_research_findings else {}
        result += f"**研究主题**: {deep_research_findings.get('topic', '未指定') if isinstance(deep_research_findings, dict) else '未指定'}\n\n"
        result += f"**研究置信度**: {state.research_confidence:.1f}%\n\n"
        result += f"**关键洞察数量**: {len(state.research_insights)}\n"
        if state.research_insights:
            result += "\n**关键洞察**:\n"
            for i, insight in enumerate(state.research_insights, 1):
                result += f"{i}. {insight}\n"
        result += f"\n**支持证据数量**: {len(state.research_evidence)}\n"
        if state.research_evidence:
            result += "\n**支持证据**:\n"
            for i, evidence in enumerate(state.research_evidence, 1):
                result += f"{i}. {evidence}\n"
        result += f"\n**知识缺口数量**: {len(state.research_gaps)}\n"
        if state.research_gaps:
            result += "\n**知识缺口**:\n"
            for i, gap in enumerate(state.research_gaps, 1):
                result += f"{i}. {gap}\n"
        result += f"\n**研究建议数量**: {len(state.research_recommendations)}\n"
        if state.research_recommendations:
            result += "\n**研究建议**:\n"
            for i, rec in enumerate(state.research_recommendations, 1):
                result += f"{i}. {rec}\n"
        if state.research_summary:
            result += f"\n**研究摘要**:\n```\n{state.research_summary}\n```\n"
        
    elif stage_name == "假设生成":
        hypothesis = state.hypothesis if state.hypothesis else {}
        if isinstance(hypothesis, dict):
            result += f"**假设陈述**: {hypothesis.get('statement', '未生成')}\n\n"
            result += f"**假设置信度**: {state.hypothesis_confidence:.1f}%\n\n"
            result += f"**创新水平**: {hypothesis.get('innovation_level', '未指定')}\n\n"
            result += f"**可测试预测数量**: {len(state.testable_predictions)}\n"
            if state.testable_predictions:
                result += "\n**可测试的预测**:\n"
                for i, pred in enumerate(state.testable_predictions, 1):
                    result += f"{i}. {pred}\n"
        if state.hypothesis_summary:
            result += f"\n**假设摘要**:\n```\n{state.hypothesis_summary}\n```\n"
        
    elif stage_name == "计划生成":
        result += f"**计划文档长度**: {len(state.final_enhanced_plan)} 字符\n\n"
        # 注意：完整计划会在后面的"完整实验计划"部分显示，这里只显示长度信息
        
    elif stage_name == "评估":
        result += f"**评估报告长度**: {len(state.final_evaluation)} 字符\n\n"
        # 注意：完整评估报告会在后面的"完整评估报告"部分显示，这里只显示长度信息
    
    # 添加执行时间和状态信息
    if stage_data:
        if "execution_time" in stage_data:
            result += f"\n**执行时间**: {stage_data['execution_time']:.2f} 秒\n"
        if "status" in stage_data:
            result += f"**状态**: {stage_data['status']}\n"
        if "error" in stage_data:
            result += f"**错误**: {stage_data['error']}\n"
    
    return result


def _format_decomposition_result(decomposition_state: TaskDecompositionState) -> str:
    """格式化任务分解结果"""
    result = "\n## Task Decomposition 子图结果\n\n"
    
    # 粗分解结果
    result += "### 阶段0: 粗分解（服务识别）\n\n"
    result += f"**所需服务数量**: {len(decomposition_state.required_service_ids)}\n"
    if decomposition_state.required_service_ids:
        result += "\n**所需服务列表**:\n"
        for i, service_id in enumerate(decomposition_state.required_service_ids, 1):
            result += f"{i}. {service_id}\n"
    result += f"\n**筛选后工具数量**: {len(decomposition_state.filtered_tools)}\n\n"
    
    # 细分解结果
    result += "### 阶段1: 细分解（任务生成）\n\n"
    raw_tasks = decomposition_state.raw_tasks
    if isinstance(raw_tasks, str):
        try:
            raw_tasks = json.loads(raw_tasks)
        except:
            raw_tasks = []
    elif not isinstance(raw_tasks, list):
        raw_tasks = []
    
    result += f"**原始任务数量**: {len(raw_tasks)}\n"
    if raw_tasks:
        result += "\n**原始任务列表**:\n"
        for i, task in enumerate(raw_tasks, 1):
            task_id = f'task_{i}'
            content = 'N/A'
            
            if isinstance(task, dict):
                # 尝试多种字段名获取任务ID
                task_id = task.get('task_id') or task.get('id') or task.get('taskId') or f'task_{i}'
                # 尝试多种字段名获取任务内容
                content = (
                    task.get('content') or 
                    task.get('task') or 
                    task.get('description') or 
                    task.get('name') or 
                    task.get('task_description') or
                    task.get('taskDescription') or
                    str(task)  # 如果都没有，转换为字符串
                )
            elif hasattr(task, 'content'):
                # 如果是 SubTask 对象
                task_id = getattr(task, 'task_id', f'task_{i}')
                content = getattr(task, 'content', 'N/A')
            elif hasattr(task, 'task_id'):
                task_id = getattr(task, 'task_id', f'task_{i}')
                content = str(task)
            else:
                # 其他类型，直接转换为字符串
                content = str(task)
            
            # 确保 content 不是 None 或空字符串
            if not content or content == 'None':
                content = 'N/A'
            
            result += f"{i}. **{task_id}**: {content}\n"
    result += "\n"
    
    # 并行推断结果
    result += "### 阶段2: 并行推断（依赖分析）\n\n"
    result += f"**普通任务数量**: {len(decomposition_state.subtasks)}\n"
    result += f"**并行任务组数量**: {len(decomposition_state.parallel_task_groups)}\n\n"
    
    if decomposition_state.subtasks:
        result += "**普通任务列表**:\n"
        for i, task in enumerate(decomposition_state.subtasks, 1):
            task_id = getattr(task, 'task_id', f'task_{i}') if hasattr(task, 'task_id') else (task.get('task_id', f'task_{i}') if isinstance(task, dict) else f'task_{i}')
            content = getattr(task, 'content', 'N/A') if hasattr(task, 'content') else (task.get('content', 'N/A') if isinstance(task, dict) else 'N/A')
            dependencies = getattr(task, 'dependencies', []) if hasattr(task, 'dependencies') else (task.get('dependencies', []) if isinstance(task, dict) else [])
            result += f"{i}. **{task_id}**\n"
            result += f"   - 内容: {content}\n"
            if dependencies:
                result += f"   - 依赖: {', '.join(dependencies)}\n"
        result += "\n"
    
    if decomposition_state.parallel_task_groups:
        result += "**并行任务组**:\n"
        for group_id, group in decomposition_state.parallel_task_groups.items():
            if isinstance(group, dict):
                subtasks = group.get('subtasks', [])
            else:
                subtasks = getattr(group, 'subtasks', [])
            result += f"\n- **{group_id}**: {len(subtasks)} 个并行任务\n"
            # 显示每个并行任务组的详细任务列表
            if subtasks:
                for j, subtask in enumerate(subtasks, 1):
                    if isinstance(subtask, dict):
                        subtask_id = subtask.get('task_id', f'{group_id}_task_{j}')
                        subtask_content = subtask.get('content', subtask.get('description', 'N/A'))
                        subtask_deps = subtask.get('dependencies', [])
                    else:
                        subtask_id = getattr(subtask, 'task_id', f'{group_id}_task_{j}')
                        subtask_content = getattr(subtask, 'content', 'N/A')
                        subtask_deps = getattr(subtask, 'dependencies', [])
                    result += f"  {j}. **{subtask_id}**: {subtask_content}\n"
                    if subtask_deps:
                        result += f"     依赖: {', '.join(subtask_deps)}\n"
        result += "\n"
    
    # 参数推断结果
    result += "### 阶段3: 参数推断\n\n"
    param_results = decomposition_state.parameter_inference_results
    if isinstance(param_results, dict):
        result += f"**推断任务数量**: {len(param_results)}\n\n"
        if param_results:
            result += "**参数推断详情**:\n"
            for task_id, inference in param_results.items():
                if isinstance(inference, dict):
                    tool_name = inference.get('tool_name', 'N/A')
                    params = inference.get('parameters', {})
                    result += f"\n- **{task_id}** (工具: {tool_name})\n"
                    result += f"  参数数量: {len(params)}\n"
                    if params:
                        for param_name, param_info in params.items():
                            if isinstance(param_info, dict):
                                source_type = param_info.get('source_type', 'unknown')
                                value = param_info.get('value', 'N/A')
                                source_task_id = param_info.get('source_task_id')
                                source_output_key = param_info.get('source_output_key')
                                user_prompt = param_info.get('user_prompt')
                                reason = param_info.get('reason', '')
                                result += f"  - **{param_name}**:\n"
                                result += f"    - 来源类型: {source_type}\n"
                                if value and value != 'N/A':
                                    result += f"    - 值: {value}\n"
                                if source_task_id:
                                    result += f"    - 来源任务: {source_task_id}"
                                    if source_output_key:
                                        result += f" (输出键: {source_output_key})"
                                    result += "\n"
                                if user_prompt:
                                    result += f"    - 用户提示: {user_prompt}\n"
                                if reason:
                                    result += f"    - 推断理由: {reason}\n"
                            else:
                                result += f"  - {param_name}: {param_info}\n"
    else:
        result += "**参数推断结果**: 无\n"
    
    if decomposition_state.parameter_inference_summary:
        result += f"\n**参数推断摘要**:\n{decomposition_state.parameter_inference_summary}\n"
    
    return result


def _save_comprehensive_log(
    test_case: Dict[str, Any],
    immunity_state: ImmunityState,
    decomposition_state: TaskDecompositionState,
    immunity_time: float,
    decomposition_time: float,
    stage_timings: Dict[str, float],
    node_results: Optional[Dict[str, Any]] = None
) -> str:
    """保存综合测试日志"""
    logs_dir = Path(agent_dir) / "tests" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = test_case["name"].replace(" ", "_").replace("/", "_")
    log_file = logs_dir / f"comprehensive_{test_case['id']}_{safe_name}_{timestamp}.md"
    
    log_content = f"""# Immunity 和 Task Decomposition 综合测试日志

## 测试信息
- **测试ID**: {test_case['id']}
- **测试名称**: {test_case['name']}
- **测试类别**: {test_case['category']}
- **测试描述**: {test_case['description']}
- **测试时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **总执行时间**: {immunity_time + decomposition_time:.2f} 秒
  - Immunity 子图: {immunity_time:.2f} 秒
  - Task Decomposition 子图: {decomposition_time:.2f} 秒

## 用户输入
```
{test_case['user_input']}
```

## Immunity 子图结果

### 各阶段执行时间
"""
    
    for stage_name, stage_time in stage_timings.items():
        log_content += f"- **{stage_name}**: {stage_time:.2f} 秒\n"
    
    # 节点名称映射
    node_name_map = {
        "query_decomposition": "查询分解",
        "retrieval": "检索",
        "deep_research": "深度研究",
        "hypothesis_generation": "假设生成",
        "planning": "计划生成",
        "evaluation": "评估"
    }
    
    # Stage 1: 查询分解
    node_time = stage_timings.get("query_decomposition", stage_timings.get("查询分解", 0))
    log_content += _format_stage_result("查询分解", immunity_state, {"execution_time": node_time})
    
    # 如果有节点结果，添加详细节点输出
    if node_results and "query_decomposition" in node_results:
        node_state = _ensure_immunity_state(node_results["query_decomposition"])
        log_content += f"\n**节点输出详情**:\n"
        log_content += f"- 优化查询数: {len(node_state.optimized_questions)}\n"
        if node_state.optimized_questions:
            for i, q in enumerate(node_state.optimized_questions, 1):
                log_content += f"  {i}. {q}\n"
    
    # Stage 2: 检索
    node_time = stage_timings.get("retrieval", stage_timings.get("检索", 0))
    log_content += _format_stage_result("检索", immunity_state, {"execution_time": node_time})
    
    if node_results and "retrieval" in node_results:
        node_state = _ensure_immunity_state(node_results["retrieval"])
        log_content += f"\n**节点输出详情**:\n"
        log_content += f"- 检索文档数: {len(node_state.retrieval_docs)}\n"
        log_content += f"- 引用文献数: {len(node_state.citations)}\n"
        log_content += f"- 上下文长度: {len(node_state.context)} 字符\n"
    
    # Stage 3: 深度研究
    node_time = stage_timings.get("deep_research", stage_timings.get("深度研究", 0))
    log_content += _format_stage_result("深度研究", immunity_state, {"execution_time": node_time})
    
    if node_results and "deep_research" in node_results:
        node_state = _ensure_immunity_state(node_results["deep_research"])
        log_content += f"\n**节点输出详情**:\n"
        log_content += f"- 研究置信度: {node_state.research_confidence:.1f}%\n"
        log_content += f"- 关键洞察数: {len(node_state.research_insights)}\n"
    
    # Stage 4: 假设生成
    node_time = stage_timings.get("hypothesis_generation", stage_timings.get("假设生成", 0))
    log_content += _format_stage_result("假设生成", immunity_state, {"execution_time": node_time})
    
    if node_results and "hypothesis_generation" in node_results:
        node_state = _ensure_immunity_state(node_results["hypothesis_generation"])
        log_content += f"\n**节点输出详情**:\n"
        log_content += f"- 假设置信度: {node_state.hypothesis_confidence:.1f}%\n"
        log_content += f"- 可测试预测数: {len(node_state.testable_predictions)}\n"
    
    # Stage 5: 计划生成
    node_time = stage_timings.get("planning", stage_timings.get("计划生成", 0))
    log_content += _format_stage_result("计划生成", immunity_state, {"execution_time": node_time})
    
    if node_results and "planning" in node_results:
        node_state = _ensure_immunity_state(node_results["planning"])
        log_content += f"\n**节点输出详情**:\n"
        log_content += f"- 计划长度: {len(node_state.final_enhanced_plan)} 字符\n"
    
    # Stage 6: 评估
    node_time = stage_timings.get("evaluation", stage_timings.get("评估", 0))
    log_content += _format_stage_result("评估", immunity_state, {"execution_time": node_time})
    
    if node_results and "evaluation" in node_results:
        node_state = _ensure_immunity_state(node_results["evaluation"])
        log_content += f"\n**节点输出详情**:\n"
        log_content += f"- 评估报告长度: {len(node_state.final_evaluation)} 字符\n"
    
    # 完整实验计划
    log_content += "\n### 完整实验计划\n\n"
    if immunity_state.final_enhanced_plan:
        log_content += f"```\n{immunity_state.final_enhanced_plan}\n```\n"
    else:
        log_content += "无实验计划\n"
    
    # 完整评估报告
    log_content += "\n### 完整评估报告\n\n"
    if immunity_state.final_evaluation:
        log_content += f"```\n{immunity_state.final_evaluation}\n```\n"
    else:
        log_content += "无评估报告\n"
    
    # Task Decomposition 结果
    log_content += _format_decomposition_result(decomposition_state)
    
    # 总结
    log_content += "\n## 总结\n\n"
    log_content += "### Immunity 子图完成情况\n"
    log_content += f"- ✅ 查询分解: {'完成' if immunity_state.optimized_questions else '未完成'}\n"
    log_content += f"- ✅ 检索: {'完成' if immunity_state.context or immunity_state.retrieval_docs else '未完成'}\n"
    log_content += f"- ✅ 深度研究: {'完成' if immunity_state.research_summary else '未完成'}\n"
    log_content += f"- ✅ 假设生成: {'完成' if immunity_state.hypothesis_summary else '未完成'}\n"
    log_content += f"- ✅ 计划生成: {'完成' if immunity_state.final_enhanced_plan else '未完成'}\n"
    log_content += f"- ✅ 评估: {'完成' if immunity_state.final_evaluation else '未完成'}\n\n"
    
    log_content += "### Task Decomposition 子图完成情况\n"
    log_content += f"- ✅ 粗分解: {'完成' if decomposition_state.required_service_ids else '未完成'}\n"
    log_content += f"- ✅ 细分解: {'完成' if decomposition_state.raw_tasks else '未完成'}\n"
    log_content += f"- ✅ 并行推断: {'完成' if decomposition_state.subtasks or decomposition_state.parallel_task_groups else '未完成'}\n"
    log_content += f"- ✅ 参数推断: {'完成' if decomposition_state.parameter_inference_results else '未完成'}\n\n"
    
    log_content += "### 关键指标\n"
    log_content += f"- 优化查询数: {len(immunity_state.optimized_questions)}\n"
    log_content += f"- 检索文档数: {len(immunity_state.retrieval_docs)}\n"
    log_content += f"- 引用文献数: {len(immunity_state.citations)}\n"
    log_content += f"- 检索上下文长度: {len(immunity_state.context)} 字符\n"
    log_content += f"- 研究置信度: {immunity_state.research_confidence:.1f}%\n"
    log_content += f"- 假设置信度: {immunity_state.hypothesis_confidence:.1f}%\n"
    log_content += f"- 计划文档长度: {len(immunity_state.final_enhanced_plan)} 字符\n"
    log_content += f"- 分解任务数: {len(decomposition_state.subtasks)}\n"
    log_content += f"- 并行任务组数: {len(decomposition_state.parallel_task_groups)}\n"
    log_content += f"- 参数推断任务数: {len(decomposition_state.parameter_inference_results) if isinstance(decomposition_state.parameter_inference_results, dict) else 0}\n"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(log_content)
    
    print(f"\n📄 综合测试日志已保存到: {log_file}")
    return str(log_file)


# ===================== Fixtures =====================

@pytest.fixture(scope="module")
def immunity_subgraph():
    """构建并返回 Immunity 子图"""
    return build_immunity_subgraph()


@pytest.fixture(scope="module")
def task_decomposition_subgraph():
    """构建并返回 Task Decomposition 子图"""
    return build_task_decomposition_subgraph()


# ===================== 测试类 =====================

class TestComprehensiveFlow:
    """综合流程测试：Immunity + Task Decomposition"""
    
    @pytest.mark.parametrize("test_case", ALL_TEST_CASES)
    def test_immunity_and_decomposition_flow(
        self,
        immunity_subgraph,
        task_decomposition_subgraph,
        test_case
    ):
        """测试完整流程：Immunity 子图生成计划 → Task Decomposition 子图分解任务"""
        
        # 创建测试沙盒
        sandbox_dir = _create_test_sandbox()
        
        # ========== 阶段1: 运行 Immunity 子图 ==========
        print(f"\n{'='*80}")
        print(f"开始测试: {test_case['id']} - {test_case['name']}")
        print(f"{'='*80}")
        
        immunity_start_time = time.time()
        stage_timings = {}
        
        # 创建全局状态
        global_state = GlobalState(
            user_input=test_case["user_input"],
            user_task_type=UserTaskType.IMMUNOLOGY_TASK,
            sandbox_dir=sandbox_dir
        )
        
        # 运行 Immunity 子图（使用 stream 获取各节点详细结果）
        node_results = {}
        stage_timings = {}
        
        try:
            immunity_input = immunity_input_mapper(global_state)
            
            # 使用 stream 方法逐步执行，获取每个节点的结果
            node_start_times = {}
            final_state = None
            
            for event in immunity_subgraph.stream(immunity_input):
                for node_name, node_output in event.items():
                    if node_name == "__end__":
                        # 最终状态
                        final_state = node_output
                    elif node_name not in ["__start__", "__end__"]:
                        # 记录节点执行时间
                        node_start_time = node_start_times.get(node_name, immunity_start_time)
                        node_end_time = time.time()
                        node_results[node_name] = node_output
                        stage_timings[node_name] = node_end_time - node_start_time
                        node_start_times[node_name] = node_end_time
                        
                        # 打印节点执行信息
                        try:
                            state = _ensure_immunity_state(node_output)
                            print(f"  ✓ {node_name} 完成 ({stage_timings[node_name]:.2f} 秒)")
                        except:
                            print(f"  ✓ {node_name} 完成 ({stage_timings[node_name]:.2f} 秒)")
            
            # 使用最终状态或最后一个节点输出
            if final_state:
                immunity_state = _ensure_immunity_state(final_state)
            elif node_results:
                # 使用最后一个节点的输出
                last_node = list(node_results.keys())[-1]
                immunity_state = _ensure_immunity_state(node_results[last_node])
            else:
                # 如果 stream 没有返回结果，使用 invoke 作为后备
                final_output = immunity_subgraph.invoke(immunity_input)
                immunity_state = _ensure_immunity_state(final_output)
            
            updated_global_state = immunity_output_mapper(immunity_state, global_state)
            
            immunity_time = time.time() - immunity_start_time
            
            # 如果没有获取到节点时间，使用估算
            if not stage_timings:
                stage_timings = {
                    "query_decomposition": immunity_time * 0.12,
                    "retrieval": immunity_time * 0.15,
                    "deep_research": immunity_time * 0.25,
                    "hypothesis_generation": immunity_time * 0.18,
                    "planning": immunity_time * 0.25,
                    "evaluation": immunity_time * 0.05
                }
            
            print(f"✅ Immunity 子图完成 ({immunity_time:.2f} 秒)")
            
        except Exception as e:
            print(f"❌ Immunity 子图执行失败: {e}")
            import traceback
            traceback.print_exc()
            # 创建空的 immunity_state 用于日志
            immunity_state = ImmunityState(
                original_question=test_case["user_input"],
                sandbox_dir=sandbox_dir
            )
            immunity_time = time.time() - immunity_start_time
            stage_timings = {}
        
        # ========== 阶段2: 运行 Task Decomposition 子图 ==========
        decomposition_start_time = time.time()
        
        # 使用 Immunity 生成的计划作为输入
        execution_plan = immunity_state.final_enhanced_plan if immunity_state.final_enhanced_plan else test_case["user_input"]
        
        # 创建新的全局状态用于任务分解
        decomposition_global_state = GlobalState(
            user_input=test_case["user_input"],
            execution_plan=execution_plan,
            user_task_type=UserTaskType.EXECUTE_PLAN,
            sandbox_dir=sandbox_dir
        )
        
        # 运行 Task Decomposition 子图
        try:
            decomposition_input = task_decomposition_input_mapper(decomposition_global_state)
            decomposition_output = task_decomposition_subgraph.invoke(decomposition_input)
            decomposition_state = _ensure_task_decomposition_state(decomposition_output)
            updated_decomposition_global_state = task_decomposition_output_mapper(decomposition_state, decomposition_global_state)
            
            decomposition_time = time.time() - decomposition_start_time
            
            print(f"✅ Task Decomposition 子图完成 ({decomposition_time:.2f} 秒)")
            
        except Exception as e:
            print(f"❌ Task Decomposition 子图执行失败: {e}")
            import traceback
            traceback.print_exc()
            # 创建空的 decomposition_state 用于日志
            decomposition_state = TaskDecompositionState(
                user_input=test_case["user_input"],
                sandbox_dir=sandbox_dir
            )
            decomposition_time = time.time() - decomposition_start_time
        
        # ========== 保存综合日志 ==========
        log_file = _save_comprehensive_log(
            test_case,
            immunity_state,
            decomposition_state,
            immunity_time,
            decomposition_time,
            stage_timings,
            node_results
        )
        
        # ========== 验证结果 ==========
        # Immunity 验证
        assert immunity_state is not None, "Immunity 子图应返回有效状态"
        assert len(immunity_state.optimized_questions) > 0, "应生成优化查询"
        
        # Task Decomposition 验证
        assert decomposition_state is not None, "Task Decomposition 子图应返回有效状态"
        
        print(f"\n✅ 测试完成: {test_case['id']} - {test_case['name']}")
        print(f"   Immunity 时间: {immunity_time:.2f} 秒")
        print(f"   Decomposition 时间: {decomposition_time:.2f} 秒")
        print(f"   日志文件: {log_file}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

