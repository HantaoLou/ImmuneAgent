"""
Improved LangGraph workflow with reordered stages.
This ordering ensures that the plan generation is informed by deep research
and hypothesis rather than generating plans before understanding the research.
"""

import asyncio
import json
from datetime import datetime

from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from usecases.immunity.common.factory import (
    get_hypothesis_model,
    get_planning_model,
)
from usecases.immunity.common.utils import (
    clean_json_response,
    save_planning_report,
)
from usecases.immunity.graph.retrieval_graph import query_rewriter, retrieval_agent
from usecases.immunity.prompts.prompts import (
    ImmunityPrompts,
)
from usecases.immunity.state.state import (
    ImprovedCellState,
)
from usecases.immunity.schema.common_schemas import PlanStep
from usecases.immunity.tools.deepresearch_tools import (
    DeepResearchTool,
)


async def _confirm_plan_step_console(
    step: PlanStep, step_index: int, total_steps: int
) -> PlanStep | None | bool:
    """
    在控制台逐条确认计划步骤
    
    Args:
        step: 要确认的计划步骤
        step_index: 当前步骤索引（从1开始）
        total_steps: 总步骤数
    
    Returns:
        PlanStep: 确认后的步骤（可能被修改）
        None: 用户拒绝整个计划
        False: 用户拒绝/跳过当前步骤（不影响其他步骤）
    """
    print("\n" + "-" * 70)
    print(f"步骤 {step_index}/{total_steps}")
    print("-" * 70)
    
    # 显示步骤信息
    print(f"步骤ID: {step.step_id}")
    print(f"标题: {step.title}")
    if step.description:
        print(f"Description: {step.description}")
    if step.objective:
        print(f"Objective: {step.objective}")
    
    # 显示工具信息
    all_tools = step.tools or step.toolchain or step.recommended_tools or []
    if all_tools:
        tools_str = ", ".join(all_tools)
        print(f"工具: {tools_str}")
    
    # 显示输入输出
    if step.inputs:
        print(f"输入: {', '.join(step.inputs)}")
    if step.outputs:
        print(f"输出: {', '.join(step.outputs)}")
    
    if step.notes:
        print(f"备注: {step.notes}")
    
    print("-" * 70)
    print("请选择操作:")
    print("  1. 确认此步骤")
    print("  2. 修改此步骤")
    print("  3. 拒绝此步骤（跳过，不影响其他步骤）")
    print("  4. 拒绝整个计划（取消所有步骤）")
    print("-" * 70)
    
    while True:
        try:
            choice = input("请选择 (1-4): ").strip()
            
            if choice == "1":
                # 确认步骤
                return step
            
            elif choice == "2":
                # 修改步骤
                modified_step = await _modify_plan_step_console(step)
                if modified_step:
                    return modified_step
                # 如果修改失败或取消，继续循环
                continue
            
            elif choice == "3":
                # 拒绝/跳过当前步骤（不影响其他步骤）
                print(f"⚠️ 步骤 {step_index} 已被拒绝，将跳过此步骤")
                return False
            
            elif choice == "4":
                # 拒绝整个计划
                return None
            
            else:
                print("❌ 无效选择，请输入 1-4")
        
        except KeyboardInterrupt:
            print("\n⚠️ 操作已取消")
            return None
        except Exception as e:
            print(f"❌ 输入错误: {e}")


async def _modify_plan_step_console(step: PlanStep) -> PlanStep | None:
    """
    在控制台修改计划步骤
    
    Args:
        step: 要修改的步骤
    
    Returns:
        PlanStep: 修改后的步骤
        None: 取消修改
    """
    print("\n" + "=" * 70)
    print("Modify Plan Step")
    print("=" * 70)
    print("Current step information:")
    print(f"  Title: {step.title}")
    print(f"  Description: {step.description}")
    print(f"  Objective: {step.objective}")
    print(f"  Tools: {', '.join(step.tools or step.toolchain or [])}")
    print("=" * 70)
    print("Modifiable fields:")
    print("  1. Title")
    print("  2. Description")
    print("  3. Objective")
    print("  4. Tools list")
    print("  5. Notes")
    print("  6. Complete modification and confirm")
    print("  7. Cancel modification")
    print("=" * 70)
    
    modified_data = step.model_dump()
    
    while True:
        try:
            choice = input("Please select field to modify (1-7): ").strip()
            
            if choice == "1":
                new_title = input(f"New title (current: {step.title}): ").strip()
                if new_title:
                    modified_data["title"] = new_title
                    print(f"Title updated to: {new_title}")
            
            elif choice == "2":
                print(f"Current description: {step.description}")
                new_description = input("New description (multi-line, empty line to end): ").strip()
                if new_description:
                    modified_data["description"] = new_description
                    print(f"Description updated")
            
            elif choice == "3":
                new_objective = input(f"New objective (current: {step.objective}): ").strip()
                if new_objective:
                    modified_data["objective"] = new_objective
                    print(f"Objective updated to: {new_objective}")
            
            elif choice == "4":
                current_tools = ", ".join(step.tools or step.toolchain or [])
                print(f"Current tools: {current_tools}")
                new_tools_str = input("New tools list (comma-separated): ").strip()
                if new_tools_str:
                    new_tools = [t.strip() for t in new_tools_str.split(",") if t.strip()]
                    modified_data["tools"] = new_tools
                    print(f"Tools list updated to: {', '.join(new_tools)}")
            
            elif choice == "5":
                print(f"Current notes: {step.notes}")
                new_notes = input("New notes: ").strip()
                if new_notes:
                    modified_data["notes"] = new_notes
                    print(f"Notes updated")
            
            elif choice == "6":
                # Complete modification
                try:
                    modified_step = PlanStep(**modified_data)
                    print("\nStep modification completed")
                    return modified_step
                except Exception as e:
                    print(f"Error: Step modification failed: {e}")
                    return None
            
            elif choice == "7":
                # Cancel modification
                print("Modification cancelled")
                return None
            
            else:
                print("Invalid selection, please enter 1-7")
        
        except KeyboardInterrupt:
            print("\nModification cancelled")
            return None
        except Exception as e:
            print(f"Input error: {e}")


def stage0_plan_detection(
    state: ImprovedCellState, config: RunnableConfig
) -> ImprovedCellState:
    """
    Stage 0: 检测用户输入是否是计划格式
    
    判断逻辑：
    1. 包含明确的计划关键词（"计划"、"步骤"、"任务"、"使用工具"等）
    2. 包含工具调用指示（"使用 metabcr"、"调用工具"等）
    3. 结构化格式（列表、编号等）
    
    Args:
        state: 当前工作流状态
        config: 运行配置
        
    Returns:
        更新后的状态（设置 skip_planning 和 is_user_provided_plan）
    """
    print("\n" + "=" * 50)
    print("🔍 STAGE 0: PLAN DETECTION")
    print("=" * 50)
    
    query = state.original_question
    
    # 调试：打印查询内容的前500个字符
    
    # 计划格式特征关键词（中文）
    plan_keywords_cn = [
        "使用工具",
        "调用工具",
        "使用.*工具",  # 正则模式：使用 + 工具名称 + 工具
        "执行以下",
        "按照以下",
        "任务1",
        "步骤1",
        "第一步",
        "请执行",
        "请按照",
        "计划如下",
        "工作流程",
        "流程",
        "计划",
        "进行",  # "进行抗体预测"、"分析B细胞"
        "分析",  # "分析B细胞亚型"
    ]
    
    # 计划格式特征关键词（英文）
    plan_keywords_en = [
        "tools:",
        "tool:",
        "workflow:",
        "workflow",
        "pipeline",
        "pipeline:",
        "step",
        "steps",
        "task",
        "tasks",
        "execute",
        "following",
        "plan",
        "planning",
    ]
    
    # 工具调用关键词（扩展，包括更多工具名称）
    tool_keywords = [
        "metabcr",
        "bcell",
        "bioinformatics",
        "run_figure",
        "igblast",
        "anarci",
        "alphafold",
        "foldx",
        "ddg",
        "gearbind",
        "sabdab",
        "airr",
        "oas",
        "nettcr",
        "tulip",
        "clustering",
        "trajectory",
        "cell communication",
        "celltype",
        "distribution",
        "analysis",
        r"使用.*工具",
        r"调用.*工具",
        r"\btool\w*\b",  # 匹配 tool, tools, tooling 等
        r"\w+_\w+",      # 匹配下划线分隔的工具名称（如 bcell_celltype_distribution_analysis）
    ]
    
    # 结构化格式特征（扩展）
    import re
    structured_patterns = [
        r"步骤\s*\d+[:：]",           # 步骤1:、步骤2:
        r"任务\s*\d+[:：]",           # 任务1:、任务2:
        r"\d+\.\s+.*[Tt]ool",        # 1. Tool name
        r"\d+\.\s+.*工具",            # 1. 使用工具
        r"-.*[Tt]ool",               # - Tool name
        r"-.*工具",                   # - 使用工具
        r"•.*[Tt]ool",               # • Tool name
        r"•.*工具",                   # • 使用工具
        r"^\d+\.\s+",                # 行首数字编号：1. 2. 3.
        r"^[Tt]ools?:",              # 行首 Tools: 或 Tool:
        r"^[Ww]orkflow:",            # 行首 Workflow:
        r"^[Dd]ataset:",             # 行首 Dataset:
        r"^[Pp]ipeline:",            # 行首 Pipeline:
        r":\s*[A-Z]",                # 冒号后跟大写字母（如 Tools: OAS）
    ]
    
    # 检测是否包含计划特征
    query_lower = query.lower()
    # 对于中文关键词，需要检查是否包含关键词或匹配正则模式
    has_plan_keywords_cn = False
    for keyword in plan_keywords_cn:
        # 检查是否包含正则模式（包含 .* 或其他正则特殊字符）
        if '.*' in keyword or '\\w' in keyword or '\\s' in keyword:
            # 正则模式
            try:
                if re.search(keyword, query, re.IGNORECASE):
                    has_plan_keywords_cn = True
                    break
            except re.error:
                # 如果正则表达式无效，跳过
                pass
        else:
            # 普通字符串
            if keyword in query:
                has_plan_keywords_cn = True
                break
    
    has_plan_keywords_en = any(keyword in query_lower for keyword in plan_keywords_en)
    has_plan_keywords = has_plan_keywords_cn or has_plan_keywords_en
    
    # 检测工具关键词（使用正则匹配）
    has_tool_keywords = False
    for kw in tool_keywords:
        if re.search(kw, query, re.IGNORECASE):
            has_tool_keywords = True
            break
    
    # 检测结构化格式（按行检测，因为用户输入可能是多行）
    query_lines = query.split('\n')
    has_structured = False
    structured_count = 0
    
    for line in query_lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        # 检查是否匹配结构化模式
        for pattern in structured_patterns:
            if re.search(pattern, line_stripped):
                structured_count += 1
                has_structured = True
                break
        # 如果一行中同时包含多个工具名称，也认为是结构化（不区分大小写）
        tool_names_in_line = ["OAS", "IgBLAST", "ANARCI", "AlphaFold", "FoldX", "MetaBCR", "AIRR", "GearBind", "DDG", "SAbDab", "NetTCR", "AlphaFold3", "GearBind", "DDG Predictor"]
        line_upper = line_stripped.upper()
        tool_count_in_line = sum(1 for tool in tool_names_in_line if tool.upper() in line_upper)
        if tool_count_in_line >= 2:
            structured_count += 1
            has_structured = True
    
    # 检测是否包含多个工具名称（工具列表特征，不区分大小写）
    # 去重工具列表，按长度排序（长的先匹配，避免 AlphaFold3 被 AlphaFold 匹配）
    tool_list_keywords = [
        "AlphaFold3", "DDG Predictor", "GearBind", "IgBLAST", "AlphaFold", 
        "MetaBCR", "SAbDab", "NetTCR", "TULIP", "OAS", "ANARCI", "FoldX", 
        "AIRR", "DDG", "GEO", "PDB", "bcell", "bcell_celltype", 
        "celltype_distribution", "distribution_analysis"
    ]
    query_upper = query.upper()
    tool_count = 0
    found_tools = []
    matched_tool_names = set()  # 用于去重
    
    for tool in tool_list_keywords:
        # 不区分大小写检测
        tool_upper = tool.upper()
        if tool_upper in query_upper:
            # 避免重复匹配：如果已有更长的工具名称（如 AlphaFold3），则跳过短的（AlphaFold）
            is_duplicate = False
            for existing_tool in matched_tool_names:
                existing_upper = existing_tool.upper()
                # 如果当前工具是已有工具的子串（如 AlphaFold 是 AlphaFold3 的子串），跳过
                if tool_upper != existing_upper and (tool_upper in existing_upper or existing_upper in tool_upper):
                    # 保留更长的那个
                    if len(tool) < len(existing_tool):
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                # 移除可能被当前工具包含的已有工具
                matched_tool_names = {t for t in matched_tool_names if not (t.upper() != tool_upper and (t.upper() in tool_upper or tool_upper in t.upper()) and len(t) < len(tool))}
                tool_count += 1
                found_tools.append(tool)
                matched_tool_names.add(tool)
    
    # 检测工具名称模式（支持下划线分隔的工具名称，如 bcell_celltype_distribution_analysis）
    # 使用正则表达式检测下划线分隔的工具名称模式
    underscore_tool_pattern = r'\b\w+_\w+(_\w+)*\b'
    underscore_tools = re.findall(underscore_tool_pattern, query)
    # 过滤掉常见的非工具名称（如 input_path, output_path）
    filtered_underscore_tools = [t for t in underscore_tools if not any(
        skip in t.lower() for skip in ['input_path', 'output_path', 'file_path', 'path', 'session_id']
    )]
    underscore_tool_count = len(set(filtered_underscore_tools))
    
    # 如果检测到下划线分隔的工具名称，也计入工具数量
    if underscore_tool_count > 0:
        tool_count += underscore_tool_count
        found_tools.extend(filtered_underscore_tools)
    
    has_multiple_tools = tool_count >= 2  # 降低要求：至少包含2个工具名称
    
    # 检测是否有编号列表（如 1. 2. 3. 或 1) 2) 3)）
    # 支持多种格式：1. 2. 3. 或 1) 2) 3) 或 1. Title 等
    numbered_list_patterns = [
        r"^\s*\d+[\.\)]\s+",           # 1. 或 1)
        r"^\d+\.\s+[A-Z]",            # 1. Title (编号后跟大写字母)
        r"^\d+\.\s+",                 # 1. (任意内容)
    ]
    numbered_lines = 0
    for line in query_lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        for pattern in numbered_list_patterns:
            if re.match(pattern, line_stripped):
                numbered_lines += 1
                break
    has_numbered_list = numbered_lines >= 2  # 降低要求：至少2个编号项即可
    
    # 综合判断：使用更灵活的评分系统
    plan_score = 0
    if has_plan_keywords:
        plan_score += 2
    if has_tool_keywords:
        plan_score += 1
    if has_structured or structured_count >= 2:
        plan_score += 2
    if has_multiple_tools:
        plan_score += 2
    if has_numbered_list:
        plan_score += 2
    
    # 降低阈值：满足至少2分即可认为是计划（更宽松的检测）
    # 如果包含多个工具或编号列表，即使没有关键词也可以识别为计划
    # 或者：有编号列表 + 工具关键词，也应该识别为计划
    is_plan = plan_score >= 2 or (has_multiple_tools and has_numbered_list) or (has_numbered_list and has_tool_keywords)
    
    if is_plan:
        print(f"✅ 检测到用户直接提供了计划")
        print(f"   - 计划关键词: {has_plan_keywords} (CN: {has_plan_keywords_cn}, EN: {has_plan_keywords_en})")
        print(f"   - 工具关键词: {has_tool_keywords}")
        print(f"   - 结构化格式: {has_structured} (结构化行数: {structured_count})")
        print(f"   - 多个工具: {has_multiple_tools} (工具数量: {tool_count})")
        print(f"   - 编号列表: {has_numbered_list} (编号行数: {numbered_lines})")
        print(f"   - 计划评分: {plan_score}/9")
        
        # 设置标志位
        state.skip_planning = True
        state.is_user_provided_plan = True
        
        # 直接将用户输入作为计划（可以稍后优化格式化）
        state.final_enhanced_plan = query
        
        # 跳过研究阶段，直接设置为空（节省时间）
        state.deep_research_findings = {}
        state.hypothesis = {}
        
        print(f"   📋 计划内容预览: {query[:200]}...")
    else:
        print(f"📝 检测为普通问题，需要生成计划")
        print(f"   - 计划评分: {plan_score}/9 (阈值: 3)")
        print(f"   - 详细检测结果:")
        print(f"     * 计划关键词: {has_plan_keywords} (CN: {has_plan_keywords_cn}, EN: {has_plan_keywords_en})")
        print(f"     * 工具关键词: {has_tool_keywords}")
        print(f"     * 结构化格式: {has_structured} (结构化行数: {structured_count})")
        print(f"     * 多个工具: {has_multiple_tools} (工具数量: {tool_count}, 找到的工具: {found_tools})")
        print(f"     * 编号列表: {has_numbered_list} (编号行数: {numbered_lines})")
        print(f"   - 前10行内容:")
        for i, line in enumerate(query_lines[:10], 1):
            print(f"     {i}: {line.strip()[:80]}")
        state.skip_planning = False
        state.is_user_provided_plan = False
    
    return state


def stage1_query_decomposition(
    state: ImprovedCellState, config: RunnableConfig
) -> ImprovedCellState:
    """
    Stage 1: Query Decomposition and Optimization.

    Breaks down complex queries into optimized sub-questions.
    """
    print("\n" + "=" * 50)
    print("📝 STAGE 1: QUERY DECOMPOSITION")
    print("=" * 50)

    # Use existing query rewriter (not async)
    result = query_rewriter(state, config)

    print(f"✅ Query optimized:")
    print(f"  - Original: {state.original_question[:100]}...")
    print(f"  - Optimized queries: {len(state.optimized_questions)}")

    return result


async def stage2_immunology_retrieval(
    state: ImprovedCellState, config: RunnableConfig
) -> ImprovedCellState:
    """
    Stage 2: Retrieve from Immunology Knowledge Base.

    Retrieves relevant context from immunology papers and databases.
    Initializes citation manager for tracking references.
    """
    print("\n" + "=" * 50)
    print("🔍 STAGE 2: IMMUNOLOGY RETRIEVAL")
    print("=" * 50)

    # Use existing retrieval agent (now async)
    result = await retrieval_agent(state, config)
    retrieval_docs = state.retrieval_docs
    # Filter each document collection, taking only the first 15 documents from each docs collection
    content_list = []
    filtered_retrieval_docs = []
    for docs in retrieval_docs:
        # Take the first 15 documents, no longer limiting character length
        filtered_docs = docs[:15]

        for document in filtered_docs:
            content_list.append(
                f"""
<document>
    <source>{document.source}</source>
    <content>{document.content}</content>
</document>
"""
            )
        filtered_retrieval_docs.append(filtered_docs)
    # Update retrieval_docs in state with filtered results
    state.retrieval_docs = filtered_retrieval_docs
    state.context = "\n\n".join(content_list)

    print(f"\n✅ Retrieved context:")
    print(f"  - Context length: {len(result.context)} chars")
    uuid = config["configurable"]["uuid"]
    # 保存文件并将路径存储到state中
    retrieval_file_path = save_planning_report(state.context, uuid, "retrieval")
    state.retrieval_report_path = retrieval_file_path
    return result


async def stage3_deep_research_analysis(
    state: ImprovedCellState, config: RunnableConfig
) -> ImprovedCellState:
    """
    Stage 3: Deep Research Analysis (MOVED EARLIER).

    Analyzes retrieved context deeply BEFORE planning.
    This ensures research insights inform the planning process.
    """
    print("\n" + "=" * 50)
    print("🔬 STAGE 3: DEEP RESEARCH ANALYSIS (Early Phase)")
    print("=" * 50)

    try:
        # Initialize deep research tool
        research_tool = DeepResearchTool(config)

        # Extract data from state
        query = state.original_question
        context = state.context
        optimized_queries = state.optimized_questions

        # Conduct deep research analysis (already in async context)
        research_findings = await research_tool.conduct_deep_research(
            query=query,
            context=context,
            optimized_queries=optimized_queries,
        )

        # Store research findings in state
        state.deep_research_findings = research_findings.model_dump()
        state.research_confidence = research_findings.confidence
        state.research_insights = research_findings.key_insights
        state.research_evidence = research_findings.evidence
        state.research_gaps = research_findings.gaps
        state.research_recommendations = research_findings.recommendations
        context_summary = f"""
Research Topic: {research_findings.topic}

Sub-questions from decomposition:
{chr(10).join([f"- {q}" for q in state.optimized_questions[:5]])}

Key research insights (Confidence: {state.research_confidence:.1f}%):
{chr(10).join([f"- {insight}" for insight in state.research_insights[:5]])}

Supporting evidence:
{chr(10).join([f"- {evidence}" for evidence in state.research_evidence[:5]])}

Research findings summary:
{research_findings.summary}

Structured evidence claims:
{
            chr(10).join(
                [
                    f"- {claim.get('claim', '')} (Confidence: {claim.get('confidence', 0):.1f}%)"
                    for claim in research_findings.evidenced_claims[:5]
                ]
            )
        }

Knowledge gaps identified:
{chr(10).join([f"- {gap}" for gap in (state.research_gaps or [])[:5]])}

Research recommendations:
{chr(10).join([f"- {rec}" for rec in (state.research_recommendations or [])[:5]])}

Confidence breakdown:
{
            chr(10).join(
                [
                    f"- {aspect}: {conf:.1f}%"
                    for aspect, conf in research_findings.confidence_breakdown.items()
                ]
            )
        }
"""
        research_summary = f"""
<research_findings>
    <research_finding>
        {context_summary}
    </research_finding>
</research_findings>
"""
        state.research_summary = research_summary
        print(f"✅ Deep research summary: {state.research_summary}")
        uuid = config["configurable"]["uuid"]
        # 保存深度研究报告并记录文件路径
        research_file_path = save_planning_report(
            state.research_summary, uuid, "deep_research"
        )
        state.research_report_path = research_file_path
        return state
    except Exception as e:
        print(f"⚠️ Deep research analysis error in stage3_deep_research_analysis: {e}")
        print(
            f"  Context: Query='{state.original_question[:50]}...', Context length={len(state.context)}"
        )
        print("  Using fallback analysis with dynamic confidence calculation...")


async def stage4_hypothesis_generation(
    state: ImprovedCellState, config: RunnableConfig
) -> ImprovedCellState:
    """
    Stage 4: Hypothesis Generation based on previous research stages.

    Generates testable hypotheses based on:
    1. Decomposed sub-questions from Stage 1
    2. Retrieved scientific literature from Stage 2
    3. Deep research analysis from Stage 3

    Returns structured JSON with falsification criteria and evidence basis.
    """
    print("\n" + "=" * 50)
    print("🧬 STAGE 4: HYPOTHESIS GENERATION")
    print("=" * 50)

    try:
        # Gather context from previous stages
        print("\n📋 Gathering research context...")
        context = state.context
        question = f"""
<questions>
    <question>
        {state.original_question}
    </question>
</questions>
"""
        # Format the hypothesis generation prompt
        formatted_prompt = ImmunityPrompts.HYPOTHESIS_GENERATION_PROMPT.format(
            research_findings=state.research_summary, context=context, question=question
        )

        print("\n🤖 Generating hypothesis...")

        # Get model and generate hypothesis
        model = get_hypothesis_model(config)

        # Use system message to enforce JSON output
        from langchain_core.messages import SystemMessage

        messages = [SystemMessage(content=formatted_prompt)]

        # Generate hypothesis with JSON format enforcement
        raw_response = await model.ainvoke(
            messages, response_format={"type": "json_object"}
        )
        # Parse JSON response
        hypothesis_result = clean_json_response(
            raw_response.content
            if hasattr(raw_response, "content")
            else str(raw_response)
        )

        if not isinstance(hypothesis_result, dict) or not hypothesis_result.get(
            "statement"
        ):
            raise ValueError("Invalid hypothesis structure generated")

        # Store hypothesis results
        state.hypothesis = hypothesis_result
        state.hypothesis_confidence = float(
            hypothesis_result.get("confidence_score", 70.0)
        )

        # Extract testable predictions
        predictions = hypothesis_result.get("testable_predictions", [])
        state.testable_predictions = []
        for pred in predictions:
            if isinstance(pred, dict):
                state.testable_predictions.append(pred.get("prediction", ""))
            else:
                state.testable_predictions.append(str(pred))

        # Format hypothesis summary for next stage, similar to stage3's context_summary approach
        hypothesis_summary = f"""
Hypothesis Statement: {state.hypothesis.get("statement", "Not specified")}

Confidence Score: {state.hypothesis.get("confidence_score", 0)}%
Innovation Level: {state.hypothesis.get("innovation_level", "moderate")}

Testable Predictions:
{
            chr(10).join(
                [
                    f"- {pred.get('prediction', '')} (Timeline: {pred.get('timeline', 'TBD')})"
                    for pred in state.hypothesis.get("testable_predictions", [])[:5]
                ]
            )
        }

Validation Methods:
{
            chr(10).join(
                [
                    f"- {pred.get('validation_method', '')}"
                    for pred in state.hypothesis.get("testable_predictions", [])[:5]
                ]
            )
        }

Expected Outcomes:
{
            chr(10).join(
                [
                    f"- {pred.get('expected_outcome', '')}"
                    for pred in state.hypothesis.get("testable_predictions", [])[:5]
                ]
            )
        }

Falsification Criteria:
{
            chr(10).join(
                [
                    f"- {criteria}"
                    for criteria in state.hypothesis.get("falsification_criteria", [])[
                        :5
                    ]
                ]
            )
        }

Evidence Basis:
{
            chr(10).join(
                [
                    f"- {evidence}"
                    for evidence in state.hypothesis.get("evidence_basis", [])[:5]
                ]
            )
        }

Expected Information Gain:
{state.hypothesis.get("expected_information_gain", "To be determined")}

Scientific Rationale:
{state.hypothesis.get("rationale", "Not provided")}
"""

        # Store formatted hypothesis summary for next stage use
        hypothesis_findings = f"""
<hypothesis_findings>
    <hypothesis_finding>
        {hypothesis_summary}
    </hypothesis_finding>
</hypothesis_findings>
"""
        state.hypothesis_summary = hypothesis_findings
        print(f"✅ Hypothesis summary: {state.hypothesis_summary}")
        uuid = config["configurable"]["uuid"]
        # 保存假设报告并记录文件路径
        hypothesis_file_path = save_planning_report(
            state.hypothesis_summary, uuid, "hypothesis"
        )
        state.hypothesis_report_path = hypothesis_file_path
        return state
    except Exception as e:
        print(f"\n⚠️ Hypothesis generation failed: {e}")
        print("  Using fallback hypothesis based on available research...")


async def stage5_research_informed_planning(
    state: ImprovedCellState, config: RunnableConfig
) -> ImprovedCellState:
    """
    Stage 5: Research-Informed Planning.
    """
    print("\n" + "=" * 60)
    print("🔧 STAGE 5: RESEARCH-INFORMED PLANNING")
    print("=" * 60)

    try:
        import json

        from usecases.immunity.common.constants import get_tools_json

        prompt_tools_info = get_tools_json()
        research_findings = state.research_summary
        hypothesis_findings = state.hypothesis_summary
        context = state.context
        original_question = state.original_question
        optimized_questions = state.optimized_questions

        # Get the first 15 citations and convert to JSON format (remove abstract field to reduce token consumption)
        citations_json = "[]"
        if state.citations:
            try:
                # Get the first 15 citations
                top_citations = state.citations[:15]
                # Convert to dictionary list and remove abstract field
                citations_data = []
                for citation in top_citations:
                    citation_dict = citation.model_dump()
                    # Remove abstract field to reduce token consumption
                    if "abstract" in citation_dict:
                        del citation_dict["abstract"]
                    citations_data.append(citation_dict)
                # Convert to JSON string
                citations_json = json.dumps(
                    citations_data, indent=2, ensure_ascii=False
                )
            except Exception as e:
                citations_json = "[]"

        format_querys = []
        for i, optimized_query in enumerate(optimized_questions, 1):
            format_querys.append(f"""
<sub_questions>
    <q{i}>
        {optimized_query}
    </q{i}>
</sub_questions>
""")
        optimized_questions = "\n\n".join(format_querys) if format_querys else "None"

        # Format the planning prompt
        formatted_prompt = ImmunityPrompts.IMMUNITY_PLANNING_PROMPT.format(
            original_question=original_question,
            optimized_questions=optimized_questions,
            hypothesis_findings=hypothesis_findings,
            tools_info=prompt_tools_info,
            research_findings=research_findings,
            context=context,
            citations_json=citations_json,
        )
        # Get model and generate hypothesis
        model = get_planning_model(config)

        # Use system message to enforce JSON output
        from langchain_core.messages import SystemMessage

        messages = [SystemMessage(content=formatted_prompt)]

        # Generate hypothesis with JSON format enforcement
        response = await model.ainvoke(messages)
        final_enhanced_plan = (
            response.content if hasattr(response, "content") else str(response)
        )
        state.final_enhanced_plan = final_enhanced_plan
        uuid = config["configurable"]["uuid"]
        # 保存文件并将路径存储到state中
        planning_file_path = save_planning_report(final_enhanced_plan, uuid, "planning")
        state.planning_report_path = planning_file_path
        return state
    except Exception as e:
        print(f"⚠️ Tool selection failed: {str(e)}")
        print("  Using default tool set as fallback")


async def stage6_evaluate_planning(
    state: ImprovedCellState, config: RunnableConfig
) -> ImprovedCellState:
    """
    Stage 6: Evaluate Planning.
    Evaluate the planning based on the evaluation framework.
    """
    print("\n" + "=" * 60)
    print("🔧 STAGE 6: EVALUATE PLANNING")
    print("=" * 60)

    # === 新增：如果是用户提供的计划，直接使用 ===
    if state.is_user_provided_plan and state.final_enhanced_plan:
        print("\n" + "=" * 50)
        print("STAGE 6: Using User Provided Plan")
        print("=" * 50)
        
        # 可选：对用户计划进行简单验证和格式化
        # 但不改变计划的核心内容
        
        # 保存计划报告
        uuid = config["configurable"]["uuid"]
        planning_file_path = save_planning_report(
            state.final_enhanced_plan, 
            uuid, 
            "user_provided_planning"
        )
        state.planning_report_path = planning_file_path
        
        # 创建一个简单的评估结果（用户计划视为已验证）
        state.final_evaluation = f"""用户直接提供的执行计划：

{state.final_enhanced_plan}

---
评估说明：此计划由用户直接提供，已跳过自动评估步骤。系统将按照用户指定的计划执行任务。
"""
        
        # 保存评估报告
        evaluation_file_path = save_planning_report(
            state.final_evaluation, uuid, "evaluation"
        )
        state.evaluation_report_path = evaluation_file_path
        
        return state

    # === 原有逻辑（自动生成的计划需要评估）===
    try:
        # Format the evaluation prompt
        formatted_prompt = ImmunityPrompts.EVALUATE_PLANNING_PROMPT.format(
            plan=state.final_enhanced_plan
        )
        # Get model and generate hypothesis
        model = get_planning_model(config)

        # Use system message to enforce JSON output
        from langchain_core.messages import SystemMessage

        messages = [SystemMessage(content=formatted_prompt)]

        # Generate hypothesis with JSON format enforcement
        response = await model.ainvoke(messages)
        final_evaluation = (
            response.content if hasattr(response, "content") else str(response)
        )
        state.final_evaluation = final_evaluation
        # Print evaluation results description and content
        print("\n" + "=" * 50)
        print("📊 Experimental Plan Evaluation Results")
        print("=" * 50)
        print(
            "The following is a detailed evaluation report based on five core performance dimensions and eight expert evaluation criteria:"
        )
        print("-" * 50)
        print(f"{final_evaluation}")
        final_evaluation = final_evaluation + "\n\n" + state.original_question
        uuid = config["configurable"]["uuid"]
        # 保存文件并将路径存储到state中
        evaluation_file_path = save_planning_report(
            final_evaluation, uuid, "evaluation"
        )
        state.evaluation_report_path = evaluation_file_path
        return state
    except Exception as e:
        print(f"⚠️ Tool selection failed: {str(e)}")
        print("  Using default tool set as fallback")
        return state


async def stage7_task_execution(
    state: ImprovedCellState, config: RunnableConfig
) -> ImprovedCellState:
    """
    Stage 7: Task Execution.
    Extract tasks from the planning and execute them using available tools.
    """
    print("\n" + "=" * 60)
    print("🚀 STAGE 7: TASK EXECUTION")
    print("=" * 60)

    def _build_plan_text_from_steps(steps: List[Any]) -> str:
        """Convert structured plan steps into textual format for downstream models."""
        lines: List[str] = []
        for idx, step in enumerate(steps, 1):
            if isinstance(step, PlanStep):
                step_data = step.model_dump()
            elif isinstance(step, dict):
                step_data = step
            else:
                step_data = {}
            title = step_data.get("title") or step_data.get("name") or f"Step {idx}"
            objective = step_data.get("objective") or step_data.get("description") or ""
            tools = step_data.get("tools") or step_data.get("toolchain") or step_data.get("recommended_tools") or []
            if isinstance(tools, list):
                tools_text = ", ".join(str(t) for t in tools if t)
            else:
                tools_text = str(tools)
            parts = [f"{idx}. {title}"]
            if objective:
                parts.append(f"Objective: {objective}")
            if tools_text:
                parts.append(f"Tools: {tools_text}")
            notes = step_data.get("notes") or step_data.get("details") or step_data.get("commentary")
            if notes:
                parts.append(f"Notes: {notes}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    try:
        # Import required modules for task execution
        from usecases.immunity.graph.task_executor import (
            execute_task_list,
            task_decomposition_node,
        )

        # Step 1: Task decomposition - extract tasks from planning
        print("\n📋 Step 1: Task Decomposition")
        # 保存原始计划步骤，用于后续检测计划是否被修改
        original_plan_steps = state.plan_step_details.copy() if state.plan_step_details else []
        original_plan_text = state.final_enhanced_plan
        
        state = await task_decomposition_node(state, config)

        if not state.decomposed_tasks:
            print("⚠️ No tasks extracted from planning")
            return state

        print("✅ Extracted {len(state.decomposed_tasks)} tasks from planning")

        # 推送结构化计划摘要到前端
        sse_streamer = None
        plan_id = None
        if config and "configurable" in config:
            configurable = config["configurable"]
            sse_streamer = configurable.get("sse_streamer")
            plan_id = configurable.get("uuid") or configurable.get("thread_id")
        current_plan_summary: Dict[str, Any] | None = None
        if sse_streamer and state.plan_step_details:
            try:
                plan_summary_payload = {
                    "planId": str(plan_id) if plan_id else None,
                    "totalSteps": len(state.plan_step_details),
                    "originalQuestion": state.original_question,
                    "planText": state.final_enhanced_plan,
                    "steps": [step.model_dump() for step in state.plan_step_details],
                }
                current_plan_summary = plan_summary_payload
                sse_streamer.send_plan_summary(plan_summary_payload)
                print("📤 Sent plan summary to frontend via SSE")

                confirmation_timestamp = datetime.utcnow().isoformat()
                sse_streamer.send_plan_confirmation_request(plan_summary_payload)
                confirmation_request = {
                    "type": "plan_confirmation_request",
                    "planId": plan_summary_payload["planId"],
                    "plan": plan_summary_payload,
                    "timestamp": confirmation_timestamp,
                }
                confirmation_event_id = str(confirmation_timestamp)
                confirmation_request["timestamp"] = confirmation_event_id
                # 复合键：session_id + event_name
                session_id = None
                try:
                    if config and "configurable" in config:
                        session_id = config["configurable"].get("session_id")
                except Exception:
                    session_id = None
                composite_event_name = f"{(session_id or 'no-session')}:{confirmation_event_id}"
                sse_streamer.push_action_request({**confirmation_request, "event_name": composite_event_name, "session_id": session_id})
                if hasattr(sse_streamer, "persist_plan_state"):
                    initial_execution_state = {
                        (step.get("step_id") or step.get("stepId") or f"step-{idx + 1}"):
                        {"status": step.get("status", "pending")}
                        for idx, step in enumerate(plan_summary_payload["steps"])
                    }
                    await sse_streamer.persist_plan_state(
                        plan_summary=plan_summary_payload,
                        execution_state=initial_execution_state,
                        confirmation={"status": "pending"},
                    )
                user_plan_response = await sse_streamer.wait_for_action_response(timeout=600, event_name=composite_event_name, session_id=session_id)

                if user_plan_response:
                    try:
                        status = user_plan_response.get("status")
                        updated_plan = user_plan_response.get("plan") or {}
                        updated_steps_raw = (
                            user_plan_response.get("steps")
                            or updated_plan.get("steps")
                            or []
                        )
                        updated_steps: List[PlanStep] = []
                        for idx, step_payload in enumerate(updated_steps_raw, 1):
                            try:
                                if isinstance(step_payload, PlanStep):
                                    updated_steps.append(step_payload)
                                elif isinstance(step_payload, dict):
                                    updated_steps.append(PlanStep(**step_payload))
                            except Exception as step_parse_err:
                                print(
                                    f"⚠️ Failed to parse user-provided plan step {idx}: {step_parse_err}"
                                )
                        updated_plan_text = (
                            user_plan_response.get("planText")
                            or updated_plan.get("planText")
                            or _build_plan_text_from_steps(updated_steps)
                            or state.final_enhanced_plan
                        )
                        updated_plan_summary = {
                            "planId": plan_summary_payload.get("planId"),
                            "totalSteps": len(updated_steps) if updated_steps else len(state.plan_step_details),
                            "originalQuestion": state.original_question,
                            "planText": updated_plan_text,
                            "steps": [
                                step.model_dump() if isinstance(step, PlanStep) else step
                                for step in (updated_steps or state.plan_step_details)
                            ],
                        }

                        if status == "confirmed":
                            state.plan_step_details = updated_steps or state.plan_step_details
                            state.final_enhanced_plan = updated_plan_text
                            state.approved_plan = {
                                "planId": plan_summary_payload["planId"],
                                "planText": updated_plan_text,
                                "steps": [step.model_dump() for step in state.plan_step_details],
                                "confirmed_by_user": True,
                            }
                            state.plan_confirmation_status = "confirmed"
                            print("✅ Plan updated with user confirmation")
                        elif status == "rejected":
                            state.plan_confirmation_status = "rejected"
                            print("⚠️ User rejected the plan; continuing with original plan")
                        else:
                            state.plan_confirmation_status = status or "acknowledged"
                            if updated_steps or updated_plan_text != state.final_enhanced_plan:
                                state.plan_step_details = updated_steps or state.plan_step_details
                                state.final_enhanced_plan = updated_plan_text
                                state.approved_plan = {
                                    "planId": plan_summary_payload["planId"],
                                    "planText": updated_plan_text,
                                    "steps": [step.model_dump() for step in state.plan_step_details],
                                    "confirmed_by_user": False,
                                }
                                print("ℹ️ Applied user-provided plan adjustments without explicit confirmation")
                        current_plan_summary = updated_plan_summary
                        if hasattr(sse_streamer, "persist_plan_state"):
                            await sse_streamer.persist_plan_state(
                                plan_summary=updated_plan_summary,
                                confirmation={
                                    "status": state.plan_confirmation_status,
                                    "approved_plan": state.approved_plan,
                                },
                            )
                    except Exception as e:
                        print(f"⚠️ Failed to process user plan response: {e}")
                        state.plan_confirmation_status = "error"
                else:
                    # 超时后自动确认计划
                    print("⌛ Plan confirmation timed out; auto-confirming plan")
                    state.plan_step_details = state.plan_step_details
                    state.final_enhanced_plan = state.final_enhanced_plan
                    state.approved_plan = {
                        "planId": plan_summary_payload["planId"],
                        "planText": state.final_enhanced_plan,
                        "steps": [step.model_dump() for step in state.plan_step_details],
                        "confirmed_by_user": False,  # 标记为自动确认
                    }
                    state.plan_confirmation_status = "auto_confirmed"
                    print("✅ Plan auto-confirmed due to timeout")
                    if hasattr(sse_streamer, "persist_plan_state") and current_plan_summary:
                        await sse_streamer.persist_plan_state(
                            plan_summary=current_plan_summary,
                            confirmation={"status": "auto_confirmed"},
                        )
            except Exception as e:
                print(f"⚠️ Failed to send plan summary via SSE: {e}")
        
        # 如果没有SSE流处理器，使用控制台进行计划确认（逐条确认）
        if not sse_streamer and state.plan_step_details:
            try:
                print("\n" + "=" * 70)
                print("Plan Confirmation (Console Mode)")
                print("=" * 70)
                print(f"Original Question: {state.original_question}")
                print(f"Total Plan Steps: {len(state.plan_step_details)}")
                print("=" * 70)
                
                confirmed_steps: List[PlanStep] = []
                rejected = False
                
                # 逐条确认每个步骤
                for idx, step in enumerate(state.plan_step_details, 1):
                    step_confirmed = await _confirm_plan_step_console(step, idx, len(state.plan_step_details))
                    
                    if step_confirmed is None:
                        # User rejected entire plan
                        print("\nEntire plan has been rejected")
                        rejected = True
                        break
                    elif step_confirmed:
                        # User confirmed current step
                        confirmed_steps.append(step_confirmed)
                        print(f"Step {idx} confirmed")
                    else:
                        # User rejected/skipped current step (False case)
                        print(f"Step {idx} rejected/skipped, will not be executed")
                
                if rejected:
                    # User rejected plan, ask if continue with original plan
                    print("\n" + "=" * 70)
                    print("Plan has been rejected")
                    print("=" * 70)
                    while True:
                        choice = input("Continue with original plan? (y/n): ").strip().lower()
                        if choice in ['y', 'yes']:
                            state.plan_confirmation_status = "auto_confirmed"
                            state.approved_plan = {
                                "planId": str(plan_id) if plan_id else None,
                                "planText": state.final_enhanced_plan,
                                "steps": [step.model_dump() for step in state.plan_step_details],
                                "confirmed_by_user": False,
                            }
                            print("Continuing with original plan")
                            break
                        elif choice in ['n', 'no']:
                            state.plan_confirmation_status = "rejected"
                            print("Plan execution cancelled")
                            break
                        else:
                            print("Please enter y or n")
                else:
                    # 更新确认的步骤
                    if confirmed_steps:
                        state.plan_step_details = confirmed_steps
                        state.final_enhanced_plan = _build_plan_text_from_steps(confirmed_steps)
                        state.plan_confirmation_status = "confirmed"
                        state.approved_plan = {
                            "planId": str(plan_id) if plan_id else None,
                            "planText": state.final_enhanced_plan,
                            "steps": [step.model_dump() for step in confirmed_steps],
                            "confirmed_by_user": True,
                        }
                        print("\n" + "=" * 70)
                        print(f"Plan confirmation completed: {len(confirmed_steps)}/{len(state.plan_step_details)} steps confirmed")
                        print("=" * 70)
                    else:
                        state.plan_confirmation_status = "rejected"
                        print("\nNo steps confirmed, plan execution cancelled")
            except Exception as e:
                print(f"Error: Console plan confirmation failed: {e}")
                import traceback
                traceback.print_exc()
                # 失败时使用原始计划
                state.plan_confirmation_status = "auto_confirmed"
                state.approved_plan = {
                    "planId": str(plan_id) if plan_id else None,
                    "planText": state.final_enhanced_plan,
                    "steps": [step.model_dump() for step in state.plan_step_details],
                    "confirmed_by_user": False,
                }

        # 如果用户提供了新的计划文本，确保后续使用最新内容
        if state.plan_step_details and not state.final_enhanced_plan:
            state.final_enhanced_plan = _build_plan_text_from_steps(state.plan_step_details)

        # 如果计划在确认过程中被修改（例如某些步骤被拒绝），需要重新进行任务分解
        # 以确保被拒绝的步骤不会出现在任务列表中
        plan_was_modified = False
        
        # 检查计划是否被修改
        if state.plan_confirmation_status in ["confirmed", "auto_confirmed"]:
            # 比较步骤数量和步骤ID，判断是否有步骤被拒绝
            current_steps_count = len(state.plan_step_details) if state.plan_step_details else 0
            original_steps_count = len(original_plan_steps) if original_plan_steps else 0
            
            # 如果步骤数量减少，说明有步骤被拒绝
            if current_steps_count < original_steps_count:
                plan_was_modified = True
            # 或者比较步骤ID列表，看是否有变化
            elif current_steps_count == original_steps_count and original_plan_steps:
                current_step_ids = {step.step_id for step in state.plan_step_details}
                original_step_ids = {step.step_id for step in original_plan_steps}
                if current_step_ids != original_step_ids:
                    plan_was_modified = True
            # 或者比较计划文本
            elif state.final_enhanced_plan != original_plan_text:
                plan_was_modified = True
        
        # 如果计划被修改，重新进行任务分解以确保只包含确认的步骤
        if plan_was_modified:
            # 保存当前的任务列表（用于对比）
            old_decomposed_tasks = state.decomposed_tasks.copy() if state.decomposed_tasks else []
            # 重新分解任务（基于确认后的计划）
            state = await task_decomposition_node(state, config)
            if state.decomposed_tasks:
                # Task decomposition completed
                if len(state.decomposed_tasks) < len(old_decomposed_tasks):
                    removed_count = len(old_decomposed_tasks) - len(state.decomposed_tasks)
                    # Tasks related to rejected steps have been removed
            else:
                if not old_decomposed_tasks:
                    return state
                # Use previous task list
                    state.decomposed_tasks = old_decomposed_tasks

        # Step 2: Execute tasks using execute_task_list
        print("\n🔧 Step 2: Task Execution")
        try:
            # 检查config中是否有UI交互回调和SSE流处理器
            ui_callback = None
            ui_interaction_mode = False
            sse_streamer = None

            if config and "configurable" in config:
                configurable = config["configurable"]
                ui_callback = configurable.get("ui_callback")
                ui_interaction_mode = configurable.get("ui_interaction_mode", False)
                sse_streamer = configurable.get("sse_streamer")
                configurable["plan_steps"] = [
                    step.model_dump() if isinstance(step, PlanStep) else step
                    for step in state.plan_step_details
                ]
                configurable["plan_id"] = state.approved_plan.get("planId") if state.approved_plan else plan_id

            if sse_streamer:
                try:
                    sse_streamer.send_execution_progress(
                        {
                            "planId": configurable.get("plan_id") if configurable else plan_id,
                            "stepId": "initializing",
                            "status": "running",
                            "message": "Initializing task executor",
                        }
                    )
                except Exception as progress_err:
                    print(f"[TaskExecution] Warning: failed to emit initialization progress: {progress_err}")

            # 调用execute_task_list，传递UI交互参数和SSE流处理器
            # 如果state中有merged_csv_result_path，将其作为初始文件路径传递
            initial_file_path = state.merged_csv_result_path if state.merged_csv_result_path else None
            results = await execute_task_list(
                state.decomposed_tasks,
                config,
                ui_interaction_mode=ui_interaction_mode,
                ui_callback=ui_callback,
                sse_streamer=sse_streamer,
                initial_file_path=initial_file_path,
            )
        except Exception as e:
            import traceback

            print(f"Error: Task execution failed: {e}")
            # Return empty results to continue workflow
            results = []

        # Step 3: Display execution results
        print("\nTask Execution Results:")
        print("=" * 50)
        for i, result in enumerate(results, 1):
            print(f"\nTask {i}:")
            print(f"  Success: {result['success']}")
            if result["result"]:
                print(f"  Result: {result['result'][:100]}...")
            if result.get('tools_called'):
                print(f"  Tools: {[tc['tool_name'] for tc in result['tools_called']]}")

        # Save execution summary to state (optional)
        execution_summary = f"Executed {len(results)} tasks, successful: {sum(1 for r in results if r['success'])}"
        print(f"\n{execution_summary}")
        
        # 保存最终合并的CSV文件路径到state
        merged_csv_path = None
        for result in results:
            if "merged_csv_path" in result and result["merged_csv_path"]:
                merged_csv_path = result["merged_csv_path"]
                break
        
        if merged_csv_path:
            state.merged_csv_result_path = merged_csv_path

        return state

    except Exception as e:
        print(f"⚠️ Task execution failed: {str(e)}")
        print("  Continuing workflow without task execution")
        return state


def sync_wrapper_stage2(
    state: ImprovedCellState, config: RunnableConfig
) -> ImprovedCellState:
    """
    Synchronous wrapper for async stage2_immunology_retrieval.

    Args:
        state: Current workflow state
        config: Runtime configuration

    Returns:
        Updated state after retrieval
    """
    return asyncio.run(stage2_immunology_retrieval(state, config))


def sync_wrapper_stage3(
    state: ImprovedCellState, config: RunnableConfig
) -> ImprovedCellState:
    """
    Synchronous wrapper for async stage3_deep_research_analysis.

    Args:
        state: Current workflow state
        config: Runtime configuration

    Returns:
        Updated state after deep research analysis
    """
    return asyncio.run(stage3_deep_research_analysis(state, config))


def sync_wrapper_stage4(
    state: ImprovedCellState, config: RunnableConfig
) -> ImprovedCellState:
    """
    Synchronous wrapper for async stage4_hypothesis_generation.

    Args:
        state: Current workflow state with research findings
        config: Runtime configuration

    Returns:
        Updated state with generated hypothesis
    """
    return asyncio.run(stage4_hypothesis_generation(state, config))


def sync_wrapper_stage7(
    state: ImprovedCellState, config: RunnableConfig
) -> ImprovedCellState:
    """
    Synchronous wrapper for async stage7_task_execution.

    Args:
        state: Current workflow state with planning
        config: Runtime configuration

    Returns:
        Updated state after task execution
    """
    return asyncio.run(stage7_task_execution(state, config))


def route_after_decomposition(
    state: ImprovedCellState
) -> str:
    """
    在查询分解后的路由函数
    
    根据 skip_planning 标志决定：
    - True: 跳过研究阶段，直接到计划评估
    - False: 正常流程（检索 → 研究 → 假设 → 计划）
    
    Returns:
        下一个节点的名称
    """
    if state.skip_planning:
        return "evaluate_planning"
    else:
        return "immunology_retrieval"


def build_improved_graph() -> StateGraph:
    """
    Build the improved workflow graph with plan detection and reordered stages.

    Key improvements:
    1. Plan detection: Automatically detects if user provides a plan directly
    2. Research and hypothesis generation happen BEFORE planning,
       ensuring plans are scientifically grounded.
    3. Conditional routing: Skips research stages if user provides plan directly.

    Returns:
        Compiled StateGraph with reordered stages and plan detection
    """
    # Create graph with improved state
    graph = StateGraph(ImprovedCellState)

    # === 新增：计划检测节点（入口节点）===
    graph.add_node("plan_detection", stage0_plan_detection)

    # === 原有节点 ===
    # Stage 1: Query Decomposition (synchronous)
    graph.add_node("query_decomposition", stage1_query_decomposition)

    # Stage 2: Immunology Retrieval (async with sync wrapper)
    graph.add_node("immunology_retrieval", sync_wrapper_stage2)

    # Stage 3: Deep Research (async with sync wrapper)
    graph.add_node("deep_research", sync_wrapper_stage3)

    # Stage 4: Hypothesis Generation (async with sync wrapper)
    graph.add_node("hypothesis_generation", sync_wrapper_stage4)

    # Stage 5: Research-Informed Planning (synchronous)
    graph.add_node("research_informed_planning", stage5_research_informed_planning)

    # Stage 6: Evaluate Planning (synchronous)
    graph.add_node("evaluate_planning", stage6_evaluate_planning)

    # Stage 7: Task Execution (async with sync wrapper)
    graph.add_node("task_execution", sync_wrapper_stage7)

    # === 修改入口点 ===
    graph.set_entry_point("plan_detection")  # 从计划检测开始

    # === 修改边连接 ===
    # 计划检测 → 查询分解
    graph.add_edge("plan_detection", "query_decomposition")

    # 查询分解后 → 条件路由
    graph.add_conditional_edges(
        "query_decomposition",
        route_after_decomposition,  # 路由函数
        {
            "immunology_retrieval": "immunology_retrieval",  # 正常流程
            "evaluate_planning": "evaluate_planning",        # 跳过计划生成
        }
    )

    # === 正常流程的边（保持不变）===
    graph.add_edge("immunology_retrieval", "deep_research")
    graph.add_edge("deep_research", "hypothesis_generation")
    graph.add_edge("hypothesis_generation", "research_informed_planning")
    graph.add_edge("research_informed_planning", "evaluate_planning")

    # === 两条路径都汇聚到 evaluate_planning ===
    graph.add_edge("evaluate_planning", "task_execution")
    graph.add_edge("task_execution", END)

    return graph
