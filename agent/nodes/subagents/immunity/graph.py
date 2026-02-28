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
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from state import GlobalState
from utils.llm_factory import (
    create_reasoning_advanced_llm,
    create_reasoning_llm,
    create_bioinformatics_llm
)
from .state import ImmunityState
from .prompts import ImmunityPrompts

# Import deep_research subgraph
from nodes.subagents.deep_research.deep_researcher import (
    deep_researcher,
    get_default_config as get_deep_research_config,
)

# Add agent directory to path
import sys
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))


# ===================== Helper Functions =====================

def _load_tools_json() -> str:
    """
    Load tool information (JSON format)
    
    Returns:
        JSON string of tool information
    """
    mcp_tools_path = agent_dir / "config" / "mcp_tools.json"
    
    try:
        if mcp_tools_path.exists():
            with open(mcp_tools_path, 'r', encoding='utf-8') as f:
                tools_data = json.load(f)
                return json.dumps(tools_data, ensure_ascii=False, indent=2)
        else:
            print(f"⚠️ mcp_tools.json does not exist: {mcp_tools_path}")
            return "[]"
    except Exception as e:
        print(f"⚠️ Failed to load tool information: {e}")
        return "[]"


def _save_report(content: str, report_type: str, sandbox_dir: str) -> str:
    """
    Save report to file
    
    Args:
        content: Report content
        report_type: Report type (retrieval, deep_research, hypothesis, planning, evaluation)
        sandbox_dir: Sandbox directory
    
    Returns:
        Saved file path
    """
    try:
        reports_dir = Path(sandbox_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"{report_type}_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"📄 {report_type} report saved to: {report_file}")
        return str(report_file)
    except Exception as e:
        print(f"⚠️ Failed to save report: {e}")
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
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    
    for pattern in json_block_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Try extracting the first JSON object
    json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # If all fail, return empty dictionary
    return {}


# ===================== Stage 1: Query Decomposition Node =====================

def query_decomposition_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 1: Query Decomposition Node
    
    Decompose user question into optimized sub-questions
    """
    print("\n" + "=" * 60)
    print("📝 STAGE 1: Query Decomposition")
    print("=" * 60)
    
    if not state.original_question:
        print("⚠️ No original question, skipping query decomposition")
        return state
    
    llm = create_bioinformatics_llm()
    if not llm:
        print("⚠️ LLM unavailable, using original question")
        state.optimized_questions = [state.original_question]
        state.optimized_question = state.original_question
        return state
    
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_core.output_parsers import JsonOutputParser
        
        tools_info = _load_tools_json()
        
        # Use reference project's QUERY_EXPANSION_PROMPT
        query_expansion_prompt = ImmunityPrompts.QUERY_EXPANSION_PROMPT.format(
            tools_info=tools_info,
            query=state.original_question
        )
        
        # Define output schema
        class QueryExpansion(BaseModel):
            queries: List[str]
        
        output_parser = JsonOutputParser(pydantic_object=QueryExpansion)
        structured_llm = llm.with_structured_output(QueryExpansion)
        
        messages = [
            SystemMessage(content="You are a professional query optimization expert capable of decomposing complex research queries into multiple optimized sub-queries."),
            HumanMessage(content=query_expansion_prompt)
        ]
        
        response = structured_llm.invoke(messages)
        
        if hasattr(response, 'queries'):
            state.optimized_questions = response.queries
        elif isinstance(response, dict):
            state.optimized_questions = response.get('queries', [state.original_question])
        else:
            state.optimized_questions = [state.original_question]
        
        if not state.optimized_questions:
            state.optimized_questions = [state.original_question]
        
        state.optimized_question = "; ".join(state.optimized_questions)
        
        print(f"✅ Query decomposition completed")
        print(f"  - Original query: {state.original_question[:100]}...")
        print(f"  - Optimized queries count: {len(state.optimized_questions)}")
        for i, q in enumerate(state.optimized_questions, 1):
            print(f"    {i}. {q[:80]}...")
        
    except Exception as e:
        print(f"⚠️ Query decomposition failed: {e}")
        state.optimized_questions = [state.original_question]
        state.optimized_question = state.original_question
    
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
    print("\n" + "=" * 60)
    print("📚 STAGE 2: Immunology Retrieval (Parallel Retrieval)")
    print("=" * 60)
    
    if not state.optimized_questions:
        print("⚠️ No optimized queries, skipping retrieval")
        return state
    
    try:
        from .retrieval_tools import parallel_retrieval_sync
        
        # Parallel execution of three retrieval methods
        print("🔍 Executing three retrieval methods in parallel:")
        print("  1. Qdrant vector database retrieval")
        print("  2. Tavily API web search")
        print("  3. Web retrieval (multiple sources)")
        
        retrieval_results = parallel_retrieval_sync(
            queries=state.optimized_questions,
            original_question=state.original_question,
            k_per_query=10
        )
        
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
{chr(10).join([f"{i+1}. **{doc.get('title', 'N/A')}** (Relevance: {doc.get('relevance_score', 0):.2f})" + chr(10) + f"   - Source: {doc.get('source', 'N/A')}" + chr(10) + f"   - Summary: {doc.get('summary', '')[:200]}..." for i, doc in enumerate(state.retrieval_docs[:10])])}

Main Citations (Top 10):
{chr(10).join([f"{i+1}. {cite.get('author', 'N/A')} et al. ({cite.get('year', 'N/A')}). {cite.get('title', 'N/A')}. *{cite.get('journal', 'N/A')}*" + (f" DOI: {cite.get('doi', '')}" if cite.get('doi') else "") for i, cite in enumerate(state.citations[:10])])}
"""
        
        print(f"✅ Retrieval completed")
        print(f"  - Retrieved documents: {len(state.retrieval_docs)}")
        print(f"  - Citations: {len(state.citations)}")
        print(f"  - Context length: {len(state.context)} characters")
        
        # Save retrieval report
        report_path = _save_report(retrieval_summary, "retrieval", state.sandbox_dir)
        state.retrieval_report_path = report_path
        
    except Exception as e:
        print(f"⚠️ Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        # Use empty context on failure
        state.context = ""
        state.retrieval_docs = []
        state.citations = []
    
    return state


# ===================== Stage 3: Deep Research Node (using deep_research subgraph) =====================

def deep_research_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 3: Deep Research Node
    
    Uses the deep_research subgraph to conduct in-depth analysis of research questions.
    The deep_research subgraph provides multi-step research with web search and synthesis.
    """
    print("\n" + "=" * 60)
    print("🔬 STAGE 3: Deep Research Analysis (via deep_research subgraph)")
    print("=" * 60)
    
    if not state.original_question:
        print("⚠️ No original question, skipping deep research")
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
        
        print(f"  📋 Research question: {state.original_question[:100]}...")
        print(f"  🔍 Using deep_research subgraph for multi-step analysis...")
        
        # Get deep_research subgraph configuration
        dr_config = get_deep_research_config(
            thread_id=f"immunity_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            max_researcher_iterations=3,  # Limit iterations for efficiency
            max_concurrent_research_units=2,
            max_react_tool_calls=6,
        )
        
        # Prepare input for deep_research subgraph
        research_input = {
            "messages": [{"role": "user", "content": research_question}]
        }
        
        # Run deep_research subgraph asynchronously
        async def run_deep_research_async():
            from langgraph.checkpoint.memory import MemorySaver
            from nodes.subagents.deep_research.deep_researcher import deep_researcher_builder
            
            # Compile with memory checkpointing
            graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
            return await graph.ainvoke(research_input, dr_config)
        
        # Execute async function
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new event loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, run_deep_research_async())
                    result = future.result(timeout=300)  # 5 minute timeout
            else:
                result = loop.run_until_complete(run_deep_research_async())
        except RuntimeError:
            # No event loop, create one
            result = asyncio.run(run_deep_research_async())
        
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
            
            print(f"✅ Deep research completed")
            print(f"  - Final report length: {len(final_report)} characters")
            print(f"  - Research brief length: {len(research_brief)} characters")
            print(f"  - Notes count: {len(notes)}")
            print(f"  - Confidence: {state.research_confidence:.1f}%")
            
            # Save research report
            report_path = _save_report(state.research_summary, "deep_research", state.sandbox_dir)
        else:
            print("⚠️ Deep research returned no results, using fallback")
            # Fallback to simple context-based research
            state.research_summary = f"""
<research_findings>
    <research_finding>
        Research Topic: {state.original_question}
        
        Based on retrieval context:
        {state.context[:2000] if state.context else 'No context available'}
        
        Optimized queries:
        {chr(10).join([f"- {q}" for q in state.optimized_questions[:5]])}
    </research_finding>
</research_findings>
"""
            state.research_confidence = 50.0
    
    except Exception as e:
        print(f"⚠️ Deep research failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: create research summary from available context
        state.research_summary = f"""
<research_findings>
    <research_finding>
        Research Topic: {state.original_question}
        
        Note: Deep research subgraph encountered an error. Using available context.
        
        Context from retrieval:
        {state.context[:2000] if state.context else 'No context available'}
        
        Optimized queries:
        {chr(10).join([f"- {q}" for q in state.optimized_questions[:5]])}
    </research_finding>
</research_findings>
"""
        state.research_confidence = 40.0
    
    return state


# ===================== Stage 4: Hypothesis Generation Node =====================

def hypothesis_generation_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 4: Hypothesis Generation Node
    
    Generate testable hypotheses based on research results
    """
    print("\n" + "=" * 60)
    print("🧬 STAGE 4: Hypothesis Generation")
    print("=" * 60)
    
    if not state.research_summary:
        print("⚠️ No research results, skipping hypothesis generation")
        return state
    
    llm = create_bioinformatics_llm()
    if not llm:
        print("⚠️ LLM unavailable, skipping hypothesis generation")
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
            research_findings=state.research_summary,
            context=context,
            question=question
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
        message_lengths = [
            len(msg.content) if hasattr(msg, "content") and msg.content is not None else len(str(msg))
            for msg in messages
        ]
        print("  🔍 LLM invoke diagnostics:")
        print(f"    - llm: {llm_info}")
        print(f"    - env LLM_TIMEOUT: {os.getenv('LLM_TIMEOUT')}")
        print(f"    - prompt stats: {prompt_stats}")
        print(f"    - messages: count={len(messages)}, lengths={message_lengths}")
        print(f"    - prompt length: {len(hypothesis_prompt)} characters")

        # Directly invoke LLM, then parse JSON
        start_time = time.perf_counter()
        try:
            response = llm.invoke(messages)
            elapsed = time.perf_counter() - start_time
            print(f"  ⏱️ LLM invoke completed in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            print(f"  ⚠️ LLM invoke failed after {elapsed:.2f}s: {type(e).__name__}")
            raise
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Use JsonOutputParser to parse response
        try:
            hypothesis_data = output_parser.parse(response_content)
            if not isinstance(hypothesis_data, dict):
                hypothesis_data = {}
        except Exception as e:
            print(f"⚠️ JSON parsing failed, attempting direct parsing: {e}")
            # Try extracting JSON from response
            import json
            import re
            # Try extracting JSON code block or direct parsing
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                try:
                    hypothesis_data = json.loads(json_match.group())
                except:
                    hypothesis_data = {}
            else:
                hypothesis_data = {}
        
        if hypothesis_data and hypothesis_data.get("statement"):
            state.hypothesis = hypothesis_data
            state.hypothesis_confidence = float(hypothesis_data.get("confidence_score", 70.0))
            
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
            
            print(f"✅ Hypothesis generation completed")
            print(f"  - Hypothesis: {hypothesis_data.get('statement', 'Not specified')[:100]}...")
            print(f"  - Confidence: {state.hypothesis_confidence:.1f}%")
            print(f"  - Innovation level: {hypothesis_data.get('innovation_level', 'moderate')}")
            
            # Save hypothesis report
            report_path = _save_report(state.hypothesis_summary, "hypothesis", state.sandbox_dir)
        else:
            print("⚠️ Unable to parse hypothesis results")
    
    except Exception as e:
        print(f"⚠️ Hypothesis generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    return state


# ===================== Stage 5: Plan Generation Node ⭐ =====================

def planning_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 5: Planning Node ⭐
    
    Generate executable experimental plan based on research results and hypotheses
    """
    print("\n" + "=" * 60)
    print("🔧 STAGE 5: Research-Driven Plan Generation ⭐")
    print("=" * 60)
    
    llm = create_bioinformatics_llm()
    if not llm:
        print("⚠️ LLM unavailable, using simple plan generation")
        return _generate_simple_plan(state)
    
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        
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
        optimized_questions_text = "\n\n".join(format_queries) if format_queries else "None"
        
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
            citations_json=citations_json
        )
        
        # ZhipuAI requires at least one user message, so send prompt as user message
        messages = [HumanMessage(content=planning_prompt)]

        llm_info = {
            "type": type(llm).__name__,
            "model": getattr(llm, "model", getattr(llm, "model_name", None)),
            "temperature": getattr(llm, "temperature", None),
            "timeout": getattr(llm, "timeout", None),
            "max_retries": getattr(llm, "max_retries", None),
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
        message_lengths = [
            len(msg.content) if hasattr(msg, "content") and msg.content is not None else len(str(msg))
            for msg in messages
        ]
        print("  🔍 LLM invoke diagnostics:")
        print(f"    - llm: {llm_info}")
        print(f"    - env LLM_TIMEOUT: {os.getenv('LLM_TIMEOUT')}")
        print(f"    - prompt stats: {prompt_stats}")
        print(f"    - messages: count={len(messages)}, lengths={message_lengths}")
        print(f"    - prompt length: {len(planning_prompt)} characters")

        start_time = time.perf_counter()
        try:
            response = llm.invoke(messages)
            elapsed = time.perf_counter() - start_time
            print(f"  ⏱️ LLM invoke completed in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            print(f"  ⚠️ LLM invoke failed after {elapsed:.2f}s: {type(e).__name__}")
            raise
        plan_content = response.content.strip() if hasattr(response, 'content') else str(response)
        
        state.final_enhanced_plan = plan_content
        state.research_informed_plan = plan_content
        state.generated_plan = plan_content
        
        print(f"✅ Plan generation completed")
        print(f"  - Plan length: {len(plan_content)} characters")
        
        # Save plan report
        report_path = _save_report(plan_content, "planning", state.sandbox_dir)
        
    except Exception as e:
        print(f"⚠️ Plan generation failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback solution
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
    print("\n" + "=" * 60)
    print("📊 STAGE 6: Plan Evaluation")
    print("=" * 60)
    
    if not state.final_enhanced_plan:
        print("⚠️ No plan to evaluate")
        state.final_evaluation = "No plan generated, cannot evaluate"
        return state
    
    # If user-provided plan, skip evaluation
    if state.is_user_provided_plan:
        print("📋 User-provided plan, skipping automatic evaluation")
        state.final_evaluation = f"""User-provided execution plan:

{state.final_enhanced_plan}

---
Evaluation Note: This plan was directly provided by the user, automatic evaluation step has been skipped.
"""
        report_path = _save_report(state.final_evaluation, "evaluation", state.sandbox_dir)
        return state
    
    llm = create_bioinformatics_llm()
    if not llm:
        print("⚠️ LLM unavailable, skipping evaluation")
        state.final_evaluation = "LLM unavailable, cannot perform evaluation"
        return state
    
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # Use reference project's EVALUATE_PLANNING_PROMPT
        evaluation_prompt = ImmunityPrompts.EVALUATE_PLANNING_PROMPT.format(
            plan=state.final_enhanced_plan
        )
        
        # ZhipuAI requires at least one user message, so send prompt as user message
        messages = [HumanMessage(content=evaluation_prompt)]
        
        response = llm.invoke(messages)
        evaluation_content = response.content.strip() if hasattr(response, 'content') else str(response)
        
        state.final_evaluation = evaluation_content
        
        print(f"✅ Evaluation completed")
        print(f"  - Evaluation report length: {len(evaluation_content)} characters")
        
        # Save evaluation report
        full_evaluation = evaluation_content + "\n\n" + state.original_question
        report_path = _save_report(full_evaluation, "evaluation", state.sandbox_dir)
        
    except Exception as e:
        print(f"⚠️ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        state.final_evaluation = f"Evaluation process error: {str(e)}"
    
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
    immunity_state = ImmunityState(
        original_question=global_state.user_input,
        subtasks=global_state.subtasks,
        parallel_task_groups=global_state.parallel_task_groups,
        sandbox_dir=global_state.sandbox_dir,
        parent_state=global_state
    )
    
    return immunity_state


def immunity_output_mapper(immunity_state: ImmunityState, global_state: GlobalState) -> GlobalState:
    """
    Map Immunity subgraph state back to main graph state
    
    Args:
        immunity_state: Immunity subgraph state
        global_state: Main graph global state
    
    Returns:
        Updated main graph state
    """
    # Store complete experimental plan to merged_result
    if not global_state.merged_result:
        global_state.merged_result = {}
    
    global_state.merged_result["immunity_plan"] = {
        "original_question": immunity_state.original_question,
        "optimized_questions": immunity_state.optimized_questions,
        "research_summary": immunity_state.research_summary,
        "hypothesis_summary": immunity_state.hypothesis_summary,
        "experimental_plan": immunity_state.final_enhanced_plan,
        "final_enhanced_plan": immunity_state.final_enhanced_plan,
        "plan_steps": immunity_state.plan_steps,
        "plan_summary": immunity_state.plan_summary,
        "evaluation": immunity_state.final_evaluation,
        "executable_plan": immunity_state.executable_plan
    }

    # Persist execution plan to global state for downstream logging/usage
    execution_plan = None
    if immunity_state.final_enhanced_plan:
        execution_plan = immunity_state.final_enhanced_plan
    elif immunity_state.generated_plan:
        execution_plan = immunity_state.generated_plan
    elif immunity_state.plan_summary:
        execution_plan = immunity_state.plan_summary

    if not execution_plan:
        if immunity_state.skip_planning:
            execution_plan = "PLAN_NOT_GENERATED: skip_planning is true"
        elif not immunity_state.original_question:
            execution_plan = "PLAN_NOT_GENERATED: original_question is empty"
        else:
            execution_plan = "PLAN_NOT_GENERATED: planning output is empty"

    global_state.execution_plan = execution_plan
    
    print(f"✅ Immunity subgraph completed: Generated complete experimental plan")
    print(f"  - Optimized queries count: {len(immunity_state.optimized_questions)}")
    print(f"  - Research confidence: {immunity_state.research_confidence:.1f}%")
    print(f"  - Hypothesis confidence: {immunity_state.hypothesis_confidence:.1f}%")
    print(f"  - Plan document length: {len(immunity_state.final_enhanced_plan)} characters")
    
    return global_state


# ===================== Build Immunity Subgraph =====================

def build_immunity_subgraph():
    """
    Build Immunity Agent subgraph
    
    Complete workflow:
    Query Decomposition → Retrieval → Deep Research → Hypothesis Generation → Planning ⭐ → Evaluation
    
    Returns:
        Compiled subgraph
    """
    graph = StateGraph(ImmunityState)
    
    # Add all nodes
    graph.add_node("query_decomposition", query_decomposition_node)  # Stage 1
    graph.add_node("retrieval", retrieval_node)  # Stage 2: Retrieval node
    graph.add_node("deep_research", deep_research_node)  # Stage 3
    graph.add_node("hypothesis_generation", hypothesis_generation_node)  # Stage 4
    graph.add_node("planning", planning_node)  # Stage 5 ⭐
    graph.add_node("evaluation", evaluation_node)  # Stage 6
    
    # Define flow rules
    graph.add_edge(START, "query_decomposition")
    graph.add_edge("query_decomposition", "retrieval")  # After query decomposition, enter retrieval
    graph.add_edge("retrieval", "deep_research")  # After retrieval, enter deep research
    graph.add_edge("deep_research", "hypothesis_generation")
    graph.add_edge("hypothesis_generation", "planning")
    graph.add_edge("planning", "evaluation")
    graph.add_edge("evaluation", END)
    
    return graph.compile()
