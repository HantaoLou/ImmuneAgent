"""
Simplified GeneralQA Subgraph - Core: Deep Research + X-Masters

Workflow design:
1. N0: Simple preprocessing (question classification, domain detection)
2. Deep Research: Deep knowledge retrieval and research
3. X-Masters: Multi-path reasoning + critique + synthesis selection
4. N8: Answer formatting and output

Core philosophy:
- Deep Research provides sufficient knowledge
- X-Masters handles multi-path reasoning and verification
- Redundant intermediate nodes removed for LLM flexibility

Usage:
    USE_SIMPLIFIED_GRAPH=true
"""

from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, START, END
import json
import os
import time

# Import state
from .state import GeneralQAState

# Import GlobalState to make it available for type resolution
# This is needed because GeneralQAState has Optional["GlobalState"] type hint
# and langgraph's StateGraph uses get_type_hints() which requires the type to be resolvable
try:
    from agent.state import GlobalState

    # Make GlobalState available in the state module's namespace for type resolution
    import sys

    state_module = sys.modules.get("agent.nodes.subagents.general_qa.state")
    if state_module and not hasattr(state_module, "GlobalState"):
        state_module.GlobalState = GlobalState
    # Rebuild the model to resolve forward references
    GeneralQAState.model_rebuild()
except ImportError:
    pass

# Import X-Masters
try:
    from agent.nodes.subagents.x_masters.graph import (
        build_graph as build_xmasters_graph,
    )

    XMASTERS_AVAILABLE = True
except ImportError:
    XMASTERS_AVAILABLE = False
    print("Warning: X-Masters not available")

# Import Deep Research
try:
    from agent.nodes.subagents.deep_research.deep_researcher import (
        graph as deep_research_graph,
    )

    DEEP_RESEARCH_AVAILABLE = True
except ImportError:
    DEEP_RESEARCH_AVAILABLE = False
    print("Warning: Deep Research not available")

# Import LLM factory
from agent.utils.llm_factory import create_bioinformatics_llm, is_llm_available

# Try importing langchain
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: langchain not available")

# Use llm_factory's availability check (which properly checks zhipuai)
LLM_AVAILABLE = is_llm_available()


# ===================== Configuration =====================

# Whether to enable Deep Research (enabled by default)
ENABLE_DEEP_RESEARCH = os.getenv("ENABLE_DEEP_RESEARCH", "true").lower() == "true"

# Number of X-Masters solvers
XMASTERS_NUM_SOLVERS = int(os.getenv("XMASTERS_NUM_SOLVERS", "3"))

# ===================== LLM Helper =====================

_llm_instance = None


def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        if not LLM_AVAILABLE:
            print(f"[graph_simplified] LLM unavailable, cannot create LLM")
            return None
        _llm_instance = create_bioinformatics_llm(temperature=0.2)
        print(f"[graph_simplified] LLM created: {type(_llm_instance).__name__}")
    return _llm_instance


def _call_llm(prompt: str, node_name: str = "") -> Optional[str]:
    llm = _get_llm()
    if llm is None:
        return None

    try:
        start_time = time.time()
        response = llm.invoke([HumanMessage(content=prompt)])
        elapsed = time.time() - start_time
        print(f"  [{node_name}] LLM call completed in {elapsed:.1f}s")

        if hasattr(response, "content"):
            return response.content.strip()
        return str(response).strip()
    except Exception as e:
        print(f"  [{node_name}] LLM error: {e}")
        return None


def _call_llm_simple(prompt: str, node_name: str = "") -> Optional[str]:
    return _call_llm(prompt, node_name)


def _parse_json(response: str) -> Optional[Dict]:
    """Parse JSON from response"""
    if not response:
        return None

    import re

    try:
        return json.loads(response)
    except:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None


# ===================== N0: Simple Preprocessing =====================

N0_PROMPT = """Analyze this biomedical question and extract key information.

Question:
{user_input}

Output JSON format:
{{
    "cleaned_text": "Cleaned question text",
    "question_type": "multiple_choice|true_false|calculation|explanation|short_answer",
    "answer_format": "single_letter|true/false|number|text|sequence",
    "options": ["Option A text", "Option B text", ...] or [] if no options,
    "domain": "genetics|immunology|biochemistry|bioinformatics|clinical|microbiology|general",
    "key_terms": ["keyword1", "keyword2", ...]
}}

Rules:
- True/False question → question_type: "true_false", answer_format: "true/false"
- Multiple choice → question_type: "multiple_choice", answer_format: "single_letter"
- Numerical calculation → question_type: "calculation", answer_format: "number"
- Extract actual option text, not just labels
"""


def n0_preprocess_node(state: GeneralQAState) -> GeneralQAState:
    """
    N0: Simple preprocessing
    Only performs question classification and key information extraction, no complex analysis
    """
    print("\n" + "=" * 60)
    print("N0: Simple Preprocessing (Simplified)")
    print("=" * 60)

    if not state.user_input or not state.user_input.strip():
        state.error_message = "Empty input"
        return state

    # Get prompt
    prompt = N0_PROMPT.format(user_input=state.user_input)

    # Call LLM
    response = _call_llm_simple(prompt, "N0")

    if response is None:
        state.error_message = "N0 LLM call failed"
        return state

    # Parse response
    result = _parse_json(response)

    if result is None:
        # Fallback
        state.cleaned_text = state.user_input
        state.question_type_label = "Multiple Choice"
        state.answer_format_label = "Single Choice"
        state.core_domains = ["general"]
        return state

    # Update state
    state.cleaned_text = result.get("cleaned_text", state.user_input)
    state.question_type_label = result.get("question_type", "Multiple Choice")
    state.answer_format_label = result.get("answer_format", "Single Choice")
    state.question_options = result.get("options", [])
    state.core_domains = [result.get("domain", "general")]
    state.core_keywords = result.get("key_terms", [])

    print(f"  [OK] Question type: {state.question_type_label}")
    print(f"  [OK] Domain: {state.core_domains}")
    print(
        f"  [OK] Number of options: {len(state.question_options) if state.question_options else 0}"
    )

    return state


# ===================== Deep Research Integration =====================


def deep_research_node(state: GeneralQAState) -> GeneralQAState:
    """
    Deep Research: Deep knowledge retrieval
    Uses deep_research subgraph to obtain sufficient knowledge background
    """
    print("\n" + "=" * 60)
    print("[Deep Research] Deep Knowledge Retrieval")
    print("=" * 60)

    if not DEEP_RESEARCH_AVAILABLE:
        print("  [WARN] Deep Research unavailable, skipping")
        state.domain_knowledge_map = {"general": {"facts": [], "context": ""}}
        return state

    try:
        # Build research question
        research_question = state.cleaned_text

        # Add option context (if available)
        if state.question_options:
            options_text = "\n".join(
                [
                    f"{chr(65 + i)}. {opt}"
                    for i, opt in enumerate(state.question_options)
                ]
            )
            research_question = f"""
Question: {state.cleaned_text}

Options:
{options_text}

Please research this question thoroughly, retrieve relevant knowledge and literature to provide sufficient background for answering.
"""

        print(f"  [INFO] Research question: {state.cleaned_text[:100]}...")

        # Call Deep Research
        from agent.nodes.subagents.deep_research.configuration import Configuration

        config = Configuration(
            max_researcher_iterations=2,  # Limit iterations for speed
            max_concurrent_research_units=2,
        )

        # Run deep research
        result = deep_research_graph.invoke(
            {"messages": [HumanMessage(content=research_question)]},
            config={"configurable": config.__dict__},
        )

        # Extract research results
        if result:
            # Get final report or messages
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    research_context = last_message.content
                else:
                    research_context = str(last_message)
            else:
                research_context = ""

            # Store in state
            domain = state.core_domains[0] if state.core_domains else "general"
            state.domain_knowledge_map = {
                domain: {
                    "context": research_context,
                    "facts": [],
                    "source": "deep_research",
                }
            }
            state.deep_research_result = {"context": research_context, "success": True}

            print(
                f"  [OK] Deep Research completed, obtained {len(research_context)} characters of knowledge background"
            )
        else:
            print("  [WARN] Deep Research returned empty results")
            state.domain_knowledge_map = {"general": {"facts": [], "context": ""}}

    except Exception as e:
        print(f"  [FAIL] Deep Research failed: {e}")
        state.domain_knowledge_map = {"general": {"facts": [], "context": ""}}

    return state


# ===================== X-Masters Integration =====================


def xmasters_node(state: GeneralQAState) -> GeneralQAState:
    """
    X-Masters: Multi-path reasoning
    Uses X-Masters for multi-solution, critique, synthesis, and selection
    """
    print("\n" + "=" * 60)
    print("[X-Masters] X-Masters: Multi-path Reasoning")
    print("=" * 60)

    if not XMASTERS_AVAILABLE:
        print("  [WARN] X-Masters unavailable, using single-path reasoning")
        return _fallback_inference(state)

    try:
        # Build problem
        problem = state.cleaned_text

        # Add options
        if state.question_options:
            options_text = "\n".join(
                [
                    f"{chr(65 + i)}. {opt}"
                    for i, opt in enumerate(state.question_options)
                ]
            )
            problem = f"""
Question: {state.cleaned_text}

Options:
{options_text}
"""

        # Add Deep Research context
        if state.domain_knowledge_map:
            domain = state.core_domains[0] if state.core_domains else "general"
            domain_data = state.domain_knowledge_map.get(domain, {})
            context = domain_data.get("context", "")
            if context:
                problem += f"""

Background knowledge:
{context[:4000]}
"""

        print(f"  [INFO] Problem length: {len(problem)} characters")
        print(f"  [RUN] Using {XMASTERS_NUM_SOLVERS} Solvers")

        # Build X-Masters graph
        xmasters_graph = build_xmasters_graph().compile()

        # Run X-Masters
        from agent.utils.llm_factory import get_current_llm_config

        llm_config = get_current_llm_config()

        result = xmasters_graph.invoke(
            {
                "problem": problem,
                "num_solvers": XMASTERS_NUM_SOLVERS,
                "llm": llm_config.get("model", ""),
                "source": llm_config.get("provider", ""),
                "temperature": 0.7,
                "timeout_seconds": 300,
            }
        )

        # Extract final answer
        final_answer = result.get("final_answer", "")

        if final_answer:
            state.final_answer = _extract_answer(final_answer, state)
            state.core_conclusion = final_answer
            print(f"  [OK] X-Masters completed")
            print(f"  [OK] Answer: {state.final_answer}")
        else:
            print("  [WARN] X-Masters returned empty answer, using fallback reasoning")
            return _fallback_inference(state)

    except Exception as e:
        print(f"  [FAIL] X-Masters failed: {e}")
        return _fallback_inference(state)

    return state


def _fallback_inference(state: GeneralQAState) -> GeneralQAState:
    """
    Fallback inference: Single-path reasoning when X-Masters is unavailable
    """
    print("\n  Using fallback single-path reasoning...")

    # Build inference prompt
    prompt = f"""Answer the question based on the following knowledge.

Question: {state.cleaned_text}

"""

    # Add options
    if state.question_options:
        options_text = "\n".join(
            [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(state.question_options)]
        )
        prompt += f"Options:\n{options_text}\n\n"

    # Add knowledge background
    if state.domain_knowledge_map:
        domain = state.core_domains[0] if state.core_domains else "general"
        domain_data = state.domain_knowledge_map.get(domain, {})
        context = domain_data.get("context", "")
        if context:
            prompt += f"Background knowledge:\n{context[:3000]}\n\n"

    prompt += """
Output JSON format:
{
    "analysis": "Analysis process",
    "answer": "Final answer (single letter like A/B/C for MCQ, True or False for T/F)"
}
"""

    response = _call_llm_simple(prompt, "Fallback")

    if response:
        result = _parse_json(response)
        if result:
            state.final_answer = _extract_answer(result.get("answer", ""), state)
            state.core_conclusion = result.get("analysis", "")
            print(f"  [OK] Fallback inference completed, Answer: {state.final_answer}")
        else:
            state.final_answer = response[:100]
            state.core_conclusion = response
    else:
        state.error_message = "Inference failed"

    return state


def _extract_answer(raw_answer: str, state: GeneralQAState) -> str:
    """
    Extract final answer from raw response
    """
    if not raw_answer:
        return ""

    raw_answer = raw_answer.strip()

    # True/False questions
    if (
        state.answer_format_label == "true/false"
        or state.question_type_label == "true_false"
    ):
        if "true" in raw_answer.lower():
            return "True"
        elif "false" in raw_answer.lower():
            return "False"

    # Multiple choice
    if state.question_options:
        # Try to extract option letter
        import re

        match = re.search(r"\b([A-E])\b", raw_answer.upper())
        if match:
            return match.group(1)

        # If response contains full option text, match to corresponding letter
        for i, opt in enumerate(state.question_options):
            if opt.lower() in raw_answer.lower():
                return chr(65 + i)

    # Numerical calculation - extract number
    if (
        state.answer_format_label == "number"
        or state.question_type_label == "calculation"
    ):
        import re

        numbers = re.findall(r"[-+]?\d*\.?\d+", raw_answer)
        if numbers:
            return numbers[-1]  # Return the last number

    return raw_answer[:200]  # Return first 200 characters by default


# ===================== N8: Answer Formatting =====================


def n8_format_node(state: GeneralQAState) -> GeneralQAState:
    """
    N8: Answer Formatting
    Ensures answer format meets requirements
    """
    print("\n" + "=" * 60)
    print("N8: Answer Formatting")
    print("=" * 60)

    if not state.final_answer:
        print("  [WARN] No answer to format")
        return state

    # Validate answer format
    answer = state.final_answer.strip()
    format_label = state.answer_format_label or "Single Choice"

    # True/False validation
    if format_label == "true/false":
        if answer.lower() not in ["true", "false"]:
            # Try to correct
            if "true" in answer.lower():
                answer = "True"
            elif "false" in answer.lower():
                answer = "False"

    # Multiple choice validation
    elif format_label == "Single Choice" and state.question_options:
        import re

        match = re.search(r"^[A-Ea-e]$", answer)
        if not match:
            # Try to extract from text
            match = re.search(r"\b([A-Ea-e])\b", answer.upper())
            if match:
                answer = match.group(1).upper()

    state.final_answer = answer
    state.format_valid_label = "Valid"

    print(f"  [OK] Final answer: {state.final_answer}")
    print(f"  [OK] Format: {format_label}")

    return state


# ===================== Routing Functions =====================


def should_skip_deep_research(state: GeneralQAState) -> str:
    """Determine whether to skip Deep Research"""
    if not ENABLE_DEEP_RESEARCH:
        return "skip"

    # Simple questions can be skipped
    if state.question_type_label == "true_false":
        # True/False questions usually do not need deep research
        if len(state.cleaned_text) < 200:
            return "skip"

    return "research"


# ===================== Build Graph =====================


def build_simplified_general_qa_graph():
    """
    Build simplified General QA subgraph

    Flow:
    START → N0(Preprocess) → Deep Research → X-Masters → N8(Format) → END
                           ↑              ↑
                        (optional skip)     (fallback inference)
    """

    graph = StateGraph(GeneralQAState)

    # Add nodes
    graph.add_node("n0_preprocess", n0_preprocess_node)
    graph.add_node("deep_research", deep_research_node)
    graph.add_node("xmasters", xmasters_node)
    graph.add_node("n8_format", n8_format_node)

    # Define edges
    graph.add_edge(START, "n0_preprocess")

    # Conditional routing: skip Deep Research?
    graph.add_conditional_edges(
        "n0_preprocess",
        should_skip_deep_research,
        {
            "research": "deep_research",
            "skip": "xmasters",  # Skip Deep Research, go directly to inference
        },
    )

    # Deep Research → X-Masters
    graph.add_edge("deep_research", "xmasters")

    # X-Masters → N8
    graph.add_edge("xmasters", "n8_format")

    # N8 → END
    graph.add_edge("n8_format", END)

    return graph.compile()


# ===================== Main Entry Point =====================


def run_simplified_general_qa(user_input: str, **kwargs) -> Dict[str, Any]:
    """
    Run simplified General QA workflow

    Args:
        user_input: Question text
        **kwargs: Additional parameters

    Returns:
        Result dictionary containing final_answer
    """
    print("\n" + "=" * 60)
    print("[START] Simplified GENERAL QA Workflow")
    print("   Core: Deep Research + X-Masters")
    print("=" * 60)

    start_time = time.time()

    # Initialize state
    state = GeneralQAState(user_input=user_input)

    # Apply additional parameters
    for key, value in kwargs.items():
        if hasattr(state, key):
            setattr(state, key, value)

    # Run graph
    graph = build_simplified_general_qa_graph()

    try:
        result_state = graph.invoke(state)

        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print(f"[SUCCESS] Completed in {elapsed:.1f}s")
        print("=" * 60)

        return {
            "final_answer": result_state.final_answer,
            "conclusion": result_state.core_conclusion,
            "duration": elapsed,
            "success": True,
            "nodes_executed": ["n0", "deep_research", "xmasters", "n8"],
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] Workflow error: {e}")

        return {
            "final_answer": None,
            "error": str(e),
            "duration": elapsed,
            "success": False,
        }


# ===================== Export =====================

__all__ = [
    "build_simplified_general_qa_graph",
    "run_simplified_general_qa",
    "ENABLE_DEEP_RESEARCH",
    "XMASTERS_NUM_SOLVERS",
]
