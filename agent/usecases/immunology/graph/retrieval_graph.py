"""
Retrieval graph adapted for ImmuneAgent.
Based on cell/graph/retrieval_graph.py but self-contained.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from common.factory import get_default_model, get_reasoning_model
from common.util.retrieval_utils import remove_think_tags
from usecases.immunology.immunology_config import get_immunology_runnable_config
from usecases.immunology.state import ImmuneAgentState
from usecases.immunology.tools.retrieval_tools import ImmunologyRetriever


def hybrid_retrieval(query: str) -> Dict[str, Any]:
    """Hybrid retrieval using ImmunologyRetriever."""
    retriever = ImmunologyRetriever()
    results = retriever.retrieve_with_rerank(query, k=10, rerank_k=5)

    context = "\n\n".join([r["content"] for r in results])
    citations = [r["citation"] for r in results]

    return {"results": results, "context": context, "citations": citations}


# Query expansion schema
@dataclass
class QueryExpansion:
    """Query expansion results."""

    queries: List[str] = field(default_factory=list)


class ImmunologyPrompts:
    """Prompts for immunology retrieval."""

    QUERY_EXPANSION_PROMPT = """You are an immunology expert. Expand this research question into 2-3 specific sub-questions.

Question: {query}

Generate 3 focused sub-questions that cover different aspects:
- Molecular mechanisms
- Clinical applications  
- Experimental approaches

Return your response as a JSON object with a 'queries' field containing the list of questions."""

    PLAN_GENERATION_PROMPT = """Based on the research objectives and scientific context, create a detailed research plan.

Objectives: {objectives}

Context: {context}

Create a comprehensive plan including:
1. Hypothesis formulation
2. Experimental design
3. Tool selection
4. Expected outcomes
5. Validation strategy"""

    INTEGRATION_PROMPT = """Integrate multiple research plans into a cohesive strategy.

**Original Research Objectives for Integration:**
1. Objective 1:
{objective_1}

2. Objective 2:
{objective_2}

3. Objective 3:
{objective_3}

**Candidate Analysis Plans for Integration:**

1. Plan 1:
{plan_1}

2. Plan 2:
{plan_2}

3. Plan 3:
{plan_3}

Create an integrated plan that:
- Combines complementary approaches
- Eliminates redundancy
- Prioritizes critical experiments
- Maintains scientific rigor"""


def query_rewriter(state: ImmuneAgentState, config: RunnableConfig):
    """Rewrite and expand research query."""
    prompt = ChatPromptTemplate.from_template(ImmunologyPrompts.QUERY_EXPANSION_PROMPT)
    model = get_reasoning_model(config)
    structured_model = model.with_structured_output(QueryExpansion)
    runnable = prompt | structured_model
    response = runnable.invoke({"query": state.original_question})

    # 处理 structured output 可能返回字典或对象的情况
    if isinstance(response, dict):
        optimized_questions = response.get("queries", [])
    else:
        optimized_questions = response.queries

    print("优化后的查询:")
    for i, query in enumerate(optimized_questions, 1):
        print(f"  {i}. {query}")
    state.optimized_questions = optimized_questions
    return state


def retrieval_agent(state: ImmuneAgentState, config: RunnableConfig):
    """Retrieve relevant documents using parallel retrieval."""

    # Combine queries
    all_queries = state.optimized_questions or [state.research_question]

    # Retrieve for each query
    all_results = []
    all_contexts = []

    for query in all_queries:
        # Use our retrieval system
        results = hybrid_retrieval(query)
        all_results.extend(results.get("results", []))
        all_contexts.append(results.get("context", ""))

    # Combine and deduplicate
    state.context = "\n\n".join(all_contexts)
    state.retrieved_documents = all_results[:10]  # Keep top 10

    print(f"Retrieved {len(all_results)} documents")

    return state


def route_to_parallel_plans(state: ImmuneAgentState):
    """Route to parallel plan generation."""
    print(f"\n===== Parallel Planning =====")
    print(f"Query count: {len(state.optimized_questions)}")

    sends = []
    for query in state.optimized_questions:
        send_obj = Send(
            "generate_single_plan",
            {
                "research_question": state.research_question,
                "optimized_questions": state.optimized_questions,
                "context": state.context,
                "query": query,
                "individual_plans": [],
            },
        )
        sends.append(send_obj)

    print(f"Created {len(sends)} Send objects")
    return sends


def generate_single_plan(state: Dict, config: RunnableConfig):
    """Generate plan for single query."""

    print(f"\n===== Generate Single Plan =====")

    query = state.get("query", "")
    context = state.get("context", "")

    print(f"Processing: {query[:60]}...")

    prompt = ChatPromptTemplate.from_template(ImmunologyPrompts.PLAN_GENERATION_PROMPT)
    model = get_default_model(config)
    chain = prompt | model | StrOutputParser() | remove_think_tags

    plan = chain.invoke({"objectives": query, "context": context})

    print(f"Generated plan: {len(plan)} chars")

    return {"individual_plans": [plan]}


def integrate_plans(state: ImmuneAgentState, config: RunnableConfig):
    """Integrate multiple parallel plans."""
    from langchain_openai import ChatOpenAI

    print(f"\n===== Integrate Plans =====")
    print(f"Plans to integrate: {len(state.individual_plans)}")

    if len(state.individual_plans) == 1:
        integrated_plan = state.individual_plans[0]
    else:
        # Combine multiple plans
        integration_params = {
            # 分别构建每个研究目标
            "objective_1": state.optimized_questions[0]
            if len(state.optimized_questions) > 0
            else "",
            "objective_2": state.optimized_questions[1]
            if len(state.optimized_questions) > 1
            else "",
            "objective_3": state.optimized_questions[2]
            if len(state.optimized_questions) > 2
            else "",
            # 分别构建每个分析计划
            "plan_1": state.individual_plans[0]
            if len(state.individual_plans) > 0
            else "",
            "plan_2": state.individual_plans[1]
            if len(state.individual_plans) > 1
            else "",
            "plan_3": state.individual_plans[2]
            if len(state.individual_plans) > 2
            else "",
        }

        prompt = ChatPromptTemplate.from_template(ImmunologyPrompts.INTEGRATION_PROMPT)
        model = get_default_model(config)
        chain = prompt | model | StrOutputParser() | remove_think_tags
        integrated_plan = chain.invoke(integration_params)

    state.generated_plan = integrated_plan
    print(f"Integrated plan: {len(integrated_plan)} chars")

    return state


def create_parallel_rag_graph():
    """Create parallel RAG graph for immunology."""
    workflow = StateGraph(ImmuneAgentState)

    # Add nodes
    workflow.add_node("query_rewriter", query_rewriter)
    workflow.add_node("retrieval_agent", retrieval_agent)
    workflow.add_node("generate_single_plan", generate_single_plan)
    workflow.add_node("integrate_plans", integrate_plans)

    # Set entry and edges
    workflow.add_edge(START, "query_rewriter")
    workflow.add_edge("query_rewriter", "retrieval_agent")

    # Conditional routing for parallel processing
    workflow.add_conditional_edges(
        "retrieval_agent", route_to_parallel_plans, ["generate_single_plan"]
    )

    workflow.add_edge("generate_single_plan", "integrate_plans")
    workflow.add_edge("integrate_plans", END)

    return workflow.compile()


def run_parallel_rag_graph(question: str, category: str = "general_immunology"):
    """Run the parallel RAG graph."""

    graph = create_parallel_rag_graph()

    # Create initial state
    initial_state = ImmuneAgentState(
        research_question=question,
        original_question=question,
        analysis_type=category,
        domain="immunology",
        optimized_questions=[],
        individual_plans=[],
    )
    config = get_immunology_runnable_config(uuid.uuid4())

    # Run the graph
    final_state = None
    for event in graph.stream(initial_state, config):
        print(f"Current node: {list(event.keys())}")
        final_state = event

    # Extract final state
    if final_state and "integrate_plans" in final_state:
        result_state = final_state["integrate_plans"]
        # Handle both dict and ImmuneAgentState object cases
        if isinstance(result_state, dict):
            return {
                "success": True,
                "question": question,
                "optimized_questions": result_state.get("optimized_questions", []),
                "context": result_state.get("context", ""),
                "plan": result_state.get("generated_plan", ""),
            }
        else:
            return {
                "success": True,
                "question": question,
                "optimized_questions": result_state.optimized_questions,
                "context": result_state.context,
                "plan": result_state.generated_plan,
            }

    return {"success": False, "error": "Failed to generate plan"}


# Convenience function
def complete_rag_pipeline(question: str, category: str = "general_immunology"):
    """Complete RAG pipeline for immunology."""
    print("=== ImmuneAgent RAG Pipeline ===")
    result = run_parallel_rag_graph(question, category)

    if result["success"]:
        print(f"\n✅ RAG Pipeline Complete")
        print(f"   Expanded to {len(result.get('optimized_questions', []))} queries")
        print(f"   Generated comprehensive plan")
    else:
        print(f"\n❌ RAG Pipeline Failed: {result.get('error', 'Unknown')}")

    return result


if __name__ == "__main__":
    # Test the retrieval graph
    test_question = "How to design CAR-T cells for solid tumors?"
    result = complete_rag_pipeline(test_question, "cell_therapy")

    if result["success"]:
        print("\nGenerated Plan Preview:")
        print(result["plan"][:500] + "...")
