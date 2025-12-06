import re
import uuid

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from common.factory import get_default_model, get_reasoning_model
from common.util.retrieval_utils import remove_think_tags
from usecases.immunity.common.constants import get_tools_json
from usecases.immunity.config.immunity_config import get_runnable_config
from usecases.immunity.prompts.prompts import ImmunityPrompts
from usecases.immunity.schema.common_schemas import Document, QueryExpansion
from usecases.immunity.state.state import ParallelPlanState, RetrievalState
from usecases.immunity.tools.retrieve_tools import (
    retrieve,
    web_retrieval_search,
    web_search_node,
)


def query_rewriter(state: RetrievalState, config: RunnableConfig):
    """Agent node - Query optimization"""
    prompt = ChatPromptTemplate.from_template(ImmunityPrompts.QUERY_EXPANSION_PROMPT)
    tools_info = get_tools_json()
    reasoning_model = get_reasoning_model(config)
    structured_model = reasoning_model.with_structured_output(QueryExpansion)
    runnable = prompt | structured_model
    response = runnable.invoke(
        {"tools_info": tools_info, "query": state.original_question}
    )

    optimized_questions = response.queries

    print("Optimized queries:")
    for i, query in enumerate(optimized_questions, 1):
        print(f"  {i}. {query}")
    state.optimized_questions = optimized_questions
    return state


tools = [retrieve, web_search_node, web_retrieval_search]
tool_node = ToolNode(tools)


async def retrieval_agent(state: RetrievalState, config: RunnableConfig):
    """Retrieval agent node - Directly call three tools in parallel (KB + Web Search + Web Retrieval)"""
    query = ";".join(state.optimized_questions or [state.original_question])

    # Manually construct AIMessage containing multiple tool calls as shown in the diagram
    from langchain_core.messages import AIMessage

    message_with_multiple_tool_calls = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "retrieve",
                "args": {
                    "query": state.optimized_questions,
                    "config": config,
                    "state": state,
                },
                "id": "tool_call_id_1",
                "type": "tool_call",
            },
            {
                "name": "web_search_node",
                "args": {
                    "query": state.optimized_questions,
                    "state": state,
                    "config": config,
                },
                "id": "tool_call_id_2",
                "type": "tool_call",
            },
            {
                "name": "web_retrieval_search",
                "args": {"query": state.optimized_questions},
                "id": "tool_call_id_3",
                "type": "tool_call",
            },
        ],
    )

    # Use ainvoke to asynchronously call ToolNode, supporting async tools
    result = await tool_node.ainvoke({"messages": [message_with_multiple_tool_calls]})

    # Merge tool results
    tool_results = []
    documents = []  # Used to store the 2D structure of docs lists generated in each loop

    for msg in result.get("messages", []):
        if hasattr(msg, "content") and msg.content:
            tool_results.append(msg.content)
            # Parse XML format document tags
            document_pattern = r"<document>\s*<source>(.*?)</source>\s*<(?:page_content|content)>(.*?)</(?:page_content|content)>\s*</document>"
            matches = re.findall(document_pattern, msg.content, re.DOTALL)
            docs = []  # List of Document objects parsed from current message
            for source, content in matches:
                # Create Document object and add to current docs list
                doc = Document(source=source.strip(), content=content.strip())
                docs.append(doc)
            # Add current message's docs list as a whole to documents
            # This makes documents a 2D structure: [[doc1, doc2], [doc3, doc4], [doc5]]
            documents.append(docs)

    # Store Document objects in state (if needed)
    if hasattr(state, "retrieval_docs"):
        # Maintain 2D structure, directly store documents' 2D structure in state
        # state.retrieval_docs will be: [[doc1, doc2], [doc3, doc4], [doc5]]
        state.retrieval_docs.extend(documents)
    context = "\n\n".join(tool_results)
    state.context = context
    state.optimized_question = query
    print(f"Optimized query: {state.optimized_question}")
    print(f"Context: {state.context}")
    return state


def route_to_parallel_plans(state: RetrievalState):
    """Routing function - Return Send list for parallel processing"""
    print(f"\n===== Parallel Distribution Node =====")
    print(f"Number of queries: {len(state.optimized_questions)}")

    # Create Send object for each query, send to generate_single_plan node
    sends = []
    for query in state.optimized_questions:
        send_obj = Send(
            "generate_single_plan",
            {
                "original_question": state.original_question,
                "optimized_questions": state.optimized_questions,
                "context": state.context,
                "query": query,
                "individual_plans": [],
            },
        )
        sends.append(send_obj)

    print(f"Created {len(sends)} Send objects")
    return sends


def generate_single_plan(state, config: RunnableConfig):
    """Single plan generation node - Receiver for Send API"""
    print(f"\n===== Generate Single Plan =====")

    # Send API sends dictionary, access key-value directly
    query = state["query"]
    context = state["context"]

    print(f"Processing query: {query}")

    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CellPrompt.SYSTEMT_PLAN_GENERATION_PROMPT),
            ("user", CellPrompt.USER_PLAN_GENERATION_PROMPT),
        ]
    )

    model = get_default_model(config)
    chain = planner_prompt | model | StrOutputParser() | remove_think_tags

    plan_response = chain.invoke(
        {
            "objective": [query],  # Pass single query as list
            "context": context,
        }
    )

    print(f"Generated plan length: {len(plan_response)} characters")

    # Return to individual_plans list (will be automatically merged by reducer)
    # Bug fix: Return Dict instead of str to match ParallelPlanState type
    return {"individual_plans": [{"query": query, "plan": plan_response}]}


def integrate_plans(state: ParallelPlanState, config: RunnableConfig):
    """Integrate multiple parallel plans"""
    print(f"\n===== Integrate Parallel Plans =====")
    print(f"Number of plans received: {len(state.individual_plans)}")

    if len(state.individual_plans) == 1:
        # If there's only one plan, use it directly
        integrated_plan = state.individual_plans[0]
    else:
        # If there are multiple plans, use LLM to integrate
        # Build structured parameter dictionary
        integration_params = {
            # Build each research objective separately
            "objective_1": state.optimized_questions[0]
            if len(state.optimized_questions) > 0
            else "",
            "objective_2": state.optimized_questions[1]
            if len(state.optimized_questions) > 1
            else "",
            "objective_3": state.optimized_questions[2]
            if len(state.optimized_questions) > 2
            else "",
            "objective_4": state.optimized_questions[3]
            if len(state.optimized_questions) > 3
            else "",
            # Build each analysis plan separately
            "plan_1": state.individual_plans[0]
            if len(state.individual_plans) > 0
            else "",
            "plan_2": state.individual_plans[1]
            if len(state.individual_plans) > 1
            else "",
            "plan_3": state.individual_plans[2]
            if len(state.individual_plans) > 2
            else "",
            "plan_4": state.individual_plans[3]
            if len(state.individual_plans) > 3
            else "",
        }

        integration_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CellPrompt.INTEGRATION_SYSTEM_PROMPT),
                ("user", CellPrompt.INTEGRATION_USER_PROMPT),
            ]
        )
        model = get_default_model(config)
        integration_chain = (
            integration_prompt | model | StrOutputParser() | remove_think_tags
        )
        integrated_plan = integration_chain.invoke(integration_params)

    print(f"Integrated plan length: {len(integrated_plan)} characters")

    # Build complete objective for logging
    objective = ""
    for i, q in enumerate(state.optimized_questions):
        objective += f"**Requirement {i + 1}**: {q}\n\n"

    print(f"\n===== Prompt Parameters Sent to LLM =====")
    print(f"Objective: {objective}")
    print(f"Context: {state.context}")
    state.generated_plan = integrated_plan
    return state


def create_parallel_rag_graph():
    """Create RAG graph supporting parallel processing"""
    workflow = StateGraph(ParallelPlanState)

    # Add nodes
    workflow.add_node("query_rewriter", query_rewriter)
    workflow.add_node("retrieval_agent", retrieval_agent)
    workflow.add_node("generate_single_plan", generate_single_plan)
    workflow.add_node("integrate_plans", integrate_plans)

    # Set entry point and edges
    workflow.set_entry_point("query_rewriter")
    workflow.add_edge("query_rewriter", "retrieval_agent")

    # Use conditional_edges directly from retrieval_agent for parallel distribution
    workflow.add_conditional_edges(
        "retrieval_agent",
        route_to_parallel_plans,
        ["generate_single_plan"],  # Specify possible target nodes
    )

    # Single plan -> Integration
    workflow.add_edge("generate_single_plan", "integrate_plans")
    workflow.add_edge("integrate_plans", END)

    graph = workflow.compile()

    # Print workflow diagram
    try:
        print("\n===== Parallel LangGraph Workflow Diagram =====")
        print(graph.get_graph().draw_mermaid())
    except Exception as e:
        print(f"Error generating workflow diagram: {str(e)}")

    return graph


async def run_parallel_rag_graph(original_question: str, config: RunnableConfig):
    """Run parallel RAG graph"""
    graph = create_parallel_rag_graph()
    config = get_runnable_config(uuid.uuid4())

    initial_state = ParallelPlanState(
        original_question=original_question,
        optimized_questions=[],
        generated_plan="",
        context="",
        query="",
        individual_plans=[],
    )

    # Use async stream to let graph flow naturally
    final_state = None
    async for event in graph.astream(initial_state, config):
        print(f"Current node: {list(event.keys())}")
        final_state = event

    if final_state and "integrate_plans" in final_state:
        generated_plan = final_state["integrate_plans"]
    else:
        print("Warning: integrate_plans result not found")
        generated_plan = {"generated_plan": ""}

    print(f"\nParallel workflow completed")
    print(f"Question: {original_question}")

    return generated_plan


async def complete_rag_pipeline(original_question: str, config: RunnableConfig):
    """Complete RAG pipeline - Generate all PRP_6 required artifacts"""
    print("=== LangGraph RAG Retrieval Process Started ===")
    result = await run_parallel_rag_graph(original_question, config)  # Parallel version

    # Generate all required PRP_6 artifacts
    # NOTE: Artifact generator disabled for standalone version
    # print("\n=== Generate PRP_6 Evidence Files ===")
    # from ..graph.artifact_generator import ArtifactGenerator
    #
    # generator = ArtifactGenerator()
    #
    # # Extract papers from context if available
    # papers = []
    # if isinstance(result, dict) and 'context' in result:
    #     # Context contains retrieved papers
    #     papers = [result['context'][:1000]]  # Sample of context
    #
    # # Generate all artifacts
    # artifacts = generator.generate_all_artifacts(
    #     question=original_question,
    #     papers=papers,
    #     state=result
    # )
    #
    # # Add artifact info to result
    # result['artifacts_generated'] = artifacts
    # result['artifacts_dir'] = artifacts['output_dir']
    # result['session_id'] = artifacts['session_id']
    # result['overall_score'] = artifacts['metrics']['overall_score']['likert_average']

    # print(f"\n✅ Complete Result + Artifacts: {result['artifacts_dir']}")
    # print(f"📊 Planning Score: {result['overall_score']}/5.0")

    return result
