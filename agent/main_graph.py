# graph/main_graph.py
from state import GlobalState, UserTaskType
from langgraph.graph import StateGraph, START, END
from agent.utils.subgraph_sandbox_executor import execute_subgraph_in_sandbox

def supervisor_node(state: GlobalState) -> GlobalState:
    from nodes.subagents.supervisor.graph import (
        build_supervisor_subgraph,
        supervisor_input_mapper,
        supervisor_output_mapper
    )
    
    return execute_subgraph_in_sandbox(
        subgraph_name="supervisor",
        subgraph_builder=build_supervisor_subgraph,
        input_mapper=supervisor_input_mapper,
        output_mapper=supervisor_output_mapper,
        main_state=state,
        timeout=60 * 10
    )
    
def general_qa_node(state: GlobalState) -> GlobalState:
    from nodes.subagents.general_qa.graph import (
        build_general_qa_subgraph,
        general_qa_input_mapper,
        general_qa_output_mapper
    )
    
    return execute_subgraph_in_sandbox(
        subgraph_name="general_qa",
        subgraph_builder=build_general_qa_subgraph,
        input_mapper=general_qa_input_mapper,
        output_mapper=general_qa_output_mapper,
        main_state=state,
        timeout=60 * 10
    )
    
def task_decomposition_node(state: GlobalState) -> GlobalState:
    from nodes.subagents.task_decomposition.graph import (
        build_task_decomposition_subgraph,
        task_decomposition_input_mapper,
        task_decomposition_output_mapper
    )
    
    return execute_subgraph_in_sandbox(
        subgraph_name="task_decomposition",
        subgraph_builder=build_task_decomposition_subgraph,
        input_mapper=task_decomposition_input_mapper,
        output_mapper=task_decomposition_output_mapper,
        main_state=state,
        timeout=60 * 10
    )

def immunity_node(state: GlobalState) -> GlobalState:
    from nodes.subagents.immunity.graph import (
        build_immunity_subgraph,
        immunity_input_mapper,
        immunity_output_mapper
    )
    
    return execute_subgraph_in_sandbox(
        subgraph_name="immunity",
        subgraph_builder=build_immunity_subgraph,
        input_mapper=immunity_input_mapper,
        output_mapper=immunity_output_mapper,
        main_state=state,
        timeout=60 * 10
    )

def executor_node(state: GlobalState, hitl_callback=None, use_file_interaction=False) -> GlobalState:
    from nodes.subagents.executor.graph import (
        build_executor_subgraph,
        executor_input_mapper,
        executor_output_mapper,
    )

    return execute_subgraph_in_sandbox(
        subgraph_name="executor",
        subgraph_builder=build_executor_subgraph,
        input_mapper=executor_input_mapper,
        output_mapper=executor_output_mapper,
        main_state=state,
        timeout=60 * 10
    )

def analysis_node(state: GlobalState) -> GlobalState:
    from nodes.subagents.analysis.graph import (
        build_analysis_subgraph,
        analysis_input_mapper,
        analysis_output_mapper
    )
    
    return execute_subgraph_in_sandbox(
        subgraph_name="analysis",
        subgraph_builder=build_analysis_subgraph,
        input_mapper=analysis_input_mapper,
        output_mapper=analysis_output_mapper,
        main_state=state,
        timeout=60 * 10
    )

# route nodes
def supervisor_router(state: GlobalState) -> str:
    user_task_type = state.user_task_type
    if user_task_type == UserTaskType.IMMUNOLOGY_TASK:
        return "immunity"
    elif user_task_type == UserTaskType.EXECUTE_PLAN:
        return "task_decomposition"
    elif user_task_type == UserTaskType.USE_HISTORY:
         return "executor"
    else:
        return "general_qa"

# Build main graph
def build_main_graph():
    main_graph = StateGraph(GlobalState)
    
    main_graph.add_node("supervisor", supervisor_node)
    main_graph.add_node("general_qa", general_qa_node)
    main_graph.add_node("task_decomposition", task_decomposition_node)
    main_graph.add_node("immunity", immunity_node)
    main_graph.add_node("executor", executor_node)
    
    main_graph.add_edge(START, "supervisor")
    main_graph.add_conditional_edges("supervisor", supervisor_router, {
        "immunity": "immunity",
        "task_decomposition": "task_decomposition",
        "executor": "executor",
        "general_qa": "general_qa"
    })

    main_graph.add_edge("immunity", "task_decomposition")
    main_graph.add_edge("task_decomposition", "executor")

    main_graph.add_edge("executor", "analysis")
    
    main_graph.add_edge("general_qa", END)
    main_graph.add_edge("analysis", END)
    
    return main_graph.compile()
