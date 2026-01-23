"""
Main Graph with Sandbox Execution

This is an example of how to modify main_graph.py to use sandbox execution for all subgraphs.
This provides error isolation and prevents subgraph failures from affecting the main graph.
"""

from typing import Dict, Optional, Callable, Any
import json
from agent.nodes.subagents.task_decomposition.graph import (
    task_decomposition_input_mapper, 
    task_decomposition_output_mapper,
    build_task_decomposition_subgraph
)
from agent.state import GlobalState, UserTaskType
from langgraph.graph import StateGraph, START, END
from nodes.subagents.supervisor.graph import build_supervisor_subgraph, supervisor_input_mapper, supervisor_output_mapper
from nodes.subagents.general_qa.graph import build_general_qa_subgraph, general_qa_input_mapper, general_qa_output_mapper

# Import sandbox executor
from agent.utils.subgraph_sandbox_executor import (
    execute_subgraph_in_sandbox,
    IsolationStrategy
)

# Configuration: Strategy and timeout for each subgraph
SUBGRAPH_STRATEGIES: Dict[str, IsolationStrategy] = {
    "supervisor": IsolationStrategy.THREAD,  # Balanced isolation
    "general_qa": IsolationStrategy.EXCEPTION,  # Fast, simple
    "immunity": IsolationStrategy.PROCESS,  # Complex, needs full isolation
    "task_decomposition": IsolationStrategy.THREAD,  # Balanced
    "executor": IsolationStrategy.PROCESS,  # Complex, needs full isolation
}

SUBGRAPH_TIMEOUTS: Dict[str, float] = {
    "supervisor": 120.0,  # 2 minutes
    "general_qa": 60.0,  # 1 minute
    "immunity": 300.0,  # 5 minutes
    "task_decomposition": 180.0,  # 3 minutes
    "executor": 600.0,  # 10 minutes
}


# Initialize subgraphs (for non-sandbox execution if needed)
supervisor_subgraph = build_supervisor_subgraph()
general_qa_subgraph = build_general_qa_subgraph()


# Main graph node: Immunology task processing (using immunity subgraph with sandbox)
def immunity_node(state: GlobalState) -> GlobalState:
    """
    Immunology task node with sandbox execution
    
    Uses immunity subgraph to generate executable experiment plan
    Executed in isolated sandbox to prevent errors from affecting main graph
    """
    from agent.nodes.subagents.immunity.graph import (
        build_immunity_subgraph,
        immunity_input_mapper,
        immunity_output_mapper
    )
    
    # Execute in sandbox with process isolation (most secure for complex subgraph)
    return execute_subgraph_in_sandbox(
        subgraph_name="immunity",
        subgraph_builder=build_immunity_subgraph,
        input_mapper=immunity_input_mapper,
        output_mapper=immunity_output_mapper,
        main_state=state,
        strategy=SUBGRAPH_STRATEGIES.get("immunity", IsolationStrategy.PROCESS),
        timeout=SUBGRAPH_TIMEOUTS.get("immunity", 300.0)
    )


# Main graph node: Execution plan task processing (with sandbox)
def executor_node(state: GlobalState, hitl_callback=None, use_file_interaction=False) -> GlobalState:
    """
    Execution plan task node with sandbox execution
    
    Uses executor subgraph to execute tasks in task list, supports HITL interrupt handling
    Executed in isolated sandbox to prevent errors from affecting main graph
    """
    from agent.nodes.subagents.executor.graph import (
        build_executor_subgraph,
        executor_input_mapper,
        executor_output_mapper,
        execute_executor_with_interrupt_support,
        resume_executor_after_interrupt
    )
    from agent.utils.hitl_interaction import handle_hitl_interrupt
    
    # Note: Executor with HITL support needs special handling
    # For now, use basic sandbox execution
    # TODO: Enhance to support HITL in sandbox environment
    
    return execute_subgraph_in_sandbox(
        subgraph_name="executor",
        subgraph_builder=build_executor_subgraph,
        input_mapper=executor_input_mapper,
        output_mapper=executor_output_mapper,
        main_state=state,
        strategy=SUBGRAPH_STRATEGIES.get("executor", IsolationStrategy.PROCESS),
        timeout=SUBGRAPH_TIMEOUTS.get("executor", 600.0)
    )


# Build main graph with sandbox execution
def build_main_graph():
    """Build main graph with sandbox execution for all subgraphs"""
    main_graph = StateGraph(GlobalState)
    
    # Main graph node 1: Call supervisor subgraph (with sandbox)
    def run_supervisor_subgraph(state: GlobalState) -> GlobalState:
        """Supervisor node with sandbox execution"""
        return execute_subgraph_in_sandbox(
            subgraph_name="supervisor",
            subgraph_builder=build_supervisor_subgraph,
            input_mapper=supervisor_input_mapper,
            output_mapper=supervisor_output_mapper,
            main_state=state,
            strategy=SUBGRAPH_STRATEGIES.get("supervisor", IsolationStrategy.THREAD),
            timeout=SUBGRAPH_TIMEOUTS.get("supervisor", 120.0)
        )
    main_graph.add_node("supervisor", run_supervisor_subgraph)

    # Main graph node 2: General QA (with sandbox)
    def run_general_qa_node(state: GlobalState) -> GlobalState:
        """General QA node with sandbox execution"""
        return execute_subgraph_in_sandbox(
            subgraph_name="general_qa",
            subgraph_builder=build_general_qa_subgraph,
            input_mapper=general_qa_input_mapper,
            output_mapper=general_qa_output_mapper,
            main_state=state,
            strategy=SUBGRAPH_STRATEGIES.get("general_qa", IsolationStrategy.EXCEPTION),
            timeout=SUBGRAPH_TIMEOUTS.get("general_qa", 60.0)
        )
    main_graph.add_node("general_qa", run_general_qa_node)
    
    # Main graph node 3: Immunology task node
    main_graph.add_node("immunity", immunity_node)
    
    # Task decomposition node (with sandbox)
    def run_task_decomposition_node(state: GlobalState) -> GlobalState:
        """Task decomposition node with sandbox execution"""
        return execute_subgraph_in_sandbox(
            subgraph_name="task_decomposition",
            subgraph_builder=build_task_decomposition_subgraph,
            input_mapper=task_decomposition_input_mapper,
            output_mapper=task_decomposition_output_mapper,
            main_state=state,
            strategy=SUBGRAPH_STRATEGIES.get("task_decomposition", IsolationStrategy.THREAD),
            timeout=SUBGRAPH_TIMEOUTS.get("task_decomposition", 180.0)
        )
    main_graph.add_node("task_decomposition", run_task_decomposition_node)
    
    # Define main graph flow rules
    main_graph.add_edge(START, "supervisor")
    
    # Supervisor → router: Route to corresponding subgraph based on task type
    def post_supervisor_router(state: GlobalState) -> str:
        """
        Route to different processing nodes based on task type
        """
        user_task_type = state.user_task_type
        
        if user_task_type == UserTaskType.IMMUNOLOGY_TASK:
            return "immunity"
        elif user_task_type == UserTaskType.EXECUTE_PLAN:
            return "executor"
        elif user_task_type == UserTaskType.GENERAL_QA:
            return "general_qa"
        else:
            # Default route to general Q&A
            return "general_qa"
    
    main_graph.add_conditional_edges(
        "supervisor", 
        post_supervisor_router, 
        {
            "immunity": "immunity",
            "executor": "task_decomposition",
            "general_qa": "general_qa"
        }
    )
    
    # After task decomposition, if task list is generated, call immunity to generate experiment plan
    def post_task_decomposition_router(state: GlobalState) -> str:
        """
        After task decomposition, if there's a task list, generate experiment plan; otherwise end
        """
        if state.subtasks or state.parallel_task_groups:
            return "immunity_plan"
        else:
            return END
    
    main_graph.add_conditional_edges(
        "task_decomposition",
        post_task_decomposition_router,
        {
            "immunity_plan": "immunity_plan",
            END: END
        }
    )
    
    # Add immunity_plan node (generate experiment plan with sandbox)
    def run_immunity_plan_node(state: GlobalState) -> GlobalState:
        """
        Generate experiment plan node with sandbox execution
        
        Use immunity subgraph to generate executable experiment plan (don't execute tasks)
        """
        from agent.nodes.subagents.immunity.graph import (
            build_immunity_subgraph,
            immunity_input_mapper,
            immunity_output_mapper
        )
        
        return execute_subgraph_in_sandbox(
            subgraph_name="immunity_plan",
            subgraph_builder=build_immunity_subgraph,
            input_mapper=immunity_input_mapper,
            output_mapper=immunity_output_mapper,
            main_state=state,
            strategy=SUBGRAPH_STRATEGIES.get("immunity", IsolationStrategy.PROCESS),
            timeout=SUBGRAPH_TIMEOUTS.get("immunity", 300.0)
        )
    
    main_graph.add_node("immunity_plan", run_immunity_plan_node)
    
    # All processing nodes → end
    main_graph.add_edge("general_qa", END)
    main_graph.add_edge("immunity", END)
    main_graph.add_edge("immunity_plan", END)

    return main_graph.compile()


# ===================== Error Handling Helper =====================

def check_subgraph_errors(state: GlobalState) -> Dict[str, Any]:
    """
    Check for subgraph execution errors in the state
    
    Returns:
        Dictionary of subgraph errors, empty if no errors
    """
    errors = {}
    if state.merged_result:
        for key, value in state.merged_result.items():
            if key.endswith("_error") and isinstance(value, dict):
                subgraph_name = key.replace("_error", "")
                errors[subgraph_name] = value
    return errors


def log_subgraph_errors(state: GlobalState):
    """Log all subgraph errors found in the state"""
    errors = check_subgraph_errors(state)
    if errors:
        print("\n⚠ Subgraph Execution Errors Detected:")
        for subgraph_name, error_info in errors.items():
            print(f"\n  {subgraph_name}:")
            print(f"    Error: {error_info.get('error', 'Unknown error')}")
            print(f"    Type: {error_info.get('error_type', 'Unknown')}")
            print(f"    Time: {error_info.get('execution_time', 0):.2f}s")
        print("\n✓ Main graph continues execution despite subgraph errors\n")

