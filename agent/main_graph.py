# graph/main_graph.py
from typing import Dict, Optional, Callable, Any
import json
from agent.nodes.subagents.task_decomposition.graph import (
    task_decomposition_input_mapper, 
    task_decomposition_output_mapper,
    build_task_decomposition_subgraph
)
from state import GlobalState, UserTaskType
from langgraph.graph import StateGraph, START, END
from nodes.subagents.supervisor.graph import build_supervisor_subgraph, supervisor_input_mapper, supervisor_output_mapper
from nodes.subagents.general_qa.graph import build_general_qa_subgraph, general_qa_input_mapper, general_qa_output_mapper

# Initialize all subgraphs
supervisor_subgraph = build_supervisor_subgraph()
general_qa_subgraph = build_general_qa_subgraph()




# Main graph node: Immunology task processing (using immunity subgraph)
def immunity_node(state: GlobalState) -> GlobalState:
    """
    Immunology task node
    
    Uses immunity subgraph to generate executable experiment plan
    
    Args:
        state: Global state
    
    Returns:
        Updated global state
    """
    from agent.nodes.subagents.immunity.graph import (
        build_immunity_subgraph,
        immunity_input_mapper,
        immunity_output_mapper
    )
    
    # Build immunity subgraph
    immunity_subgraph = build_immunity_subgraph()
    
    # Main graph → subgraph: Map state
    immunity_input = immunity_input_mapper(state)
    
    # Execute subgraph
    immunity_output = immunity_subgraph.invoke(immunity_input)
    
    # Subgraph → main graph: Sync results
    return immunity_output_mapper(immunity_output, state)


# Main graph node: Execution plan task processing (removed, replaced with immunity subgraph)
# Note: executor functionality has been removed, now using immunity subgraph to generate experiment plan
def executor_node(state: GlobalState, hitl_callback=None, use_file_interaction=False) -> GlobalState:
    """
    Execution plan task node
    
    Uses executor subgraph to execute tasks in task list, supports HITL interrupt handling
    
    Args:
        state: Global state
        hitl_callback: HITL interaction callback function (optional)
        use_file_interaction: Whether to use file interaction (default False, use console interaction)
    
    Returns:
        Updated global state
    """
    from agent.nodes.subagents.executor.graph import (
        build_executor_subgraph,
        executor_input_mapper,
        executor_output_mapper,
        execute_executor_with_interrupt_support,
        resume_executor_after_interrupt
    )
    from agent.utils.hitl_interaction import handle_hitl_interrupt
    
    # Build executor subgraph
    executor_subgraph = build_executor_subgraph()
    
    # Main graph → subgraph: Map state
    executor_input = executor_input_mapper(state)
    
    # Use interrupt-supporting execution function
    thread_id = f"main_executor_{id(state)}"  # Use state object ID as thread ID
    
    # First execution or resume execution
    resume_value = None
    if state.hitl_status:
        # If HITL status exists, try to parse as resume value
        try:
            hitl_data = json.loads(state.hitl_status)
            if hitl_data.get("type") in ["response_parameters", "response_confirmation"]:
                resume_value = hitl_data
        except:
            pass
    
    result = execute_executor_with_interrupt_support(
        executor_subgraph,
        executor_input,
        thread_id=thread_id,
        resume_value=resume_value
    )
    
    # If interrupted, handle user interaction
    while result.get("interrupted", False):
        interrupt_data = result.get("interrupt_data")
        if not interrupt_data:
            # If no interrupt data, try to get from parent_state
            if executor_input.parent_state and executor_input.parent_state.hitl_status:
                try:
                    interrupt_data = json.loads(executor_input.parent_state.hitl_status)
                except:
                    pass
        
        if interrupt_data:
            # Handle HITL interaction
            try:
                user_response = handle_hitl_interrupt(
                    interrupt_data,
                    callback=hitl_callback,
                    use_file=use_file_interaction
                )
                
                # Update state.hitl_status (for next resume)
                state.hitl_status = json.dumps(user_response, ensure_ascii=False)
                
                # Resume execution
                result = resume_executor_after_interrupt(
                    executor_subgraph,
                    thread_id=thread_id,
                    resume_value=user_response
                )
                
                # If still interrupted, continue loop
                if not result.get("interrupted", False):
                    break
            except KeyboardInterrupt:
                # User exit
                print("\nUser exited, execution terminated")
                # Mark incomplete tasks as failed
                executor_state = result.get("result")
                if executor_state:
                    for task in executor_state.subtasks:
                        if executor_state.task_status_map.get(task.task_id) not in [
                            "completed", "failed"
                        ]:
                            executor_state.task_status_map[task.task_id] = "failed"
                            if task.task_id not in executor_state.task_results:
                                from agent.nodes.subagents.executor.graph import TaskExecutionResult, ExecutorTaskStatus
                                executor_state.task_results[task.task_id] = TaskExecutionResult(
                                    task_id=task.task_id,
                                    status=ExecutorTaskStatus.FAILED,
                                    execution_mode="",
                                    error="User exited"
                                )
                break
            except Exception as e:
                print(f"⚠ HITL interaction handling failed: {e}")
                # Continue execution, but may not get user input
                break
        else:
            # No interrupt data, cannot handle
            print("⚠ Interrupt detected, but cannot get interrupt data")
            break
    
    # Get final result
    executor_output = result.get("result")
    if executor_output is None:
        # If no result, try to use last state
        executor_output = executor_input
    
    # Subgraph → main graph: Sync results
    return executor_output_mapper(executor_output, state)


# Build main graph
def build_main_graph():
    main_graph = StateGraph(GlobalState)
    
    # Main graph node 1: Call supervisor subgraph
    def run_supervisor_subgraph(state: GlobalState) -> GlobalState:
        # Main graph → subgraph: Map state
        subgraph_input = supervisor_input_mapper(state)
        # Execute subgraph
        subgraph_output = supervisor_subgraph.invoke(subgraph_input)
        # Subgraph → main graph: Sync results
        return supervisor_output_mapper(subgraph_output, state)
    main_graph.add_node("supervisor", run_supervisor_subgraph)

    def run_general_qa_node(state: GlobalState) -> GlobalState:
        # Main graph → node: Map state
        general_input = general_qa_input_mapper(state)
        # Execute node
        general_output = general_qa_subgraph.invoke(general_input)
        # Node → main graph: Sync results
        return general_qa_output_mapper(general_output, state)
    main_graph.add_node("general_qa", run_general_qa_node)
    
    # Main graph node 3: Immunology task node (placeholder)
    main_graph.add_node("immunity", immunity_node)
    
    # Task decomposition node
    def run_task_decomposition_node(state: GlobalState) -> GlobalState:
        # Build task decomposition subgraph
        task_decomposition_subgraph = build_task_decomposition_subgraph()
        # Main graph → node: Map state
        task_decomposition_input = task_decomposition_input_mapper(state)
        # Execute node
        task_decomposition_output = task_decomposition_subgraph.invoke(task_decomposition_input)
        # Node → main graph: Sync results
        return task_decomposition_output_mapper(task_decomposition_output, state)
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
    
    # Add immunity_plan node (generate experiment plan)
    def run_immunity_plan_node(state: GlobalState) -> GlobalState:
        """
        Generate experiment plan node
        
        Use immunity subgraph to generate executable experiment plan (don't execute tasks)
        """
        from agent.nodes.subagents.immunity.graph import (
            build_immunity_subgraph,
            immunity_input_mapper,
            immunity_output_mapper
        )
        
        # Build immunity subgraph
        immunity_subgraph = build_immunity_subgraph()
        
        # Main graph → subgraph: Map state
        immunity_input = immunity_input_mapper(state)
        
        # Execute subgraph
        immunity_output = immunity_subgraph.invoke(immunity_input)
        
        # Subgraph → main graph: Sync results
        return immunity_output_mapper(immunity_output, state)
    
    main_graph.add_node("immunity_plan", run_immunity_plan_node)
    
    # All processing nodes → end
    main_graph.add_edge("general_qa", END)
    main_graph.add_edge("immunity", END)
    main_graph.add_edge("immunity_plan", END)

    return main_graph.compile()
