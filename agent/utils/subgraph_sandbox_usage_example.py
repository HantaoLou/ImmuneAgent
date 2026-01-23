"""
Example usage of SubgraphSandboxExecutor

This file demonstrates how to integrate sandbox execution into the main graph.
"""

from typing import Callable
from agent.utils.subgraph_sandbox_executor import (
    SubgraphSandboxExecutor,
    IsolationStrategy,
    execute_subgraph_in_sandbox
)
from agent.state import GlobalState


# ===================== Example 1: Basic Usage =====================

def example_basic_usage():
    """Basic usage example"""
    from agent.nodes.subagents.supervisor.graph import (
        build_supervisor_subgraph,
        supervisor_input_mapper,
        supervisor_output_mapper
    )
    
    main_state = GlobalState(
        user_input="Test input",
        sandbox_dir="/tmp/test_sandbox"
    )
    
    # Create executor
    executor = SubgraphSandboxExecutor(
        strategy=IsolationStrategy.THREAD,
        timeout=60.0,
        auto_cleanup=True
    )
    
    # Execute subgraph
    result = executor.execute_subgraph(
        subgraph_name="supervisor",
        subgraph_builder=build_supervisor_subgraph,
        input_mapper=supervisor_input_mapper,
        output_mapper=supervisor_output_mapper,
        main_state=main_state
    )
    
    if result.success:
        updated_state = result.output_state
        print("✓ Subgraph executed successfully")
    else:
        print(f"✗ Subgraph failed: {result.error}")
        # Main graph continues with original state


# ===================== Example 2: Convenience Function =====================

def example_convenience_function():
    """Using the convenience function"""
    from agent.nodes.subagents.general_qa.graph import (
        build_general_qa_subgraph,
        general_qa_input_mapper,
        general_qa_output_mapper
    )
    
    main_state = GlobalState(
        user_input="What is bioinformatics?",
        sandbox_dir="/tmp/test_sandbox"
    )
    
    # Execute with convenience function
    updated_state = execute_subgraph_in_sandbox(
        subgraph_name="general_qa",
        subgraph_builder=build_general_qa_subgraph,
        input_mapper=general_qa_input_mapper,
        output_mapper=general_qa_output_mapper,
        main_state=main_state,
        strategy=IsolationStrategy.THREAD,
        timeout=30.0
    )
    
    # State is automatically updated or preserved on error
    return updated_state


# ===================== Example 3: Modified Main Graph Node =====================

def supervisor_node_with_sandbox(state: GlobalState) -> GlobalState:
    """
    Modified supervisor node with sandbox execution
    
    This replaces the original supervisor node in main_graph.py
    """
    from agent.nodes.subagents.supervisor.graph import (
        build_supervisor_subgraph,
        supervisor_input_mapper,
        supervisor_output_mapper
    )
    
    # Use sandbox executor
    return execute_subgraph_in_sandbox(
        subgraph_name="supervisor",
        subgraph_builder=build_supervisor_subgraph,
        input_mapper=supervisor_input_mapper,
        output_mapper=supervisor_output_mapper,
        main_state=state,
        strategy=IsolationStrategy.THREAD,  # Use thread isolation (balanced)
        timeout=120.0  # 2 minute timeout
    )


def immunity_node_with_sandbox(state: GlobalState) -> GlobalState:
    """
    Modified immunity node with sandbox execution
    """
    from agent.nodes.subagents.immunity.graph import (
        build_immunity_subgraph,
        immunity_input_mapper,
        immunity_output_mapper
    )
    
    # Use process isolation for immunity (more complex, needs better isolation)
    return execute_subgraph_in_sandbox(
        subgraph_name="immunity",
        subgraph_builder=build_immunity_subgraph,
        input_mapper=immunity_input_mapper,
        output_mapper=immunity_output_mapper,
        main_state=state,
        strategy=IsolationStrategy.PROCESS,  # Full process isolation
        timeout=300.0  # 5 minute timeout
    )


def task_decomposition_node_with_sandbox(state: GlobalState) -> GlobalState:
    """
    Modified task decomposition node with sandbox execution
    """
    from agent.nodes.subagents.task_decomposition.graph import (
        build_task_decomposition_subgraph,
        task_decomposition_input_mapper,
        task_decomposition_output_mapper
    )
    
    # Use exception handling (lightweight, fast)
    return execute_subgraph_in_sandbox(
        subgraph_name="task_decomposition",
        subgraph_builder=build_task_decomposition_subgraph,
        input_mapper=task_decomposition_input_mapper,
        output_mapper=task_decomposition_output_mapper,
        main_state=state,
        strategy=IsolationStrategy.EXCEPTION,  # Lightweight
        timeout=180.0  # 3 minute timeout
    )


# ===================== Example 4: Strategy Selection Based on Subgraph =====================

def get_strategy_for_subgraph(subgraph_name: str) -> IsolationStrategy:
    """
    Select isolation strategy based on subgraph characteristics
    
    - PROCESS: For complex subgraphs that might crash or have import issues
    - THREAD: For most subgraphs (balanced isolation and performance)
    - EXCEPTION: For simple, trusted subgraphs (fastest)
    """
    strategy_map = {
        "supervisor": IsolationStrategy.THREAD,
        "general_qa": IsolationStrategy.EXCEPTION,
        "immunity": IsolationStrategy.PROCESS,
        "task_decomposition": IsolationStrategy.THREAD,
        "executor": IsolationStrategy.PROCESS,
    }
    
    return strategy_map.get(subgraph_name, IsolationStrategy.THREAD)


def execute_subgraph_with_auto_strategy(
    subgraph_name: str,
    subgraph_builder: Callable,
    input_mapper: Callable,
    output_mapper: Callable,
    main_state: GlobalState,
    **kwargs
) -> GlobalState:
    """Execute subgraph with automatically selected strategy"""
    strategy = get_strategy_for_subgraph(subgraph_name)
    
    return execute_subgraph_in_sandbox(
        subgraph_name=subgraph_name,
        subgraph_builder=subgraph_builder,
        input_mapper=input_mapper,
        output_mapper=output_mapper,
        main_state=main_state,
        strategy=strategy,
        **kwargs
    )

