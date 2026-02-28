"""
General QA Subgraph

Provides two versions:
1. Simplified (recommended): 6 nodes, fast response, optimized for GLM-4.5
2. Full: 12 nodes with X-Masters, detailed reasoning

Set USE_SIMPLIFIED_QA=false to use the full version.
Default: Simplified version (better for most use cases)
"""

import os

# Configuration flag
USE_SIMPLIFIED_QA = os.getenv("USE_SIMPLIFIED_QA", "true").lower() == "true"

# Always import state
from agent.nodes.subagents.general_qa.state import GeneralQAState

# Conditional import based on configuration
if USE_SIMPLIFIED_QA:
    from agent.nodes.subagents.general_qa.graph_simplified import (
        build_simplified_general_qa_graph as build_general_qa_graph,
        run_simplified_general_qa
    )
    
    # Create compatibility wrappers
    def build_general_qa_subgraph():
        """Build simplified subgraph"""
        return build_general_qa_graph()
    
    def general_qa_input_mapper(state):
        """Map input state (simplified - pass through)"""
        return state
    
    def general_qa_output_mapper(state):
        """Map output state (simplified - extract answer)"""
        return {
            "final_answer": getattr(state, 'final_answer', None),
            "conclusion": getattr(state, 'core_conclusion', None),
            "success": True
        }
    
    # Placeholder graph
    general_qa_graph = None
    
else:
    # Use full version
    from agent.nodes.subagents.general_qa.graph import (
        build_general_qa_graph,
        build_general_qa_subgraph,
        general_qa_graph,
        general_qa_input_mapper,
        general_qa_output_mapper
    )

__all__ = [
    "build_general_qa_graph",
    "build_general_qa_subgraph",
    "general_qa_graph",
    "general_qa_input_mapper",
    "general_qa_output_mapper",
    "GeneralQAState",
    "USE_SIMPLIFIED_QA"
]

