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
        run_simplified_general_qa,
    )

    # Create compatibility wrappers
    def build_general_qa_subgraph():
        """Build simplified subgraph"""
        return build_general_qa_graph()

    def general_qa_input_mapper(global_state):
        """Map main graph state to General QA subgraph state"""
        # 🔥 传递progress_callback和session_id，确保SSE消息能推送到前端
        return GeneralQAState(
            user_input=global_state.user_input,
            progress_callback=getattr(global_state, "progress_callback", None),
            session_id=getattr(global_state, "session_id", None),
        )

    def general_qa_output_mapper(general_qa_state, global_state):
        """Map General QA subgraph state back to main graph state"""
        if not global_state.merged_result:
            global_state.merged_result = {}

        # 🔥 优先从N8节点的final_answer提取答案
        final_answer = getattr(general_qa_state, "final_answer", None)

        # 如果没有final_answer，尝试从structured_answer中提取
        if not final_answer:
            structured_answer = getattr(general_qa_state, "structured_answer", None)
            if structured_answer and isinstance(structured_answer, dict):
                final_answer = structured_answer.get("final_answer")
                if final_answer:
                    print(
                        f"  [提取] 从structured_answer获取答案: {str(final_answer)[:100]}..."
                    )

        # 如果还是没有，尝试从core_conclusion提取
        if not final_answer:
            final_answer = getattr(general_qa_state, "core_conclusion", None)
            if final_answer:
                print(
                    f"  [提取] 从core_conclusion获取答案: {str(final_answer)[:100]}..."
                )

        # 设置到merged_result
        global_state.merged_result["general_qa_answer"] = final_answer
        global_state.merged_result["general_qa_error"] = getattr(
            general_qa_state, "error_message", None
        )
        global_state.merged_result["general_qa_conclusion"] = getattr(
            general_qa_state, "core_conclusion", None
        )

        print(f"✅ General QA subgraph completed (simplified)")
        if final_answer:
            print(f"  - Final answer: {str(final_answer)[:200]}...")

            # 🔥 通过SSE推送最终答案到前端
            if (
                hasattr(global_state, "progress_callback")
                and global_state.progress_callback
            ):
                try:
                    global_state.progress_callback(
                        event_type="final_answer",
                        message=f"最终答案: {str(final_answer)[:300]}",
                        details={
                            "answer": str(final_answer),
                            "answer_length": len(str(final_answer)),
                            "node": "general_qa_N8",
                        },
                    )
                except Exception as e:
                    print(f"  [警告] 推送答案到SSE失败: {e}")

        return global_state

    # Placeholder graph
    general_qa_graph = None

else:
    # Use full version
    from agent.nodes.subagents.general_qa.graph import (
        build_general_qa_graph,
        build_general_qa_subgraph,
        general_qa_graph,
        general_qa_input_mapper,
        general_qa_output_mapper,
    )

__all__ = [
    "build_general_qa_graph",
    "build_general_qa_subgraph",
    "general_qa_graph",
    "general_qa_input_mapper",
    "general_qa_output_mapper",
    "GeneralQAState",
    "USE_SIMPLIFIED_QA",
]
