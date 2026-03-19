"""
General QA Subgraph

Provides two versions:
1. Simplified (recommended): 6 nodes, fast response, optimized for GLM-4.5
2. Full: 12 nodes with X-Masters, detailed reasoning

Set USE_SIMPLIFIED_QA=false to use the full version.
Default: Simplified version (better for most use cases)
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional, Callable

# Configuration flag
USE_SIMPLIFIED_QA = os.getenv("USE_SIMPLIFIED_QA", "true").lower() == "true"

# Always import state
from agent.nodes.subagents.general_qa.state import GeneralQAState


def _get_progress_callback_by_session(session_id: Optional[str]) -> Optional[Callable]:
    """
    Get progress callback from global registry by session_id

    Args:
        session_id: Session ID to look up

    Returns:
        Progress callback function if found, None otherwise
    """
    if not session_id:
        return None

    try:
        backend_dir = Path(__file__).parent.parent.parent.parent / "backend"
        project_root = backend_dir.parent

        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from backend import progress_tracker as pt_module

        return pt_module.get_progress_callback(session_id)
    except (ImportError, AttributeError) as e:
        print(f"[GeneralQA] Failed to get callback: {e}")
        return None


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
        return GeneralQAState(
            user_input=global_state.user_input,
            # [FIX] Do NOT pass progress_callback - it cannot be serialized by LangGraph.
            # The callback is retrieved dynamically from global registry via session_id in get_llm().
            session_id=global_state.session_id,
        )

    def general_qa_output_mapper(general_qa_state, global_state):
        """Map General QA subgraph state back to main graph state"""
        if not global_state.merged_result:
            global_state.merged_result = {}

        # [HOT] 优先从N8节点的final_answer提取答案
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

        print(f"[SUCCESS] General QA subgraph completed (simplified)")
        if final_answer:
            print(f"  - Final answer: {str(final_answer)[:200]}...")

            # [HOT] 通过SSE推送最终答案到前端
            if global_state.session_id:
                try:
                    import sys
                    from pathlib import Path

                    backend_dir = (
                        Path(__file__).parent.parent.parent.parent.parent / "backend"
                    )
                    project_root = backend_dir.parent

                    if str(backend_dir) not in sys.path:
                        sys.path.insert(0, str(backend_dir))
                    if str(project_root) not in sys.path:
                        sys.path.insert(0, str(project_root))

                    from backend import progress_tracker as pt_module

                    progress_callback = pt_module.get_progress_callback(
                        global_state.session_id
                    )

                    if progress_callback:
                        progress_callback(
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
