"""
HITL (Human-in-the-Loop) Node - Handles user interaction and confirmation

This node uses LangGraph's interrupt mechanism to:
1. Send HITL request events to frontend via SSE
2. Pause execution using interrupt() and wait for user response
3. Handle user confirmation/modification when resuming
4. Manage iteration history

Flow:
- First call: Triggers interrupt(), raises exception, LangGraph saves state
- Resume call: interrupt() returns resume value, process user response
"""

from typing import Dict, Any, Optional
from pathlib import Path
import sys
from datetime import datetime
import uuid

# Key fix: use module-level global variable to store progress_callback
# prevent _global_callbacks from being re-initialized
_module_level_progress_callback: Optional[Dict[str, Any]] = None


def _get_module_level_callback() -> Optional[Dict[str, Any]]:
    """Get module-level progress_callback"""
    global _module_level_progress_callback
    return _module_level_progress_callback


def _set_module_level_callback(callbacks: Dict[str, Any]):
    """Set module-level progress_callback"""
    global _module_level_progress_callback
    _module_level_progress_callback = callbacks
    print(
        f"[HITL] Set module-level callbacks: {list(callbacks.keys()) if callbacks else 'None'}"
    )


def _get_callback_for_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get callback for specific session from module-level callbacks"""
    callbacks = _get_module_level_callback()
    if callbacks and session_id in callbacks:
        return callbacks[session_id]
    return None


def _remove_callback_for_session(session_id: str):
    """Remove callback for specific session from module-level callbacks"""
    callbacks = _get_module_level_callback()
    if callbacks and session_id in callbacks:
        del callbacks[session_id]
        print(f"[HITL] Removed module-level callback for session: {session_id}")


agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import GlobalState

try:
    from langgraph.types import interrupt as langgraph_interrupt, Command

    def interrupt(value: Any = None) -> Any:
        """Wrapper for langgraph interrupt with correct signature"""
        return langgraph_interrupt(value)

    INTERRUPT_AVAILABLE = True
except ImportError:
    INTERRUPT_AVAILABLE = False

    def interrupt(value: Any = None) -> Any:
        raise NotImplementedError("interrupt functionality requires LangGraph support")

    Command = None

MAX_HITL_ITERATIONS = 5


def _process_resume_value(resume_value: Any) -> Optional[Dict[str, Any]]:
    """Extract response data from resume value."""
    if resume_value is None:
        return None

    if hasattr(resume_value, "resume"):
        return resume_value.resume
    elif isinstance(resume_value, dict) and "resume" in resume_value:
        return resume_value["resume"]
    elif isinstance(resume_value, dict):
        return resume_value

    return None


async def hitl_node(state: GlobalState) -> GlobalState:
    """
    HITL Interaction Node - Pauses for user confirmation using interrupt.

    This node:
    1. Checks for resume value from previous interrupt
    2. If resuming: process user response and continue
    3. If first call: save state and trigger interrupt to pause execution

    Args:
        state: Current GlobalState

    Returns:
        Updated GlobalState with HITL response
    """
    print(f"\n{'=' * 60}")
    print("[HITL] Starting Human-in-the-Loop interaction...")
    print(f"{'=' * 60}")

    session_id = state.session_id
    iteration = state.hitl_iteration
    max_iterations = MAX_HITL_ITERATIONS

    if iteration >= max_iterations:
        print(f"[HITL] Max iterations ({max_iterations}) reached, auto-confirming")
        state.hitl_confirmed = True
        state.hitl_status = "auto_confirmed"
        return state

    hitl_request = {
        "type": "task_review",
        "session_id": session_id,
        "task_md": state.task_md_content,
        "missing_parameters": state.missing_parameters,
        "iteration": iteration,
        "max_iterations": max_iterations,
        "timestamp": datetime.now().isoformat(),
        "previous_feedback": state.user_feedback,
        "hitl_id": str(uuid.uuid4()),
    }

    state.hitl_request = hitl_request

    if not INTERRUPT_AVAILABLE:
        print("[HITL] Warning: interrupt not available, auto-confirming")
        state.hitl_confirmed = True
        state.hitl_status = "auto_confirmed"
        return state

    # Key fix: call interrupt() first to get resume_value
    # If resume_value exists, this is a resume call, process and return
    # If interrupt() raises exception, this is first call, set hitl_status and re-raise
    resume_value = None
    if INTERRUPT_AVAILABLE:
        try:
            print("[HITL] Calling interrupt()...")
            print(f"[HITL] hitl_request type: {type(hitl_request)}")
            print(
                f"[HITL] hitl_request keys: {hitl_request.keys() if isinstance(hitl_request, dict) else 'N/A'}"
            )
            resume_value = interrupt(hitl_request)
            print(f"[HITL] Got resume value: {resume_value}")
            print(f"[HITL] Resume value type: {type(resume_value)}")
        except Exception as e:
            # On first call, interrupt() raises exception (normal behavior)
            # Set hitl_status and re-raise exception so LangGraph can catch it and pause execution
            state.hitl_status = "waiting"
            print(f"[HITL] Setting hitl_status to 'waiting'")
            print(f"[HITL] Current hitl_iteration: {state.hitl_iteration}")

            # Add to history (only on first call)
            state.hitl_history.append(
                {
                    "iteration": iteration,
                    "request": hitl_request,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            print(
                f"[HITL] Added to hitl_history, total entries: {len(state.hitl_history)}"
            )

            # Push HITL request event (via session_id from global registry)
            if state.session_id:
                try:
                    # Use module-level callback to avoid dynamic imports
                    callbacks = _get_module_level_callback()
                    if callbacks and state.session_id in callbacks:
                        hitl_callback = callbacks[state.session_id]
                        print(
                            f"[HITL] Got progress callback: {hitl_callback is not None}"
                        )
                        if hitl_callback:
                            try:
                                hitl_callback(
                                    event_type="hitl_request",
                                    message="Requesting user confirmation for task plan",
                                    node_name="hitl",
                                    details=hitl_request,
                                )
                                print(f"[HITL] Progress callback called successfully")
                            except Exception as e:
                                print(f"[HITL] Error calling progress_callback: {e}")
                                import traceback

                                traceback.print_exc()
                        else:
                            print(
                                f"[HITL] WARNING: No callback found for session: {state.session_id}"
                            )
                    else:
                        print(f"[HITL] WARNING: No callbacks available")
                except Exception as e:
                    print(f"[HITL] Error getting progress callback: {e}")
                    import traceback

                    traceback.print_exc()
                    import traceback

                    traceback.print_exc()

            # Re-raise exception so LangGraph can catch it and save state
            print(
                f"[HITL] interrupt() raised exception (normal, re-raising): {type(e).__name__}"
            )
            print(f"[HITL] Exception value: {getattr(e, 'value', None)}")
            raise

    # If there's a resume value, this is resuming from interrupt
    if resume_value is not None:
        print("[HITL] Processing resume value (resuming from interrupt)...")

        # Process resume data
        resume_data = _process_resume_value(resume_value)
        print(f"[HITL] Resume data: {resume_data}")

        if resume_data and resume_data.get("type") == "task_review_response":
            confirmed = resume_data.get("confirmed", False)
            feedback = resume_data.get("feedback", "")
            additional_parameters = resume_data.get("parameters", {})

            history_entry = {
                "iteration": iteration,
                "request": hitl_request,
                "response": {
                    "confirmed": confirmed,
                    "feedback": feedback,
                    "parameters": additional_parameters,
                },
                "timestamp": datetime.now().isoformat(),
            }
            state.hitl_history.append(history_entry)

            if confirmed:
                print("[HITL] User confirmed task plan")
                state.hitl_confirmed = True
                state.hitl_status = "confirmed"

                if additional_parameters:
                    state.extracted_parameters.update(additional_parameters)
                    print(
                        f"[HITL] Updated parameters: {list(additional_parameters.keys())}"
                    )

                # Push user confirmation event
                if state.session_id:
                    try:
                        callbacks = _get_module_level_callback()
                        if callbacks and state.session_id in callbacks:
                            hitl_callback = callbacks[state.session_id]
                            if hitl_callback:
                                try:
                                    hitl_callback(
                                        event_type="hitl_confirmed",
                                        message="User confirmed task plan",
                                        node_name="hitl",
                                        details={"iteration": iteration},
                                    )
                                except Exception as e:
                                    print(
                                        f"[HITL] Error calling progress_callback: {e}"
                                    )
                            else:
                                print(f"[HITL] WARNING: hitl_callback is None")
                    except Exception as e:
                        print(f"[HITL] Warning: Could not get progress callback: {e}")
            else:
                print("[HITL] User requested modifications")
                state.hitl_confirmed = False
                state.hitl_status = "needs_modification"
                state.user_feedback = feedback
                state.hitl_iteration = iteration + 1

                # Push user rejection event
                if state.session_id:
                    try:
                        callbacks = _get_module_level_callback()
                        if callbacks and state.session_id in callbacks:
                            hitl_callback = callbacks[state.session_id]
                            if hitl_callback:
                                try:
                                    hitl_callback(
                                        event_type="hitl_rejected",
                                        message="User requested modifications",
                                        node_name="hitl",
                                        details={
                                            "iteration": iteration,
                                            "feedback": feedback[:100]
                                            if feedback
                                            else "",
                                        },
                                    )
                                except Exception as e:
                                    print(
                                        f"[HITL] Error calling progress_callback: {e}"
                                    )
                            else:
                                print(f"[HITL] WARNING: hitl_callback is None")
                    except Exception as e:
                        print(f"[HITL] Warning: Could not get progress callback: {e}")

            print(f"[HITL] HITL interaction complete (iteration {iteration})")
            print(f"  - Status: {state.hitl_status}")
            print(f"  - Confirmed: {state.hitl_confirmed}")
            print("=" * 60)
            return state


def hitl_router(state: GlobalState) -> str:
    """
    Router for HITL node.

    Returns:
        "generate_task" if user rejected and needs modification
        "orchestrator" if user confirmed
    """
    print(
        f"[HITL Router] hitl_confirmed={state.hitl_confirmed}, hitl_status={state.hitl_status}, hitl_iteration={state.hitl_iteration}"
    )

    if state.hitl_confirmed:
        print("[HITL Router] User confirmed, proceeding to orchestrator")
        return "orchestrator"

    if state.hitl_status == "needs_modification":
        print("[HITL Router] User requested modifications, returning to generate_task")
        return "generate_task"

    if state.hitl_status in ("waiting", "waiting_no_interrupt"):
        print(f"[HITL Router] ERROR: Unexpected waiting state in router!")
        print(f"[HITL Router] This means interrupt() did not properly pause the graph")
        print(f"[HITL Router] Checking for saved HITL response...")

        try:
            from backend.checkpointer import get_checkpointer

            checkpointer = get_checkpointer()
            hitl_state = checkpointer.load_hitl_state(state.session_id)

            if hitl_state and hitl_state.get("hitl_response"):
                response = hitl_state["hitl_response"]
                print(
                    f"[HITL Router] Found saved response: confirmed={response.get('confirmed')}"
                )

                if not response.get("confirmed", True):
                    print("[HITL Router] User rejected, returning to generate_task")
                    state.user_feedback = response.get("feedback", "")
                    state.hitl_status = "needs_modification"
                    state.hitl_iteration += 1
                    return "generate_task"
                else:
                    print(
                        "[HITL Router] User confirmed via saved state, proceeding to orchestrator"
                    )
                    state.hitl_confirmed = True
                    state.hitl_status = "confirmed"
                    return "orchestrator"
        except Exception as e:
            print(f"[HITL Router] Error loading HITL state: {e}")

        print("[HITL Router] Fallback: treating as needs_modification to be safe")
        state.hitl_status = "needs_modification"
        return "generate_task"

    print("[HITL Router] Default: proceeding to orchestrator")
    return "orchestrator"


def process_hitl_response(
    state: GlobalState,
    confirmed: bool,
    feedback: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> GlobalState:
    """
    Process user response to HITL request.

    This function is called by the backend when receiving user response.

    Args:
        state: Current GlobalState
        confirmed: Whether user confirmed the task plan
        feedback: User's feedback/modification suggestions
        parameters: Additional parameters provided by user

    Returns:
        Updated GlobalState
    """
    state.hitl_response = {
        "confirmed": confirmed,
        "feedback": feedback or "",
        "parameters": parameters or {},
        "timestamp": datetime.now().isoformat(),
    }

    if confirmed:
        state.hitl_confirmed = True
        state.hitl_status = "confirmed"

        if parameters:
            state.extracted_parameters.update(parameters)
    else:
        state.hitl_confirmed = False
        state.hitl_status = "needs_modification"
        state.user_feedback = feedback
        state.hitl_iteration += 1

    try:
        from backend.checkpointer import get_checkpointer

        checkpointer = get_checkpointer()
        checkpointer.save_hitl_state(
            session_id=state.session_id,
            hitl_response=state.hitl_response,
        )
    except Exception as e:
        print(f"[HITL] Warning: Could not update HITL state: {e}")

    return state
