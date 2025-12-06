"""
Gradio Visual Q&A Interface
Supports interactive execution and result download of LangGraph workflows
"""

import asyncio
import json
import queue
import sys
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from usecases.immunity.common.utils import save_planning_report
from usecases.immunity.config.immunity_config import get_runnable_config
from usecases.immunity.graph.planning_graph import (
    ImprovedCellState,
    build_improved_graph,
)


class WorkflowManager:
    """Workflow manager for handling LangGraph workflow execution"""

    def __init__(self):
        self.graph = None
        self.compiled_graph = None
        self._initialize_graph()

    def _initialize_graph(self):
        """Initialize LangGraph"""
        try:
            self.graph = build_improved_graph()
            self.compiled_graph = self.graph.compile()
            print("✅ LangGraph initialization successful")
        except Exception as e:
            print(f"❌ LangGraph initialization failed: {e}")
            raise

    async def execute_workflow(
        self, query: str, progress_callback=None
    ) -> Dict[str, Any]:
        """
        Execute improved workflow

        Args:
            query: Immunology research question
            progress_callback: Progress callback function

        Returns:
            Dictionary containing workflow results
        """
        # Mapping of node names to English descriptions for progress display
        node_descriptions = {
            "query_decomposition": "📝 Query Decomposition & Optimization",
            "immunology_retrieval": "🔍 Immunology Literature Retrieval",
            "deep_research": "🧬 Deep Research Analysis",
            "hypothesis_generation": "💡 Scientific Hypothesis Generation",
            "research_informed_planning": "📋 Experimental Planning",
            "evaluate_planning": "✅ Evaluation & Optimization",
        }

        if progress_callback:
            await progress_callback("🚀 Initializing workflow...")

        # Generate unique thread ID
        timestamp = (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        )
        config = get_runnable_config(thread_id=timestamp)

        # Initialize state
        initial_state = ImprovedCellState(
            original_question=query,
            query=query,
            optimized_questions=[],
            context="",
            individual_plans=[],
            generated_plan="",
            deep_research_findings={},
            hypothesis={},
            research_informed_plan="",
            final_enhanced_plan="",
        )

        try:
            if progress_callback:
                await progress_callback("🚀 Starting workflow...")

            # Use LangGraph's astream method to get real node execution status
            final_result = None
            node_count = 0

            # Stream workflow execution, get real-time node execution status
            async for chunk in self.compiled_graph.astream(
                initial_state.model_dump(),
                config=config,
                stream_mode="updates",  # Get node update information
            ):
                # chunk is a dictionary containing current executing node information
                for node_name, node_output in chunk.items():
                    node_count += 1

                    # Get node description
                    node_desc = node_descriptions.get(
                        node_name, f"🔄 Executing node: {node_name}"
                    )

                    if progress_callback:
                        # Display currently executing node
                        progress_msg = f"[Node {node_count}] {node_desc}"
                        await progress_callback(progress_msg)

                    # Save final result
                    final_result = node_output

            if progress_callback:
                await progress_callback(
                    "✅ Workflow execution completed, generating result files..."
                )

            return {
                "success": True,
                "query": query,
                "result": final_result,
                "timestamp": datetime.now().isoformat(),
                "thread_id": timestamp,
            }

        except Exception as e:
            error_msg = f"❌ Workflow execution failed: {str(e)}"
            if progress_callback:
                await progress_callback(error_msg)

            return {
                "success": False,
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "traceback": traceback.format_exc(),
            }


# Global workflow manager
workflow_manager = WorkflowManager()


# This function has been replaced by handle_submit, kept as backup
# def run_workflow_sync(query: str):
#     """Backup synchronous execution function, currently using handle_submit for real-time updates"""
#     pass


def create_gradio_interface():
    """Create Gradio interface"""

    # Custom CSS styles
    custom_css = """
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .query-input {
        font-size: 16px;
        min-height: 100px;
    }
    .status-output {
        font-family: 'Courier New', monospace;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        max-height: 400px;
        overflow-y: auto;
    }
    .download-section {
        background-color: #e8f5e8;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
    }
    .title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 30px;
    }
    """

    with gr.Blocks(
        title="Immunology Research Q&A System", css=custom_css, theme=gr.themes.Soft()
    ) as interface:
        # Title
        gr.Markdown(
            """
            # 🧬 Immunology Research Q&A System
            """,
            elem_classes=["title"],
        )

        with gr.Row():
            with gr.Column(scale=2):
                # Query input area
                gr.Markdown("### 📝 Enter Your Immunology Research Question")
                query_input = gr.Textbox(
                    label="Research Question",
                    placeholder="Please enter your immunology research question, for example:\n• What molecular programs determine which germinal center B cells become long-lived plasma cells?\n• How do atypical memory B cells provide protection in chronic infections?",
                    lines=4,
                    elem_classes=["query-input"],
                )

                # Submit button
                submit_btn = gr.Button(
                    "🚀 Start Analysis", variant="primary", size="lg"
                )

            with gr.Column(scale=1):
                # Execution status display area (moved to right side)
                gr.Markdown("### 📊 Execution Status")
                status_output = gr.Textbox(
                    label="Real-time Status",
                    value="Waiting for query input...",
                    lines=10,
                    interactive=False,
                    elem_classes=["status-output"],
                )

        # File download area
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    planning_download = gr.File(
                        label="📋 Planning Report (Step 5)",
                        visible=True,  # Modified to visible, allow users to download planning report
                        interactive=False,
                    )
                    evaluation_download = gr.File(
                        label="📊 Evaluation Results (Step 6)",
                        visible=True,  # Modified to visible, allow users to download evaluation results
                        interactive=False,
                    )

        # Bind events - Use generator and queue for real-time status updates
        def handle_submit(query):
            """Handle submit event, use generator and queue for real-time status updates"""
            if not query.strip():
                yield (
                    "❌ Please enter a valid query question",
                    None,
                    None,
                    gr.update(interactive=True),
                )
                return

            # Immediately disable button and update status
            yield "🚀 Starting workflow...", None, None, gr.update(interactive=False)

            # Status message list and queue
            status_messages = []
            status_queue = queue.Queue()

            def update_status_display(message: str):
                """Update status display"""
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_message = f"[{timestamp}] {message}"
                status_messages.append(formatted_message)
                # Display latest 15 messages
                current_status = "\n".join(status_messages[-15:])
                return current_status

            try:
                # Run async workflow in new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    # Create async execution function
                    async def execute_workflow_with_updates():
                        """Execute workflow asynchronously with real-time status updates"""

                        # Define async status update callback
                        async def progress_callback(message: str):
                            """Async progress callback function"""
                            # Put status message into queue
                            status_queue.put(message)
                            return message

                        # Execute workflow
                        result = await workflow_manager.execute_workflow(
                            query, progress_callback
                        )
                        # Mark completion
                        status_queue.put("WORKFLOW_COMPLETE")
                        return result

                    # Start async task
                    import threading

                    result_container = {}

                    def run_async_workflow():
                        """Run async workflow in thread"""
                        try:
                            result = loop.run_until_complete(
                                execute_workflow_with_updates()
                            )
                            result_container["result"] = result
                        except Exception as e:
                            result_container["error"] = e
                        finally:
                            status_queue.put("THREAD_COMPLETE")

                    # Start workflow thread
                    workflow_thread = threading.Thread(target=run_async_workflow)
                    workflow_thread.start()

                    # Real-time monitoring of status updates
                    while True:
                        try:
                            # Non-blocking get status message
                            message = status_queue.get(timeout=0.5)

                            if message == "THREAD_COMPLETE":
                                break
                            elif message == "WORKFLOW_COMPLETE":
                                continue
                            else:
                                # Update status and yield to interface
                                current_status = update_status_display(message)
                                yield (
                                    current_status,
                                    None,
                                    None,
                                    gr.update(interactive=False),
                                )

                        except queue.Empty:
                            # Queue is empty, continue waiting
                            continue

                    # Wait for thread completion
                    workflow_thread.join()

                finally:
                    loop.close()

                # Check execution result
                if "error" in result_container:
                    raise result_container["error"]

                result = result_container["result"]

                if result["success"]:
                    # Update status
                    status = update_status_display(
                        "✅ Workflow execution completed, generating result files..."
                    )
                    yield status, None, None, gr.update(interactive=False)

                    # Save planning report
                    planning_content = ""
                    evaluation_content = ""

                    workflow_result = result["result"]

                    # Extract planning content (Step 5)
                    if "final_enhanced_plan" in workflow_result:
                        planning_content = workflow_result["final_enhanced_plan"]
                    elif "research_informed_plan" in workflow_result:
                        planning_content = workflow_result["research_informed_plan"]
                    elif "generated_plan" in workflow_result:
                        planning_content = workflow_result["generated_plan"]

                    # Extract evaluation content (Step 6)
                    if "final_evaluation" in workflow_result:
                        evaluation_content = workflow_result["final_evaluation"]
                    elif "evaluation_results" in workflow_result:
                        evaluation_content = workflow_result["evaluation_results"]
                    elif "deep_research_findings" in workflow_result:
                        # Format deep research results
                        evaluation_content = f"# Deep Research Results\n\n{workflow_result['deep_research_findings']}"

                    # Generate files
                    planning_file = None
                    evaluation_file = None

                    try:
                        if planning_content:
                            planning_file = save_planning_report(
                                planning_content, result["thread_id"], "planning"
                            )

                        if evaluation_content:
                            evaluation_file = save_planning_report(
                                evaluation_content, result["thread_id"], "evaluation"
                            )

                        final_status = update_status_display(
                            "✅ All files generated successfully!"
                        )
                        yield (
                            final_status,
                            planning_file,
                            evaluation_file,
                            gr.update(interactive=True),
                        )

                    except Exception as file_error:
                        error_status = update_status_display(
                            f"⚠️ File generation partially failed: {str(file_error)}"
                        )
                        # Return None instead of empty string even if file generation fails, to avoid permission errors
                        yield error_status, None, None, gr.update(interactive=True)
                else:
                    error_status = update_status_display(
                        f"❌ Workflow execution failed: {result.get('error', 'Unknown error')}"
                    )
                    yield error_status, None, None, gr.update(interactive=True)

            except Exception as e:
                error_status = f"❌ Error occurred during execution: {str(e)}\n\nDetailed error information:\n{traceback.format_exc()}"
                yield error_status, None, None, gr.update(interactive=True)

        submit_btn.click(
            fn=handle_submit,
            inputs=[query_input],
            outputs=[status_output, planning_download, evaluation_download, submit_btn],
            show_progress=True,
        )

    return interface


def main():
    """Launch Gradio application"""
    print("🚀 Starting Immunology Research Q&A System...")

    try:
        # Create interface
        interface = create_gradio_interface()

        # Launch application
        interface.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=17860,  # Default port
            share=False,  # Don't create public link
            debug=True,  # Enable debug mode
            show_error=True,  # Show error information
            quiet=False,  # Show startup information
            # Add allowed access paths to solve file access permission issues on Linux systems
            allowed_paths=[
                str(
                    Path(__file__).parent.parent.parent.absolute()
                ),  # Project root directory
                "/tmp",  # Linux temporary directory
                str(Path.cwd().absolute()),  # Current working directory
            ],
        )

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
