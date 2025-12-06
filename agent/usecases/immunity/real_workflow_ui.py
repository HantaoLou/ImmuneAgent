"""
Real Immunology Research Workflow Interface - Integrated with planning_graph.py

Implementation according to the design:
1. Left workflow execution window - progressively displays results of 7 stages
2. Right human-machine interaction control window - handles real tool call confirmations
3. Calls real planning_graph code to execute workflow
"""

import asyncio
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# Add agent path to sys.path
agent_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, agent_path)

try:
    from langchain_core.runnables import RunnableConfig

    from usecases.immunity.config.immunity_config import get_runnable_config
    from usecases.immunity.graph.planning_graph import build_improved_graph
    from usecases.immunity.graph.task_executor import TaskExecutor
    from usecases.immunity.state.state import ImprovedCellState

    print("✅ Successfully imported planning modules")
except ImportError as e:
    print(f"❌ Failed to import planning modules: {e}")

    # Create mock classes to avoid errors
    class ImprovedCellState:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def build_improved_graph():
        return None

    def get_runnable_config(thread_id=None):
        return {"configurable": {"uuid": str(uuid.uuid4())}}


class RealWorkflowManager:
    """
    Real workflow manager - calls code from planning_graph.py
    """

    def __init__(self):
        self.workflow_graph = None
        self.config = None
        self.current_state = None
        self.is_running = False
        self.workflow_thread = None
        self.workflow_history = []
        self.interaction_history = []
        self.pending_interaction = None
        self.current_stage = 0

        # UI interaction related
        self.ui_interaction_enabled = True  # UI interaction mode switch
        self.interaction_queue = []  # Interaction request queue
        self.interaction_response = None  # Interaction response
        self.interaction_event = threading.Event()  # Interaction synchronization event

        # File management related
        self.generated_files = {}  # Store generated file paths {file_id: file_path}
        self.file_counter = 0  # File counter for generating unique IDs
        self.download_buttons = {}  # Store download button components {stage: button_component}
        self.stage_files = {}  # Store file paths for each stage {stage: file_path}

        # 7 real workflow stages
        self.stages = [
            {
                "name": "Query Decomposition",
                "key": "query_decomposition",
                "emoji": "📝",
            },
            {
                "name": "Immunology Retrieval",
                "key": "immunology_retrieval",
                "emoji": "🔍",
            },
            {"name": "Deep Research Analysis", "key": "deep_research", "emoji": "🔬"},
            {
                "name": "Hypothesis Generation",
                "key": "hypothesis_generation",
                "emoji": "💡",
            },
            {
                "name": "Research-Informed Planning",
                "key": "research_informed_planning",
                "emoji": "📋",
            },
            {"name": "Plan Evaluation", "key": "evaluate_planning", "emoji": "📊"},
            {"name": "Task Execution", "key": "task_execution", "emoji": "🚀"},
        ]

        # Initialize workflow graph
        self.initialize()

    def add_generated_file(self, file_path: str) -> str:
        """Add generated file to management list, return file ID"""
        if file_path and os.path.exists(file_path):
            self.file_counter += 1
            file_id = f"file_{self.file_counter}"
            self.generated_files[file_id] = os.path.abspath(file_path)
            return file_id
        return None

    def get_file_path(self, file_id: str) -> Optional[str]:
        """Get file path by file ID"""
        return self.generated_files.get(file_id)

    def set_download_buttons(self, buttons: Dict[str, Any]):
        """Set download button component references"""
        self.download_buttons = buttons

    def update_download_button(self, stage: int, file_path: str) -> str:
        """Update download button for specified stage"""
        if stage in self.download_buttons and file_path and os.path.exists(file_path):
            return file_path
        return None

    def initialize(self) -> bool:
        """Initialize the real workflow graph"""
        try:
            self.workflow_graph = build_improved_graph()
            # Generate a unique UUID as thread_id
            thread_id = str(uuid.uuid4())
            self.config = get_runnable_config(thread_id=thread_id)

            # Add UI interaction callback parameters to config
            if "configurable" not in self.config:
                self.config["configurable"] = {}
            self.config["configurable"]["ui_callback"] = self.ui_interaction_callback
            self.config["configurable"]["ui_interaction_mode"] = (
                self.ui_interaction_enabled
            )

            if self.workflow_graph is None:
                print("⚠️ Workflow graph initialization failed, using simulation mode")
                return False
            print("✅ Workflow graph initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Workflow graph initialization failed: {e}")
            return False

    def ui_interaction_callback(
        self, interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """UI interaction callback function - TaskExecutor calls this function for UI interaction"""
        try:
            # 检查交互类型
            interaction_type = interaction_data.get("type", "tool_confirmation")

            if interaction_type == "tool_result":
                # 处理工具执行结果显示
                self._handle_tool_result_display(interaction_data)
                return {"action": "acknowledged"}  # 返回确认，不需要用户响应

            # 原有的工具确认逻辑
            # Add interaction request to queue
            self.interaction_queue.append(interaction_data)

            # Update interaction history, display tool call confirmation request
            timestamp = datetime.now().strftime("%H:%M:%S")
            tool_name = interaction_data.get("tool_name", "Unknown Tool")
            parameters = interaction_data.get("parameters", {})

            # Format parameter display - 使用HTML格式和特殊CSS类，以JSON格式显示参数
            interaction_msg = f"""<div class="tool-confirmation-message">
<strong>[{timestamp}] Tool Call Confirmation</strong><br>
<strong>Tool:</strong> {tool_name}<br>
"""

            if parameters:
                # 将参数转换为格式化的JSON字符串
                import json

                try:
                    # 格式化JSON，缩进2个空格，确保中文字符正常显示
                    json_params = json.dumps(parameters, indent=2, ensure_ascii=False)
                    # 将JSON字符串转换为HTML格式，保持缩进
                    json_params_html = json_params.replace("\n", "<br>").replace(
                        " ", "&nbsp;"
                    )
                    interaction_msg += f"<strong>Parameters:</strong><br><code style='background-color: #1a202c; color: #e2e8f0; padding: 12px; display: block; border-radius: 6px; border: 1px solid #4a5568; font-family: monospace; font-size: 12px; line-height: 1.5;'>{json_params_html}</code><br>"
                except Exception as e:
                    # 如果JSON序列化失败，回退到原来的显示方式
                    interaction_msg += "<strong>Parameters:</strong><br>"
                    for key, value in parameters.items():
                        interaction_msg += f"&nbsp;&nbsp;{key}: {value}<br>"
            else:
                # 如果没有参数，显示空的JSON对象
                interaction_msg += f"<strong>Parameters:</strong><br><code style='background-color: #1a202c; color: #e2e8f0; padding: 12px; display: block; border-radius: 6px; border: 1px solid #4a5568; font-family: monospace; font-size: 12px; line-height: 1.5;'>{{}}</code><br>"

            interaction_msg += "<strong>Please select action...</strong>"
            interaction_msg += "</div>"

            self.interaction_history.append([None, interaction_msg])

            # Clear previous response and event
            self.interaction_response = None
            self.interaction_event.clear()

            # Wait for user response (maximum 60 seconds)
            if self.interaction_event.wait(timeout=300):
                response = self.interaction_response
                if response:
                    # Record user choice
                    action_text = {
                        "accept": "Confirm Execution",
                        "modify": "Modify Parameters",
                        "reject": "Reject Execution",
                        "skip": "Skip Task",
                    }.get(response.get("action"), "Unknown Action")

                    timestamp = datetime.now().strftime("%H:%M:%S")
                    choice_msg = f"[{timestamp}] 👤 User Choice: {action_text}"
                    self.interaction_history.append([choice_msg, None])

                    return response

            # Timeout or no response, default skip
            timestamp = datetime.now().strftime("%H:%M:%S")
            timeout_msg = (
                f"[{timestamp}] ⏰ Interaction timeout, automatically skipping task"
            )
            self.interaction_history.append([None, timeout_msg])
            return {"action": "skip"}

        except Exception as e:
            print(f"UI interaction callback error: {e}")
            return {"action": "skip"}

    def _handle_tool_result_display(self, result_data: Dict[str, Any]):
        """
        处理工具执行结果显示

        Args:
            result_data: 工具结果数据
        """
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            tool_name = result_data.get("tool_name", "Unknown Tool")
            args = result_data.get("args", {})
            result = result_data.get("result", "No result")
            status = result_data.get("status", "completed")

            # 格式化结果显示 - 完整显示，不截断
            result_str = str(result)
            
            # 获取合并后的CSV路径
            merged_csv_path = result_data.get("merged_csv_path")

            # 构造结果消息 - 使用HTML格式和特殊CSS类
            result_msg = f"""<div class="tool-result-message">
<strong>[{timestamp}] Tool Execution Result</strong><br>
<strong>Tool:</strong> {tool_name}<br>
"""

            # 显示所有参数，不限制
            if args:
                result_msg += "<strong>Arguments:</strong><br>"
                for key, value in args.items():
                    result_msg += f"&nbsp;&nbsp;{key}: {value}<br>"

            result_msg += f"<strong>Result:</strong><br>{result_str}"
            
            # 如果有合并后的CSV路径，显示给用户
            if merged_csv_path:
                result_msg += f"<br><strong>📊 合并后的CSV文件:</strong><br>"
                result_msg += f"&nbsp;&nbsp;📁 <code>{merged_csv_path}</code><br>"
                result_msg += f"&nbsp;&nbsp;💡 所有工具产生的CSV/Excel数据已合并到此文件中"
            
            result_msg += "</div>"

            # 添加分隔符
            separator_msg = '<hr class="message-separator">'

            # 添加到交互历史中（显示在右侧对话窗口）
            self.interaction_history.append([None, result_msg])
            self.interaction_history.append([None, separator_msg])

            print(f"✅ 工具结果已显示: {tool_name}")

        except Exception as e:
            print(f"Error handling tool result display: {e}")
            error_msg = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error displaying tool result: {str(e)}"
            self.interaction_history.append([None, error_msg])

    def display_tool_results_in_chat(
        self, tool_results: List[Dict[str, Any]], task_description: str = ""
    ):
        """
        将工具执行结果显示到右侧交互对话窗口

        Args:
            tool_results: 工具执行结果列表
            task_description: 任务描述
        """
        if not tool_results:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")

        # 创建工具结果摘要消息
        summary_msg = f"[{timestamp}] ✅ Task Execution Completed"
        if task_description:
            summary_msg += f"\nTask: {task_description}"
        summary_msg += f"\n🔧 {len(tool_results)} tools executed successfully:"

        # 添加每个工具的详细结果
        for i, tool_result in enumerate(tool_results, 1):
            tool_name = tool_result.get("tool_name", "Unknown Tool")
            result = tool_result.get("result", "No result")
            args = tool_result.get("args", {})
            merged_csv_path = tool_result.get("merged_csv_path")

            # 格式化结果显示
            result_str = str(result)
            if len(result_str) > 300:
                result_str = result_str[:300] + "... (truncated)"

            tool_msg = f"\n\n📋 Tool {i}: {tool_name}"
            if args:
                # 只显示关键参数
                key_args = {}
                for key, value in args.items():
                    if key in [
                        "input_file",
                        "output_file",
                        "input_file_path",
                        "output_file_path",
                        "data_path",
                        "model",
                    ]:
                        key_args[key] = value
                if key_args:
                    args_str = str(key_args)
                    if len(args_str) > 150:
                        args_str = args_str[:150] + "..."
                    tool_msg += f"\n   Args: {args_str}"

            tool_msg += f"\n   Result: {result_str}"
            
            # 如果有合并后的CSV路径，显示给用户
            if merged_csv_path:
                tool_msg += f"\n   📊 合并后的CSV文件: {merged_csv_path}"
            
            summary_msg += tool_msg

        # 添加到交互历史中（显示在右侧对话窗口）
        self.interaction_history.append([None, summary_msg])

        print(f"✅ 工具结果已添加到交互历史，共 {len(tool_results)} 个工具")

    def set_interaction_response(self, response: Dict[str, Any]):
        """Set interaction response - called by Gradio interface"""
        self.interaction_response = response
        self.interaction_event.set()

    def get_pending_interaction(self) -> Optional[Dict[str, Any]]:
        """Get pending interaction request"""
        if self.interaction_queue:
            return self.interaction_queue[-1]  # Return the latest interaction request
        return None

    def start_workflow(self, question: str) -> bool:
        """Start the real workflow"""
        if self.is_running:
            return False

        try:
            self.is_running = True
            self.current_stage = 0
            self.workflow_history = []
            self.interaction_history = []
            self.pending_interaction = None
            self.stage_files = {}  # Clear previous file records

            # Create initial state
            self.current_state = ImprovedCellState(
                original_question=question,
                optimized_questions=[],
                context="",
                retrieval_docs=[],
                citations=[],
                research_summary="",
                hypothesis_summary="",
                final_enhanced_plan="",
                final_evaluation="",
                decomposed_tasks=[],
            )

            # Add user question
            self.workflow_history.append([question, None])
            self.interaction_history.append([None, "🤖 Workflow is starting..."])

            # Run real workflow in background thread
            self.workflow_thread = threading.Thread(
                target=self._run_real_workflow, daemon=True
            )
            self.workflow_thread.start()

            return True

        except Exception as e:
            print(f"❌ Failed to start workflow: {e}")
            self.is_running = False
            return False

    def _run_real_workflow(self):
        """Run real workflow in background thread"""
        try:
            if self.workflow_graph is None:
                # Simulation mode
                self._run_simulated_workflow()
                return

            # Reference correct calling method from gradio_interface.py
            # Use LangGraph's compiled_graph.astream method instead of manually calling each stage
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Compile workflow graph
                compiled_graph = self.workflow_graph.compile()

                # Create initial state (using model_dump() method)
                initial_state_dict = (
                    self.current_state.__dict__
                    if hasattr(self.current_state, "__dict__")
                    else {
                        "original_question": self.current_state.original_question,
                        "optimized_questions": getattr(
                            self.current_state, "optimized_questions", []
                        ),
                        "context": getattr(self.current_state, "context", ""),
                        "retrieval_docs": getattr(
                            self.current_state, "retrieval_docs", []
                        ),
                        "citations": getattr(self.current_state, "citations", []),
                        "research_summary": getattr(
                            self.current_state, "research_summary", ""
                        ),
                        "hypothesis_summary": getattr(
                            self.current_state, "hypothesis_summary", ""
                        ),
                        "final_enhanced_plan": getattr(
                            self.current_state, "final_enhanced_plan", ""
                        ),
                        "final_evaluation": getattr(
                            self.current_state, "final_evaluation", ""
                        ),
                        "decomposed_tasks": getattr(
                            self.current_state, "decomposed_tasks", []
                        ),
                    }
                )

                # Asynchronously execute workflow in streaming mode
                async def execute_workflow_stream():
                    """Asynchronously execute workflow stream"""
                    final_result = None
                    stage_count = 0

                    async for chunk in compiled_graph.astream(
                        initial_state_dict,
                        config=self.config,
                        stream_mode="updates",  # Get node update information
                    ):
                        if not self.is_running:
                            break

                        # chunk is a dictionary containing current executing node information
                        for node_name, node_output in chunk.items():
                            stage_count += 1

                            # Get stage information
                            stage_info = None
                            for i, stage in enumerate(self.stages, 1):
                                if stage["key"] == node_name:
                                    stage_info = stage
                                    self.current_stage = i
                                    break

                            if stage_info:
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                start_msg = f"[{timestamp}] {stage_info['emoji']} Starting stage {self.current_stage}: {stage_info['name']}"
                                self.workflow_history.append([None, start_msg])

                                # Simulate execution time (optional)
                                await asyncio.sleep(1)

                                # Complete stage
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                complete_msg = f"[{timestamp}] ✅ Stage {self.current_stage} completed: {stage_info['name']}"

                                # Add stage result details - get actual file paths from state
                                try:
                                    # node_output may be dict or object, need compatible handling
                                    if isinstance(node_output, dict):
                                        # If it's a dict, access keys directly
                                        if (
                                            self.current_stage == 1
                                            and "optimized_questions" in node_output
                                            and node_output["optimized_questions"]
                                        ):
                                            questions_text = "\n".join(
                                                [
                                                    f"  {i + 1}. {q}"
                                                    for i, q in enumerate(
                                                        node_output[
                                                            "optimized_questions"
                                                        ]
                                                    )
                                                ]
                                            )
                                            complete_msg += f"\n📋 Generated {len(node_output['optimized_questions'])} optimized questions:\n{questions_text}"
                                        elif (
                                            self.current_stage == 2
                                            and "retrieval_report_path" in node_output
                                            and node_output["retrieval_report_path"]
                                        ):
                                            # Stage2: Retrieval stage
                                            complete_msg += (
                                                f"\n📚 Retrieved relevant literature"
                                            )
                                            file_name = os.path.basename(
                                                node_output["retrieval_report_path"]
                                            )
                                            complete_msg += (
                                                f"\n📄 {file_name} (Generated)"
                                            )
                                            # Save file path for download button use
                                            self.stage_files[2] = node_output[
                                                "retrieval_report_path"
                                            ]
                                        elif (
                                            self.current_stage == 3
                                            and "research_report_path" in node_output
                                            and node_output["research_report_path"]
                                        ):
                                            # Stage3: Deep research analysis
                                            complete_msg += f"\n🔬 Completed deep research analysis, extracted key insights"
                                            file_name = os.path.basename(
                                                node_output["research_report_path"]
                                            )
                                            complete_msg += (
                                                f"\n📄 {file_name} (Generated)"
                                            )
                                            # Save file path for download button use
                                            self.stage_files[3] = node_output[
                                                "research_report_path"
                                            ]
                                        elif (
                                            self.current_stage == 4
                                            and "hypothesis_report_path" in node_output
                                            and node_output["hypothesis_report_path"]
                                        ):
                                            # Stage4: Hypothesis generation
                                            complete_msg += f"\n💡 Generated scientific hypotheses and validation plans"
                                            file_name = os.path.basename(
                                                node_output["hypothesis_report_path"]
                                            )
                                            complete_msg += (
                                                f"\n📄 {file_name} (Generated)"
                                            )
                                            # Save file path for download button use
                                            self.stage_files[4] = node_output[
                                                "hypothesis_report_path"
                                            ]
                                        elif (
                                            self.current_stage == 5
                                            and "planning_report_path" in node_output
                                            and node_output["planning_report_path"]
                                        ):
                                            # Stage5: Research planning
                                            complete_msg += (
                                                f"\n📋 Developed detailed research plan"
                                            )
                                            file_name = os.path.basename(
                                                node_output["planning_report_path"]
                                            )
                                            complete_msg += (
                                                f"\n📄 {file_name} (Generated)"
                                            )
                                            # Save file path for download button use
                                            self.stage_files[5] = node_output[
                                                "planning_report_path"
                                            ]
                                        elif (
                                            self.current_stage == 6
                                            and "evaluation_report_path" in node_output
                                            and node_output["evaluation_report_path"]
                                        ):
                                            # Stage6: Plan evaluation
                                            complete_msg += f"\n📊 Completed plan evaluation and quality check"
                                            file_name = os.path.basename(
                                                node_output["evaluation_report_path"]
                                            )
                                            complete_msg += (
                                                f"\n📄 {file_name} (Generated)"
                                            )
                                            # Save file path for download button use
                                            self.stage_files[6] = node_output[
                                                "evaluation_report_path"
                                            ]
                                        elif (
                                            self.current_stage == 7
                                            and "decomposed_tasks" in node_output
                                            and node_output["decomposed_tasks"]
                                        ):
                                            complete_msg += f"\n🚀 Executed {len(node_output['decomposed_tasks'])} tasks"

                                            # 新增：检查并显示工具执行结果到右侧交互窗口
                                            if (
                                                "tool_results" in node_output
                                                and node_output["tool_results"]
                                            ):
                                                tool_results = node_output[
                                                    "tool_results"
                                                ]
                                                complete_msg += f"\n🔧 {len(tool_results)} tools executed (details in chat window)"

                                                # 显示工具结果到右侧交互对话窗口
                                                task_desc = f"Stage 7 - Task Execution ({len(node_output['decomposed_tasks'])} tasks)"
                                                self.display_tool_results_in_chat(
                                                    tool_results, task_desc
                                                )
                                    else:
                                        # If it's an object, use hasattr and getattr
                                        if (
                                            self.current_stage == 1
                                            and hasattr(
                                                node_output, "optimized_questions"
                                            )
                                            and node_output.optimized_questions
                                        ):
                                            questions_text = "\n".join(
                                                [
                                                    f"  {i + 1}. {q}"
                                                    for i, q in enumerate(
                                                        node_output.optimized_questions
                                                    )
                                                ]
                                            )
                                            complete_msg += f"\n📋 Generated {len(node_output.optimized_questions)} optimized questions:\n{questions_text}"
                                        elif (
                                            self.current_stage == 2
                                            and hasattr(
                                                node_output, "retrieval_report_path"
                                            )
                                            and node_output.retrieval_report_path
                                        ):
                                            # Stage2: Retrieval stage
                                            complete_msg += (
                                                f"\n📚 Retrieved relevant literature"
                                            )
                                            file_name = os.path.basename(
                                                node_output.retrieval_report_path
                                            )
                                            complete_msg += (
                                                f"\n📄 {file_name} (Generated)"
                                            )
                                            # Save file path for download button use
                                            self.stage_files[2] = (
                                                node_output.retrieval_report_path
                                            )
                                        elif (
                                            self.current_stage == 3
                                            and hasattr(
                                                node_output, "research_report_path"
                                            )
                                            and node_output.research_report_path
                                        ):
                                            # Stage3: Deep research analysis
                                            complete_msg += f"\n🔬 Completed deep research analysis, extracted key insights"
                                            file_name = os.path.basename(
                                                node_output.research_report_path
                                            )
                                            complete_msg += (
                                                f"\n📄 {file_name} (Generated)"
                                            )
                                            # Save file path for download button use
                                            self.stage_files[3] = (
                                                node_output.research_report_path
                                            )
                                        elif (
                                            self.current_stage == 4
                                            and hasattr(
                                                node_output, "hypothesis_report_path"
                                            )
                                            and node_output.hypothesis_report_path
                                        ):
                                            # Stage4: Hypothesis generation
                                            complete_msg += f"\n💡 Generated scientific hypotheses and validation plans"
                                            file_name = os.path.basename(
                                                node_output.hypothesis_report_path
                                            )
                                            complete_msg += (
                                                f"\n📄 {file_name} (Generated)"
                                            )
                                            # Save file path for download button use
                                            self.stage_files[4] = (
                                                node_output.hypothesis_report_path
                                            )
                                        elif (
                                            self.current_stage == 5
                                            and hasattr(
                                                node_output, "planning_report_path"
                                            )
                                            and node_output.planning_report_path
                                        ):
                                            # Stage5: Research planning
                                            complete_msg += (
                                                f"\n📋 Developed detailed research plan"
                                            )
                                            file_name = os.path.basename(
                                                node_output.planning_report_path
                                            )
                                            complete_msg += (
                                                f"\n📄 {file_name} (Generated)"
                                            )
                                            # Save file path for download button use
                                            self.stage_files[5] = (
                                                node_output.planning_report_path
                                            )
                                        elif (
                                            self.current_stage == 6
                                            and hasattr(
                                                node_output, "evaluation_report_path"
                                            )
                                            and node_output.evaluation_report_path
                                        ):
                                            # Stage6: Plan evaluation
                                            complete_msg += f"\n📊 Completed plan evaluation and quality check"
                                            file_name = os.path.basename(
                                                node_output.evaluation_report_path
                                            )
                                            complete_msg += (
                                                f"\n📄 {file_name} (Generated)"
                                            )
                                            # Save file path for download button use
                                            self.stage_files[6] = (
                                                node_output.evaluation_report_path
                                            )
                                        elif (
                                            self.current_stage == 7
                                            and hasattr(node_output, "decomposed_tasks")
                                            and node_output.decomposed_tasks
                                        ):
                                            complete_msg += f"\n🚀 Executed {len(node_output.decomposed_tasks)} tasks"

                                            # 新增：检查并显示工具执行结果到右侧交互窗口
                                            if (
                                                hasattr(node_output, "tool_results")
                                                and node_output.tool_results
                                            ):
                                                tool_results = node_output.tool_results
                                                complete_msg += f"\n🔧 {len(tool_results)} tools executed (details in chat window)"

                                                # 显示工具结果到右侧交互对话窗口
                                                task_desc = f"Stage 7 - Task Execution ({len(node_output.decomposed_tasks)} tasks)"
                                                self.display_tool_results_in_chat(
                                                    tool_results, task_desc
                                                )

                                except Exception as e:
                                    # If attribute access fails, add debug information
                                    print(
                                        f"Debug info - Stage {self.current_stage}: node_output type={type(node_output)}, error={e}"
                                    )
                                    if hasattr(node_output, "__dict__"):
                                        print(
                                            f"Object attributes: {list(node_output.__dict__.keys())}"
                                        )
                                    elif isinstance(node_output, dict):
                                        print(
                                            f"Dictionary keys: {list(node_output.keys())}"
                                        )

                                self.workflow_history.append([None, complete_msg])

                            # Save final result
                            final_result = node_output

                            # Update current state
                            if hasattr(node_output, "__dict__"):
                                for key, value in node_output.__dict__.items():
                                    if hasattr(self.current_state, key):
                                        setattr(self.current_state, key, value)

                    return final_result

                # Run asynchronous workflow
                final_result = loop.run_until_complete(execute_workflow_stream())

                # Workflow completed
                if self.is_running and final_result:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    final_msg = f"[{timestamp}] 🎉 Workflow execution completed!"
                    self.workflow_history.append([None, final_msg])
                    self.interaction_history.append(
                        [None, "✅ All stages completed, workflow finished"]
                    )

            finally:
                loop.close()

        except Exception as e:
            error_msg = f"❌ Workflow execution failed: {str(e)}"
            self.workflow_history.append([None, error_msg])
            print(f"Workflow execution error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.is_running = False
            self.current_stage = 0

    def _run_simulated_workflow(self):
        """Run simulated workflow (when real modules are not available)"""
        for i, stage in enumerate(self.stages, 1):
            if not self.is_running:
                break

            self.current_stage = i
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Start stage
            start_msg = f"[{timestamp}] {stage['emoji']} Starting stage {i}: {stage['name']} (Simulation Mode)"
            self.workflow_history.append([None, start_msg])

            # Simulate execution time
            time.sleep(2 + i)  # Different execution times for different stages

            # Complete stage
            if self.is_running:
                timestamp = datetime.now().strftime("%H:%M:%S")
                complete_msg = f"[{timestamp}] ✅ Stage {i} completed: {stage['name']} (Simulation)"
                self.workflow_history.append([None, complete_msg])

                # Simulate human-machine interaction at stages 4 and 7
                if i in [4, 7]:
                    self._simulate_interaction(i)

        # Complete
        if self.is_running:
            timestamp = datetime.now().strftime("%H:%M:%S")
            final_msg = (
                f"[{timestamp}] 🎉 Workflow execution completed! (Simulation Mode)"
            )
            self.workflow_history.append([None, final_msg])
            self.interaction_history.append(
                [None, "✅ All stages completed, workflow finished"]
            )

    def _simulate_task_execution_interaction(self):
        """Simulate human-machine interaction for task execution stage"""
        if not self.is_running:
            return

        # Simulate tool call requiring confirmation
        tool_info = {
            "name": "task_executor",
            "args": {"execution_mode": "interactive", "timeout": 300},
        }

        self.pending_interaction = tool_info
        timestamp = datetime.now().strftime("%H:%M:%S")
        interaction_msg = (
            f"[{timestamp}] 🔧 Tool call awaiting confirmation: {tool_info['name']}"
        )
        self.interaction_history.append([None, interaction_msg])

        # Wait for user response (maximum 30 seconds)
        wait_time = 0
        while self.pending_interaction and wait_time < 30 and self.is_running:
            time.sleep(1)
            wait_time += 1

        # If timeout, automatically use default parameters
        if self.pending_interaction and self.is_running:
            self.pending_interaction = None
            timeout_msg = f"[{datetime.now().strftime('%H:%M:%S')}] ⏰ Timeout, continuing with default parameters"
            self.interaction_history.append([None, timeout_msg])

    def _simulate_interaction(self, stage_num: int):
        """Simulate human-machine interaction"""
        tools = {
            4: {
                "name": "hypothesis_generator",
                "args": {"confidence_threshold": 0.8, "max_hypotheses": 5},
            },
            7: {
                "name": "task_executor",
                "args": {"execution_mode": "interactive", "timeout": 300},
            },
        }

        if stage_num in tools:
            tool_info = tools[stage_num]
            self.pending_interaction = tool_info

            timestamp = datetime.now().strftime("%H:%M:%S")
            interaction_msg = (
                f"[{timestamp}] 🔧 Tool call awaiting confirmation: {tool_info['name']}"
            )
            self.interaction_history.append([None, interaction_msg])

            # Wait for user response (maximum 30 seconds)
            wait_time = 0
            while self.pending_interaction and wait_time < 30 and self.is_running:
                time.sleep(1)
                wait_time += 1

            # If timeout, automatically use default parameters
            if self.pending_interaction and self.is_running:
                self.pending_interaction = None
                timeout_msg = f"[{datetime.now().strftime('%H:%M:%S')}] ⏰ Timeout, continuing with default parameters"
                self.interaction_history.append([None, timeout_msg])

    def handle_interaction(self, action: str, params: str = "") -> bool:
        """Handle human-machine interaction"""
        if not self.pending_interaction:
            return False

        timestamp = datetime.now().strftime("%H:%M:%S")

        if action == "accept":
            if params.strip():
                action_msg = f"[{timestamp}] 👤 User choice: Execute with modifications | Parameters: {params}"
            else:
                action_msg = f"[{timestamp}] 👤 User choice: Default execution"
        elif action == "skip_tool":
            action_msg = f"[{timestamp}] 👤 User choice: Skip tool"
        elif action == "skip_task":
            action_msg = f"[{timestamp}] 👤 User choice: Skip task"
        else:
            action_msg = f"[{timestamp}] 👤 User choice: Reject execution"

        self.interaction_history.append([action_msg, None])

        # Continue execution
        continue_msg = f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Action processed, continuing workflow execution..."
        self.interaction_history.append([None, continue_msg])

        self.pending_interaction = None
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "is_running": self.is_running,
            "current_stage": self.current_stage,
            "total_stages": len(self.stages),
            "workflow_history": self.workflow_history.copy(),
            "interaction_history": self.interaction_history.copy(),
            "pending_interaction": self.pending_interaction,
            "current_state": self.current_state,
        }

    def stop_workflow(self):
        """Stop workflow"""
        self.is_running = False
        if self.workflow_thread and self.workflow_thread.is_alive():
            self.workflow_thread.join(timeout=2.0)


# Global workflow manager
workflow_manager = RealWorkflowManager()


def create_real_workflow_interface():
    """Create real workflow interface - designed according to mockup"""

    # CSS styles and JavaScript download functions according to mockup
    custom_css = """
    /* Global styles */
    .gradio-container {
        max-width: none !important;
        padding: 0 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Top header bar - dark blue background */
    .main-header {
        background: #2c3e50;
        color: white;
        padding: 15px 20px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        margin: 0;
    }
    
    /* Main container */
    .main-container {
        display: flex;
        height: 85vh;
        background: #f5f6fa;
        gap: 0;
    }
    
    /* Left workflow panel */
    .workflow-panel {
        flex: 1;
        background: white;
        border-right: 1px solid #ddd;
        display: flex;
        flex-direction: column;
    }
    
    .workflow-header {
        background: #4a90e2;
        color: white;
        padding: 12px 20px;
        font-size: 14px;
        font-weight: bold;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #28a745;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Right interaction panel */
    .interaction-panel {
        flex: 1;
        background: white;
        display: flex;
        flex-direction: column;
    }
    
    .interaction-header {
        background: #28a745;
        color: white;
        padding: 12px 20px;
        font-size: 14px;
        font-weight: bold;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Chat message styles */
    .message-user {
        background: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 8px;
        padding: 12px;
        margin: 8px;
        max-width: 80%;
        align-self: flex-end;
    }
    
    .message-bot {
        background: #f1f8e9;
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 12px;
        margin: 8px;
        max-width: 80%;
        align-self: flex-start;
    }
    
    .message-system {
        background: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 8px;
        padding: 12px;
        margin: 8px;
        max-width: 80%;
        align-self: flex-start;
    }
    
    /* 工具执行结果样式 - 深色背景，红色边框 */
    .tool-result-message {
        background: #2d3748 !important;
        color: #e2e8f0 !important;
        border: 2px solid #e53e3e !important;
        border-radius: 8px;
        padding: 15px;
        margin: 12px 8px;
        max-width: 90%;
        align-self: flex-start;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* 工具调用确认样式 - 深色背景，橙色边框 */
    .tool-confirmation-message {
        background: #2d3748 !important;
        color: #e2e8f0 !important;
        border: 2px solid #dd6b20 !important;
        border-radius: 8px;
        padding: 15px;
        margin: 12px 8px;
        max-width: 90%;
        align-self: flex-start;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* 消息分隔符 */
    .message-separator {
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #cbd5e0, transparent);
        margin: 16px 0;
        border: none;
    }
    
    /* Input area */
    .input-section {
        padding: 15px;
        border-top: 1px solid #eee;
        background: #fafafa;
    }
    
    /* Button styles */
    .btn-group {
        display: flex;
        gap: 8px;
        padding: 15px;
        background: #f8f9fa;
        border-top: 1px solid #dee2e6;
        flex-wrap: wrap;
    }
    
    .btn-primary {
        background: #28a745 !important;
        border: none !important;
        color: white !important;
        padding: 8px 16px !important;
        border-radius: 4px !important;
        font-size: 12px !important;
        font-weight: bold !important;
    }
    
    .btn-secondary {
        background: #ffc107 !important;
        border: none !important;
        color: white !important;
        padding: 8px 16px !important;
        border-radius: 4px !important;
        font-size: 12px !important;
        font-weight: bold !important;
    }
    
    .btn-outline {
        background: #6c757d !important;
        border: none !important;
        color: white !important;
        padding: 8px 16px !important;
        border-radius: 4px !important;
        font-size: 12px !important;
        font-weight: bold !important;
    }
    
    .btn-danger {
        background: #dc3545 !important;
        border: none !important;
        color: white !important;
        padding: 8px 16px !important;
        border-radius: 4px !important;
        font-size: 12px !important;
        font-weight: bold !important;
    }
    
    /* Status information */
    .status-info {
        background: #e9ecef;
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 10px;
        margin: 10px 0;
        font-size: 11px;
        color: #495057;
    }
    
    /* Parameter input box */
    .param-input {
        font-family: 'Courier New', monospace;
        font-size: 11px;
        background: white;
        border: 1px solid #ced4da;
        border-radius: 4px;
    }
    """

    # Simplified CSS styles (JavaScript removed)
    download_css = """
    <style>
    .download-link:hover {
        background: #0056b3 !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Scrollbar styles - using more forceful selectors */
    [id="button-column"],
    [id="button-column"] > *,
    [class*="scrollable-column"],
    [class*="scrollable-column"] > *,
    div[id="button-column"],
    div[class*="scrollable-column"] {
        max-height: 120px !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-right: 5px !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
    }
    
    /* More forceful styles applied to all possible Gradio-generated elements */
    .gradio-container [id="button-column"],
    .gradio-container [class*="scrollable-column"],
    .gradio-container div[id="button-column"],
    .gradio-container div[class*="scrollable-column"] {
        max-height: 120px !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    
    /* Custom scrollbar styles */
    #button-column::-webkit-scrollbar,
    .scrollable-column::-webkit-scrollbar {
        width: 6px;
    }
    
    #button-column::-webkit-scrollbar-track,
    .scrollable-column::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 3px;
    }
    
    #button-column::-webkit-scrollbar-thumb,
    .scrollable-column::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 3px;
    }
    
    #button-column::-webkit-scrollbar-thumb:hover,
    .scrollable-column::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    </style>
    """

    with gr.Blocks(css=custom_css, title="Immunology Research Workflow") as demo:
        # Add download styles
        gr.HTML(download_css)

        # Top header
        gr.HTML(
            '<div class="main-header">Immunology Research Workflow - Dual Chat Window Mode</div>'
        )

        # State variables
        workflow_state = gr.State({})

        # Main container
        with gr.Row(elem_classes=["main-container"]):
            # Left workflow panel
            with gr.Column(scale=1, elem_classes=["workflow-panel"]):
                gr.HTML(
                    '<div class="workflow-header">🔬 Workflow Execution <div class="status-dot"></div> Running</div>'
                )

                # Workflow chat and download buttons combined area
                with gr.Row():
                    # Chat window
                    with gr.Column(scale=3):
                        workflow_chatbot = gr.Chatbot(
                            label="",
                            height=500,
                            type="messages",  # 修复deprecated警告：使用messages格式替代tuples
                            show_label=False,
                            elem_classes=["workflow-chat"],
                        )
                # Input area
                with gr.Row(elem_classes=["input-section"], height=170):
                    # First Column: Input box
                    with gr.Column(scale=4):
                        workflow_input = gr.Textbox(
                            placeholder="Please enter your immunology research question, e.g.: Please analyze the key molecular mechanisms in antibody affinity maturation",
                            label="",
                            show_label=False,
                            lines=4,
                        )
                    # Second column: Two buttons vertically arranged
                    with gr.Column(scale=1, min_width=80):
                        with gr.Row(scale=1):
                            start_btn = gr.Button(
                                "Send",
                                variant="primary",
                                elem_classes=["btn-primary"],
                                size="sm",
                            )
                        with gr.Row(scale=1):
                            download_btn_2 = gr.DownloadButton(
                                "📄 Retrieval",
                                visible=False,
                                elem_classes=["btn-outline"],
                                size="sm",
                            )
                        with gr.Row(scale=1):
                            download_btn_3 = gr.DownloadButton(
                                "📄 Research",
                                visible=False,
                                elem_classes=["btn-outline"],
                                size="sm",
                            )
                    # Third column: Two buttons vertically arranged
                    with gr.Column(scale=1, min_width=80):
                        with gr.Row(scale=1):
                            download_btn_4 = gr.DownloadButton(
                                "📄 Hypothesis",
                                visible=False,
                                elem_classes=["btn-outline"],
                                size="sm",
                            )
                        with gr.Row(scale=1):
                            download_btn_5 = gr.DownloadButton(
                                "📄 Planing",
                                visible=False,
                                elem_classes=["btn-outline"],
                                size="sm",
                            )
                        with gr.Row(scale=1):
                            download_btn_6 = gr.DownloadButton(
                                "📄 Evaluation",
                                visible=False,
                                elem_classes=["btn-outline"],
                                size="sm",
                            )

                    # Second Column: Buttons and download buttons
                    # with gr.Column(scale=1, min_width=80, elem_id="button-column", elem_classes=["scrollable-column"]):

                    #     start_btn = gr.Button("Send", variant="primary", elem_classes=["btn-primary"], size="sm")

                    #     # Dynamic download buttons - initially hidden
                    #     download_btn_2 = gr.DownloadButton("📄 Retrieval", visible=False, size="sm")
                    #     download_btn_3 = gr.DownloadButton("📄 Research", visible=False, size="sm")
                    #     download_btn_4 = gr.DownloadButton("📄 Hypothesis", visible=False, size="sm")
                    #     download_btn_5 = gr.DownloadButton("📄 Planing", visible=False, size="sm")
                    #     download_btn_6 = gr.DownloadButton("📄 Evaluation", visible=False, size="sm")

            # Right human-machine interaction panel
            with gr.Column(scale=1, elem_classes=["interaction-panel"]):
                gr.HTML(
                    '<div class="interaction-header">🤝 Human-Machine Interaction Control <div class="status-dot" style="background: #ffc107;"></div> Awaiting Interaction</div>'
                )

                interaction_chatbot = gr.Chatbot(
                    label="",
                    height=500,
                    type="messages",  # 修复deprecated警告：使用messages格式替代tuples
                    show_label=False,
                    elem_classes=["interaction-chat"],
                    render_markdown=True,  # 启用Markdown/HTML渲染
                    sanitize_html=False,  # 允许自定义HTML标签和CSS类
                )

                # Parameter input and control area - left-right layout
                with gr.Row(elem_classes=["input-section"], height=170):
                    # First column: Input box
                    with gr.Column(scale=4):
                        param_input = gr.Textbox(
                            label="",
                            placeholder=f"Parameter modification (JSON format): {{'confidence_threshold': 0.85, 'max_insights': 25}}",
                            show_label=False,
                            lines=4,
                        )

                    # Second column: Two buttons vertically arranged
                    with gr.Column(scale=1, min_width=80):
                        with gr.Row():
                            btn_accept = gr.Button(
                                "✅ Default",
                                variant="primary",
                                elem_classes=["btn-primary"],
                                size="sm",
                            )
                        with gr.Row():
                            btn_skip_tool = gr.Button(
                                "⏭️ Skip Tool",
                                variant="outline",
                                elem_classes=["btn-outline"],
                                size="sm",
                            )

                    # Third column: Two buttons vertically arranged
                    with gr.Column(scale=1, min_width=80):
                        with gr.Row():
                            btn_modify = gr.Button(
                                "🔧 Modify",
                                variant="secondary",
                                elem_classes=["btn-secondary"],
                                size="sm",
                            )
                        with gr.Row():
                            btn_skip_task = gr.Button(
                                "🚫 Skip Task",
                                variant="danger",
                                elem_classes=["btn-danger"],
                                size="sm",
                            )

        # 数据格式转换函数：将tuple格式转换为messages格式
        def convert_to_messages_format(history_list):
            """将旧的tuple格式转换为新的messages格式"""
            messages = []
            for item in history_list:
                if isinstance(item, list) and len(item) == 2:
                    user_msg, assistant_msg = item
                    if user_msg:
                        messages.append({"role": "user", "content": user_msg})
                    if assistant_msg:
                        messages.append({"role": "assistant", "content": assistant_msg})
                elif isinstance(item, dict):
                    # 如果已经是字典格式，直接添加
                    messages.append(item)
            return messages

        # Event handling functions
        def start_workflow(question: str, state: Dict):
            """Start the real workflow"""
            if not question.strip():
                return (
                    state,
                    [
                        {
                            "role": "assistant",
                            "content": "❌ Please enter a research question",
                        }
                    ],
                    [],
                )

            success = workflow_manager.start_workflow(question)
            if success:
                new_state = {"question": question, "started": True}
                status_info = workflow_manager.get_status()
                return (
                    new_state,
                    convert_to_messages_format(status_info["workflow_history"]),
                    convert_to_messages_format(status_info["interaction_history"]),
                )
            else:
                return (
                    state,
                    [{"role": "assistant", "content": "❌ Failed to start workflow"}],
                    [],
                )

        def handle_interaction_action(action: str, params: str, state: Dict):
            """Handle interaction actions - updated to handle real TaskExecutor interactions"""
            if not state.get("started"):
                return []

            # 获取待处理的交互请求
            pending_interaction = workflow_manager.get_pending_interaction()
            if not pending_interaction:
                return []

            # 构造响应
            response = {"action": action}

            # 如果是修改参数，解析参数
            if action == "modify" and params.strip():
                try:
                    modified_args = json.loads(params)
                    response["modified_args"] = modified_args
                except json.JSONDecodeError:
                    return []

            # Set interaction response
            workflow_manager.set_interaction_response(response)

            # Return updated interaction history
            status_info = workflow_manager.get_status()
            return convert_to_messages_format(status_info["interaction_history"])

        def update_status():
            """Update status (polling function)"""
            try:
                status_info = workflow_manager.get_status()

                # Check if there are newly generated files that need to update download buttons
                stage_files = workflow_manager.stage_files
                download_updates = [None] * 5  # 5 download buttons

                # Check files for each stage, show download buttons immediately after each stage completes
                if (
                    2 in stage_files
                    and stage_files[2]
                    and os.path.exists(stage_files[2])
                ):
                    download_updates[0] = gr.DownloadButton(
                        "📄 Retrieval",
                        value=stage_files[2],
                        elem_classes=["btn-outline"],
                        visible=True,
                        size="sm",
                    )

                if (
                    3 in stage_files
                    and stage_files[3]
                    and os.path.exists(stage_files[3])
                ):
                    download_updates[1] = gr.DownloadButton(
                        "📄 Research",
                        value=stage_files[3],
                        elem_classes=["btn-outline"],
                        visible=True,
                        size="sm",
                    )

                if (
                    4 in stage_files
                    and stage_files[4]
                    and os.path.exists(stage_files[4])
                ):
                    download_updates[2] = gr.DownloadButton(
                        "📄 Hypothesis",
                        value=stage_files[4],
                        elem_classes=["btn-outline"],
                        visible=True,
                        size="sm",
                    )

                if (
                    5 in stage_files
                    and stage_files[5]
                    and os.path.exists(stage_files[5])
                ):
                    download_updates[3] = gr.DownloadButton(
                        "📄 Planing",
                        value=stage_files[5],
                        elem_classes=["btn-outline"],
                        visible=True,
                        size="sm",
                    )

                if (
                    6 in stage_files
                    and stage_files[6]
                    and os.path.exists(stage_files[6])
                ):
                    download_updates[4] = gr.DownloadButton(
                        "📄 Evaluation",
                        value=stage_files[6],
                        elem_classes=["btn-outline"],
                        visible=True,
                        size="sm",
                    )

                return (
                    convert_to_messages_format(status_info["workflow_history"]),
                    convert_to_messages_format(status_info["interaction_history"]),
                    *download_updates,
                )

            except Exception as e:
                print(f"Error updating status: {e}")
                return [], [], *([None] * 5)

        # Bind events
        start_btn.click(
            start_workflow,
            inputs=[workflow_input, workflow_state],
            outputs=[workflow_state, workflow_chatbot, interaction_chatbot],
        )

        # Interaction button events - updated to correct action values
        btn_accept.click(
            lambda params, state: handle_interaction_action("accept", "", state),
            inputs=[param_input, workflow_state],
            outputs=[interaction_chatbot],
        )

        btn_modify.click(
            lambda params, state: handle_interaction_action("modify", params, state),
            inputs=[param_input, workflow_state],
            outputs=[interaction_chatbot],
        )

        btn_skip_tool.click(
            lambda params, state: handle_interaction_action("reject", "", state),
            inputs=[param_input, workflow_state],
            outputs=[interaction_chatbot],
        )

        btn_skip_task.click(
            lambda params, state: handle_interaction_action("skip", "", state),
            inputs=[param_input, workflow_state],
            outputs=[interaction_chatbot],
        )

        # Timer to update status (every 2 seconds)
        timer = gr.Timer(value=2)
        timer.tick(
            update_status,
            inputs=[],
            outputs=[
                workflow_chatbot,
                interaction_chatbot,
                download_btn_2,
                download_btn_3,
                download_btn_4,
                download_btn_5,
                download_btn_6,
            ],
        )

    return demo


if __name__ == "__main__":
    # Create and launch the real workflow interface
    demo = create_real_workflow_interface()

    # Get absolute path of current working directory
    current_dir = os.path.abspath(".")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7876,  # Change port to avoid conflicts
        share=False,
        debug=True,
        # Allow access to files in current directory and all subdirectories
        allowed_paths=[current_dir],
    )
