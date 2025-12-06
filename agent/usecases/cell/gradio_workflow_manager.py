import asyncio
import os
import uuid
from typing import Any, Dict, Optional, Tuple

from langgraph.types import Command

from common.constants import STANDARDIZED_WORKING_DIRECTORY
from usecases.cell.cell_config import get_cell_runnable_config
from usecases.cell.graph.planning_graph import create_planning_graph
from usecases.cell.state.state import State


class GradioWorkflowManager:
    """Gradio工作流管理器 - 管理LangGraph工作流的执行和状态"""

    def __init__(self):
        self.app = None
        self.config = None
        self.current_state = None
        self.workflow_id = None
        self.work_directory = None
        self.is_running = False

    def initialize_workflow(self, user_question: str) -> Tuple[bool, str]:
        """初始化工作流

        Args:
            user_question: 用户输入的问题

        Returns:
            Tuple[bool, str]: (是否成功, 状态信息)
        """
        try:
            # 创建工作流图
            self.app = create_planning_graph()

            # 生成唯一ID和工作目录
            self.workflow_id = uuid.uuid4()
            self.work_directory = os.path.join(
                STANDARDIZED_WORKING_DIRECTORY, str(self.workflow_id)
            )
            os.makedirs(self.work_directory, exist_ok=True)

            # 获取配置
            self.config = get_cell_runnable_config(
                self.workflow_id, self.work_directory
            )

            # 初始化状态
            initial_state = State(
                refine_plan="",
                plan_confirmed=False,
                model_upload_files={"flu": "", "rsv": "", "sars": "", "a1a11": ""},
                strategy_upload_files="",
                strategy_input_valid=True,
                selected_model=[],
                selected_strategy=[],
                original_question=user_question,
                optimized_questions=[],
                generated_plan="",
                context="",
                individual_plans=[],
            )

            self.current_state = initial_state
            self.is_running = True

            return True, "工作流初始化成功"

        except Exception as e:
            return False, f"工作流初始化失败: {str(e)}"

    async def start_workflow(self) -> Tuple[bool, str, Dict[str, Any]]:
        """启动工作流到第一个中断点

        Returns:
            Tuple[bool, str, Dict]: (是否成功, 状态信息, 中断信息)
        """
        try:
            if not self.app or not self.current_state:
                return False, "工作流未初始化", {}

            # 运行到第一个中断点
            _ = list(self.app.stream(self.current_state, self.config, subgraphs=True))

            # 获取当前状态和中断信息
            state = self.app.get_state(self.config)

            if state.interrupts:
                interrupt_info = state.interrupts[0].value
                return True, "工作流已启动，等待用户输入", interrupt_info
            else:
                return True, "工作流已完成", {}

        except Exception as e:
            return False, f"启动工作流失败: {str(e)}", {}

    async def continue_workflow(
        self, user_input: str
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """继续工作流执行

        Args:
            user_input: 用户输入

        Returns:
            Tuple[bool, str, Optional[Dict]]: (是否成功, 状态信息, 中断信息或None)
        """
        try:
            if not self.app:
                return False, "工作流未初始化", None

            # 使用用户输入恢复工作流
            _ = list(
                self.app.stream(Command(resume=user_input), self.config, subgraphs=True)
            )

            # 获取当前状态
            state = self.app.get_state(self.config)

            # 检查是否还有下一个节点
            if not state.next:
                self.is_running = False
                final_state = state.values
                return True, "工作流执行完成", None

            # 如果有中断，返回中断信息
            if state.interrupts:
                interrupt_info = state.interrupts[0].value
                return True, "等待用户输入", interrupt_info
            else:
                return True, "工作流继续执行中", None

        except Exception as e:
            return False, f"继续工作流失败: {str(e)}", None

    def get_current_node_info(self) -> Dict[str, Any]:
        """获取当前节点信息

        Returns:
            Dict: 当前节点信息
        """
        if not self.app:
            return {"node": "未初始化", "status": "idle"}

        try:
            state = self.app.get_state(self.config)
            if state.next:
                return {
                    "node": state.next[0] if state.next else "未知",
                    "status": "waiting" if state.interrupts else "running",
                    "has_interrupt": bool(state.interrupts),
                }
            else:
                return {"node": "完成", "status": "completed"}
        except:
            return {"node": "错误", "status": "error"}

    def reset_workflow(self):
        """重置工作流"""
        self.app = None
        self.config = None
        self.current_state = None
        self.workflow_id = None
        self.work_directory = None
        self.is_running = False

    def get_workflow_status(self) -> Dict[str, Any]:
        """获取工作流状态

        Returns:
            Dict: 工作流状态信息
        """
        return {
            "is_running": self.is_running,
            "workflow_id": str(self.workflow_id) if self.workflow_id else None,
            "work_directory": self.work_directory,
            "node_info": self.get_current_node_info(),
        }

    def get_plan_data(self) -> Dict[str, Any]:
        """获取计划数据用于显示

        Returns:
            Dict: 包含individual_plans和generated_plan的数据
        """
        if not self.app:
            return {"individual_plans": [], "generated_plan": ""}

        try:
            state = self.app.get_state(self.config)
            state_values = state.values

            return {
                "individual_plans": state_values.get("individual_plans", []),
                "generated_plan": state_values.get("generated_plan", ""),
                "original_question": state_values.get("original_question", ""),
                "optimized_questions": state_values.get("optimized_questions", []),
            }
        except Exception as e:
            print(f"获取计划数据失败: {str(e)}")
            return {
                "individual_plans": [],
                "generated_plan": "",
                "original_question": "",
                "optimized_questions": [],
            }
