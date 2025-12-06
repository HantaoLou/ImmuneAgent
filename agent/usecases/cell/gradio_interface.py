import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from gradio_workflow_manager import GradioWorkflowManager


class GradioInterface:
    """Gradio界面管理器"""

    def __init__(self):
        self.workflow_manager = GradioWorkflowManager()
        self.current_interrupt_info = None
        self.chat_history = []

    def create_interface(self):
        """创建Gradio界面"""
        # 自定义CSS样式 - 防止双滚动条
        custom_css = """
        /* 聊天显示区域样式 */
        .chatbot {
            overflow: hidden !important;
        }
        .chatbot > div {
            overflow-y: auto !important;
            max-height: 400px !important;
        }
        /* 确保容器不产生额外滚动条 */
        .gradio-container {
            overflow: visible !important;
        }
        """

        with gr.Blocks(
            title="抗体分析工作流", theme=gr.themes.Soft(), css=custom_css
        ) as interface:
            gr.Markdown("# 🧬 抗体分析工作流系统")
            gr.Markdown("基于LangGraph的智能抗体分析平台")

            with gr.Row():
                # 左侧：工作流控制面板
                with gr.Column(scale=1):
                    gr.Markdown("## 📋 工作流控制")

                    # 工作流状态显示
                    status_display = gr.JSON(
                        label="工作流状态",
                        height=400,
                        value={"status": "未启动", "node": "无"},
                    )

                    # 用户问题输入
                    question_input = gr.Textbox(
                        label="研究问题",
                        placeholder="请输入您的抗体分析问题...",
                        lines=3,
                    )

                    # 启动工作流按钮
                    start_btn = gr.Button("🚀 启动工作流", variant="primary")

                    # 重置工作流按钮
                    reset_btn = gr.Button("🔄 重置工作流", variant="secondary")

                # 右侧：交互面板
                with gr.Column(scale=2):
                    gr.Markdown("## 💬 交互面板")

                    # 聊天历史显示
                    chat_display = gr.Chatbot(
                        label="工作流交互",
                        height=400,
                        show_label=True,
                        elem_classes=["chatbot"],
                    )

                    # 当前节点信息
                    current_node_info = gr.Markdown("**当前节点:** 等待启动")

                    # 用户输入提示信息
                    input_prompt_info = gr.Markdown(
                        "**输入提示:** 请先启动工作流", visible=True
                    )

                    # 用户输入区域
                    with gr.Row():
                        user_input = gr.Textbox(
                            label="用户输入",
                            placeholder="根据当前节点要求输入相应内容...",
                            scale=4,
                        )
                        submit_btn = gr.Button("📤 提交", scale=1)

            # 底部：分析结果显示（Tab页签形式）
            with gr.Row():
                gr.Markdown("## 📊 分析结果")

            with gr.Row():
                with gr.Tabs() as result_tabs:
                    with gr.Tab("计划1") as tab1:
                        plan1_display = gr.Markdown(value="暂无数据", label="计划1")

                    with gr.Tab("计划2") as tab2:
                        plan2_display = gr.Markdown(value="暂无数据", label="计划2")

                    with gr.Tab("计划3") as tab3:
                        plan3_display = gr.Markdown(value="暂无数据", label="计划3")

                    with gr.Tab("计划4") as tab4:
                        plan4_display = gr.Markdown(value="暂无数据", label="计划4")

                    with gr.Tab("整合计划") as tab5:
                        integrated_plan_display = gr.Markdown(
                            value="暂无数据", label="整合计划"
                        )

            # 事件绑定
            start_btn.click(
                fn=self.start_workflow,
                inputs=[question_input],
                outputs=[
                    status_display,
                    chat_display,
                    current_node_info,
                    input_prompt_info,
                    plan1_display,
                    plan2_display,
                    plan3_display,
                    plan4_display,
                    integrated_plan_display,
                ],
            )

            reset_btn.click(
                fn=self.reset_workflow,
                outputs=[
                    status_display,
                    chat_display,
                    current_node_info,
                    input_prompt_info,
                    plan1_display,
                    plan2_display,
                    plan3_display,
                    plan4_display,
                    integrated_plan_display,
                ],
            )

            submit_btn.click(
                fn=self.submit_user_input,
                inputs=[user_input],
                outputs=[
                    chat_display,
                    current_node_info,
                    input_prompt_info,
                    user_input,
                    status_display,
                    plan1_display,
                    plan2_display,
                    plan3_display,
                    plan4_display,
                    integrated_plan_display,
                ],
            )

        return interface

    def _format_plan_data(self) -> Tuple[str, str, str, str, str]:
        """格式化计划数据用于显示"""
        plan_data = self.workflow_manager.get_plan_data()
        individual_plans = plan_data.get("individual_plans", [])
        integrated_plan = plan_data.get("generated_plan", "")

        # 确保有4个独立计划位置
        plan1 = individual_plans[0] if len(individual_plans) > 0 else "暂无数据"
        plan2 = individual_plans[1] if len(individual_plans) > 1 else "暂无数据"
        plan3 = individual_plans[2] if len(individual_plans) > 2 else "暂无数据"
        plan4 = individual_plans[3] if len(individual_plans) > 3 else "暂无数据"
        integrated = integrated_plan if integrated_plan else "暂无数据"

        return plan1, plan2, plan3, plan4, integrated

    def start_workflow(
        self, question: str
    ) -> Tuple[Dict, List, str, str, str, str, str, str, str]:
        """启动工作流"""
        if not question.strip():
            plan1, plan2, plan3, plan4, integrated = self._format_plan_data()
            return (
                {"status": "错误", "message": "请输入研究问题"},
                self.chat_history,
                "**当前节点:** 错误 - 请输入问题",
                "**输入提示:** 请输入有效的研究问题",
                plan1,
                plan2,
                plan3,
                plan4,
                integrated,
            )

        try:
            # 初始化工作流
            success, message = self.workflow_manager.initialize_workflow(question)
            if not success:
                plan1, plan2, plan3, plan4, integrated = self._format_plan_data()
                return (
                    {"status": "错误", "message": message},
                    self.chat_history,
                    "**当前节点:** 初始化失败",
                    "**输入提示:** 请检查输入并重试",
                    plan1,
                    plan2,
                    plan3,
                    plan4,
                    integrated,
                )

            # 启动工作流
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success, status_msg, interrupt_info = loop.run_until_complete(
                self.workflow_manager.start_workflow()
            )
            loop.close()

            if success:
                self.current_interrupt_info = interrupt_info
                self.chat_history.append(["系统", f"工作流已启动: {question}"])

                if interrupt_info:
                    # 从 interrupt_info 中提取任务和描述信息
                    task = interrupt_info.get(
                        "task", interrupt_info.get("node_name", "未知节点")
                    )
                    description = interrupt_info.get(
                        "description", interrupt_info.get("prompt", "等待用户输入")
                    )
                    self.chat_history.append(
                        ["系统", f"当前节点: {task}\n{description}"]
                    )

                    plan1, plan2, plan3, plan4, integrated = self._format_plan_data()
                    return (
                        self.workflow_manager.get_workflow_status(),
                        self.chat_history,
                        f"**当前节点:** {task}",
                        f"**输入提示:** {description}",
                        plan1,
                        plan2,
                        plan3,
                        plan4,
                        integrated,
                    )
                else:
                    plan1, plan2, plan3, plan4, integrated = self._format_plan_data()
                    return (
                        self.workflow_manager.get_workflow_status(),
                        self.chat_history,
                        "**当前节点:** 已完成",
                        "**输入提示:** 工作流已完成，无需输入",
                        plan1,
                        plan2,
                        plan3,
                        plan4,
                        integrated,
                    )
            else:
                plan1, plan2, plan3, plan4, integrated = self._format_plan_data()
                return (
                    {"status": "错误", "message": status_msg},
                    self.chat_history,
                    "**当前节点:** 启动失败",
                    "**输入提示:** 请检查问题并重新启动",
                    plan1,
                    plan2,
                    plan3,
                    plan4,
                    integrated,
                )

        except Exception as e:
            plan1, plan2, plan3, plan4, integrated = self._format_plan_data()
            return (
                {"status": "错误", "message": f"工作流启动失败: {str(e)}"},
                self.chat_history,
                "**当前节点:** 错误",
                "**输入提示:** 请重试或检查输入",
                plan1,
                plan2,
                plan3,
                plan4,
                integrated,
            )

    def submit_user_input(
        self, user_input: str
    ) -> Tuple[List, str, str, str, Dict, str, str, str, str, str]:
        """提交用户输入"""
        if not user_input.strip():
            plan1, plan2, plan3, plan4, integrated = self._format_plan_data()
            return (
                self.chat_history,
                "**当前节点:** 等待有效输入",
                "**输入提示:** 请输入有效内容",
                user_input,
                self.workflow_manager.get_workflow_status(),
                plan1,
                plan2,
                plan3,
                plan4,
                integrated,
            )

        try:
            # 添加用户输入到聊天历史
            self.chat_history.append(["用户", user_input])

            # 继续工作流
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success, status_msg, interrupt_info = loop.run_until_complete(
                self.workflow_manager.continue_workflow(user_input)
            )
            loop.close()

            if success:
                self.chat_history.append(["系统", status_msg])

                if interrupt_info:
                    # 还有下一个中断点
                    self.current_interrupt_info = interrupt_info
                    # 从 interrupt_info 中提取任务和描述信息
                    task = interrupt_info.get(
                        "task", interrupt_info.get("node_name", "未知节点")
                    )
                    description = interrupt_info.get(
                        "description", interrupt_info.get("prompt", "等待用户输入")
                    )
                    self.chat_history.append(
                        ["系统", f"当前节点: {task}\n{description}"]
                    )

                    plan1, plan2, plan3, plan4, integrated = self._format_plan_data()
                    return (
                        self.chat_history,
                        f"**当前节点:** {task}",
                        f"**输入提示:** {description}",
                        "",  # 清空输入框
                        self.workflow_manager.get_workflow_status(),
                        plan1,
                        plan2,
                        plan3,
                        plan4,
                        integrated,
                    )
                else:
                    # 工作流完成
                    self.current_interrupt_info = None
                    plan1, plan2, plan3, plan4, integrated = self._format_plan_data()
                    return (
                        self.chat_history,
                        "**当前节点:** 已完成",
                        "**输入提示:** 工作流已完成，无需输入",
                        "",
                        self.workflow_manager.get_workflow_status(),
                        plan1,
                        plan2,
                        plan3,
                        plan4,
                        integrated,
                    )
            else:
                self.chat_history.append(["系统", f"错误: {status_msg}"])
                plan1, plan2, plan3, plan4, integrated = self._format_plan_data()
                return (
                    self.chat_history,
                    "**当前节点:** 错误",
                    "**输入提示:** 请检查输入并重试",
                    user_input,
                    self.workflow_manager.get_workflow_status(),
                    plan1,
                    plan2,
                    plan3,
                    plan4,
                    integrated,
                )

        except Exception as e:
            error_msg = f"异常: {str(e)}"
            self.chat_history.append(["系统", error_msg])
            plan1, plan2, plan3, plan4, integrated = self._format_plan_data()
            return (
                self.chat_history,
                "**当前节点:** 异常",
                "**输入提示:** 发生异常，请重试",
                user_input,
                self.workflow_manager.get_workflow_status(),
                plan1,
                plan2,
                plan3,
                plan4,
                integrated,
            )

    def reset_workflow(
        self,
    ) -> Tuple[Dict, List, str, str, str, str, str, str, str, str]:
        """重置工作流"""
        self.workflow_manager.reset_workflow()
        self.current_interrupt_info = None
        self.chat_history = []

        return (
            {"status": "就绪", "message": "系统已重置"},
            [],
            "**当前节点:** 等待启动",
            "**输入提示:** 请先启动工作流",
            "",  # 清空问题输入
            "",  # 清空用户输入
            "暂无数据",  # plan1
            "暂无数据",  # plan2
            "暂无数据",  # plan3
            "暂无数据",  # plan4
            "暂无数据",  # integrated plan
        )


def create_gradio_app():
    """创建Gradio应用"""
    interface = GradioInterface()
    return interface.create_interface()


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
