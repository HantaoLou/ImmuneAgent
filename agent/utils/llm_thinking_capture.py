"""
LLM 思考过程捕获模块

提供回调机制，在LLM调用前后捕获思考过程，并通过progress_callback推送到前端
"""

from typing import Optional, Dict, Any, Callable, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
import time


class LLMThinkingCapture:
    """LLM思考过程捕获器"""

    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback
        self.call_chain: List[Dict[str, Any]] = []

    def report_thinking(
        self,
        thinking_content: str,
        step_name: str = "推理",
        node_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """报告LLM思考过程"""
        if not self.progress_callback:
            return

        try:
            # 截断长内容
            truncated = (
                thinking_content[:300]
                if len(thinking_content) > 300
                else thinking_content
            )

            event_data = {
                "event_type": "llm_thinking",
                "message": f"💭 {step_name}: {truncated}",
                "node_name": node_name,
                "details": details
                or {"step": step_name, "full_content": thinking_content[:1000]},
            }

            self.progress_callback(**event_data)

            # 记录到调用链
            self.call_chain.append(
                {
                    "type": "thinking",
                    "step": step_name,
                    "content": truncated,
                    "timestamp": time.time(),
                }
            )
        except Exception as e:
            print(f"[LLMThinkingCapture] Error reporting thinking: {e}")

    def report_reasoning_step(
        self,
        step_description: str,
        step_number: int = 0,
        total_steps: int = 0,
        node_name: Optional[str] = None,
    ):
        """报告推理步骤"""
        if not self.progress_callback:
            return

        try:
            step_info = (
                f"[步骤 {step_number}/{total_steps}] " if total_steps > 0 else ""
            )

            event_data = {
                "event_type": "llm_reasoning",
                "message": f"🧠 {step_info}{step_description}",
                "node_name": node_name,
                "details": {"step_number": step_number, "total_steps": total_steps},
                "progress_percent": int(
                    (step_number / total_steps * 100) if total_steps > 0 else 0
                ),
            }

            self.progress_callback(**event_data)

            self.call_chain.append(
                {
                    "type": "reasoning",
                    "step_number": step_number,
                    "description": step_description,
                    "timestamp": time.time(),
                }
            )
        except Exception as e:
            print(f"[LLMThinkingCapture] Error reporting reasoning: {e}")

    def report_streaming_chunk(
        self, chunk: str, accumulated_length: int = 0, node_name: Optional[str] = None
    ):
        """报告流式输出块"""
        if not self.progress_callback:
            return

        try:
            event_data = {
                "event_type": "llm_streaming",
                "message": chunk,
                "node_name": node_name,
                "details": {"accumulated_length": accumulated_length},
            }

            self.progress_callback(**event_data)
        except Exception as e:
            print(f"[LLMThinkingCapture] Error reporting streaming: {e}")

    def report_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        reasoning: str = "",
        node_name: Optional[str] = None,
    ):
        """报告工具调用决策"""
        if not self.progress_callback:
            return

        try:
            args_str = str(tool_args)[:100]
            reasoning_str = f" - {reasoning[:150]}" if reasoning else ""

            event_data = {
                "event_type": "llm_thinking",
                "message": f"🔧 决定调用工具: {tool_name}{reasoning_str}",
                "node_name": node_name,
                "details": {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "reasoning": reasoning,
                },
            }

            self.progress_callback(**event_data)

            self.call_chain.append(
                {"type": "tool_call", "tool_name": tool_name, "timestamp": time.time()}
            )
        except Exception as e:
            print(f"[LLMThinkingCapture] Error reporting tool call: {e}")

    def before_llm_call(
        self,
        messages: List[BaseMessage],
        context: str = "",
        node_name: Optional[str] = None,
    ):
        """LLM调用前的报告"""
        if not self.progress_callback:
            return

        try:
            # 分析消息，提取关键信息
            user_msg = ""
            system_prompt_summary = ""

            for msg in messages:
                if isinstance(msg, HumanMessage):
                    user_msg = str(msg.content)[:200]
                elif isinstance(msg, SystemMessage):
                    system_prompt_summary = str(msg.content)[:100] + "..."

            thinking = f"准备调用LLM处理: {user_msg[:100]}"
            if context:
                thinking = f"{context} - {thinking}"

            self.report_thinking(
                thinking_content=thinking,
                step_name="LLM调用准备",
                node_name=node_name,
                details={
                    "message_count": len(messages),
                    "user_message": user_msg,
                    "system_prompt_summary": system_prompt_summary,
                    "context": context,
                },
            )
        except Exception as e:
            print(f"[LLMThinkingCapture] Error in before_llm_call: {e}")

    def after_llm_call(
        self, response: BaseMessage, context: str = "", node_name: Optional[str] = None
    ):
        """LLM调用后的报告"""
        if not self.progress_callback:
            return

        try:
            # 提取响应内容
            response_content = ""
            tool_calls = []

            if hasattr(response, "content"):
                response_content = str(response.content)[:300]

            if hasattr(response, "tool_calls") and response.tool_calls:
                for tc in response.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    tool_calls.append(tool_name)
                    # 报告工具调用
                    self.report_tool_call(
                        tool_name=tool_name,
                        tool_args=tc.get("args", {}),
                        reasoning=f"LLM决定使用{tool_name}工具",
                        node_name=node_name,
                    )

            # 报告响应内容
            if response_content and not tool_calls:
                thinking = f"LLM返回: {response_content[:200]}"
                if context:
                    thinking = f"{context} - {thinking}"

                self.report_thinking(
                    thinking_content=thinking,
                    step_name="LLM响应分析",
                    node_name=node_name,
                    details={
                        "response_length": len(response_content),
                        "has_tool_calls": len(tool_calls) > 0,
                        "context": context,
                    },
                )
        except Exception as e:
            print(f"[LLMThinkingCapture] Error in after_llm_call: {e}")

    def get_call_chain_summary(self) -> List[Dict[str, Any]]:
        """获取调用链摘要"""
        return self.call_chain.copy()

    def clear_call_chain(self):
        """清空调用链"""
        self.call_chain.clear()


def create_thinking_capture_from_state(state: Any) -> LLMThinkingCapture:
    """从GlobalState创建思考捕获器"""
    progress_callback = None

    if hasattr(state, "progress_callback"):
        progress_callback = state.progress_callback
    elif isinstance(state, dict):
        progress_callback = state.get("progress_callback")

    return LLMThinkingCapture(progress_callback=progress_callback)


def with_thinking_capture(
    llm_func: Callable, state: Any, context: str = "", node_name: Optional[str] = None
) -> Callable:
    """
    装饰器：为LLM调用添加思考捕获

    Args:
        llm_func: LLM调用函数 (例如 llm.invoke)
        state: GlobalState对象或字典
        context: 调用上下文描述
        node_name: 节点名称

    Returns:
        包装后的函数
    """
    capture = create_thinking_capture_from_state(state)

    def wrapped_invoke(messages, **kwargs):
        # 调用前
        capture.before_llm_call(messages, context, node_name)

        # 调用LLM
        response = llm_func(messages, **kwargs)

        # 调用后
        capture.after_llm_call(response, context, node_name)

        return response

    return wrapped_invoke
