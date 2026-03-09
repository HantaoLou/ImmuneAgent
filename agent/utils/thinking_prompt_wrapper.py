"""
GLM-4.5 思考提示词包装器

自动在所有GLM-4.5的调用中注入思考引导指令，确保模型输出思考过程。
由于GLM-4.5没有原生的show_thinking参数，需要通过提示词引导。
"""

from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage


THINKING_INSTRUCTION = """
【重要】在回答问题前，请务必先展示你的思考过程。

请按以下格式组织你的回答：

## 🤔 思考过程
[展示你的推理步骤，包括：]
1. **问题理解**：我理解你在问什么...
2. **关键信息**：我注意到这些关键点...
3. **推理步骤**：我按以下步骤分析...
   - 第一步：...
   - 第二步：...
   - ...
4. **关键判断**：基于以上分析，我认为...

## 💡 最终答案
[在这里给出清晰、准确的答案]

**注意**：
- 思考过程要详细，展示你的推理逻辑
- 如果需要计算，请在思考过程中展示计算步骤
- 如果需要推理，请说明推理依据
- 最终答案要简洁、准确
"""

THINKING_INSTRUCTION_COMPACT = """
回答前请先简要展示思考过程（标注为"💭 思考"），然后给出答案（标注为"💡 答案"）。
"""


def wrap_messages_with_thinking(
    messages: List[BaseMessage],
    enable_thinking: bool = True,
    compact_mode: bool = False,
) -> List[BaseMessage]:
    """
    在消息列表中注入思考引导指令

    Args:
        messages: 原始消息列表
        enable_thinking: 是否启用思考引导（默认True）
        compact_mode: 是否使用紧凑模式的指令（默认False）

    Returns:
        包装后的消息列表
    """
    if not enable_thinking or not messages:
        return messages

    instruction = THINKING_INSTRUCTION_COMPACT if compact_mode else THINKING_INSTRUCTION

    wrapped_messages = []
    has_system = False

    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            has_system = True
            wrapped_content = f"{msg.content}\n\n{instruction}"
            wrapped_messages.append(SystemMessage(content=wrapped_content))
        else:
            wrapped_messages.append(msg)

    if not has_system and len(messages) > 0:
        wrapped_messages.insert(0, SystemMessage(content=instruction))

    return wrapped_messages


def extract_thinking_and_answer(content: str) -> Dict[str, str]:
    """
    从LLM响应中提取思考过程和最终答案

    Args:
        content: LLM响应内容

    Returns:
        包含thinking和answer的字典
    """
    thinking = ""
    answer = ""

    import re

    thinking_match = re.search(
        r"##\s*🤔\s*思考过程\s*(.*?)(?=##\s*💡\s*最终答案|$)",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if thinking_match:
        thinking = thinking_match.group(1).strip()

    answer_match = re.search(
        r"##\s*💡\s*最终答案\s*(.*?)$", content, re.DOTALL | re.IGNORECASE
    )
    if answer_match:
        answer = answer_match.group(1).strip()

    compact_thinking_match = re.search(
        r"💭\s*思考[：:]\s*(.*?)(?=💡\s*答案|$)",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if compact_thinking_match and not thinking:
        thinking = compact_thinking_match.group(1).strip()

    compact_answer_match = re.search(
        r"💡\s*答案[：:]\s*(.*?)$", content, re.DOTALL | re.IGNORECASE
    )
    if compact_answer_match and not answer:
        answer = compact_answer_match.group(1).strip()

    if not thinking and not answer:
        thinking_lines = []
        answer_lines = []
        in_thinking = True

        for line in content.split("\n"):
            line_lower = line.lower()
            if any(
                marker in line_lower
                for marker in ["最终答案", "final answer", "答案：", "answer:"]
            ):
                in_thinking = False
                continue

            if in_thinking:
                thinking_lines.append(line)
            else:
                answer_lines.append(line)

        thinking = "\n".join(thinking_lines).strip()
        answer = "\n".join(answer_lines).strip()

    if not thinking and not answer:
        answer = content

    return {"thinking": thinking, "answer": answer, "full_content": content}


class ThinkingPromptWrapper:
    """
    思考提示词包装器类

    用于包装LLM调用，自动注入思考引导指令
    """

    def __init__(
        self,
        enable_thinking: bool = True,
        compact_mode: bool = False,
        progress_callback: Optional[callable] = None,
    ):
        self.enable_thinking = enable_thinking
        self.compact_mode = compact_mode
        self.progress_callback = progress_callback

    def wrap_invoke(self, llm_instance: Any, messages: List[BaseMessage], **kwargs):
        """
        包装LLM的invoke方法，自动注入思考引导

        Args:
            llm_instance: LLM实例
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            LLM响应
        """
        wrapped_messages = wrap_messages_with_thinking(
            messages, self.enable_thinking, self.compact_mode
        )

        if self.progress_callback:
            try:
                user_msg = ""
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        user_msg = str(msg.content)[:100]
                        break

                self.progress_callback(
                    event_type="llm_thinking",
                    message=f"🤔 准备调用LLM: {user_msg}",
                    details={
                        "phase": "pre_invoke",
                        "enable_thinking": self.enable_thinking,
                        "message_count": len(messages),
                    },
                )
            except Exception as e:
                print(f"[ThinkingPromptWrapper] Error reporting pre-invoke: {e}")

        response = llm_instance.invoke(wrapped_messages, **kwargs)

        if self.progress_callback and hasattr(response, "content"):
            try:
                extracted = extract_thinking_and_answer(response.content)

                if extracted["thinking"]:
                    self.progress_callback(
                        event_type="llm_thinking",
                        message=f"💭 思考过程:\n{extracted['thinking'][:300]}",
                        details={
                            "phase": "thinking_extracted",
                            "thinking_length": len(extracted["thinking"]),
                            "has_answer": bool(extracted["answer"]),
                        },
                    )

                if extracted["answer"]:
                    self.progress_callback(
                        event_type="llm_thinking",
                        message=f"💡 最终答案:\n{extracted['answer'][:200]}",
                        details={
                            "phase": "answer_extracted",
                            "answer_length": len(extracted["answer"]),
                        },
                    )
            except Exception as e:
                print(f"[ThinkingPromptWrapper] Error extracting thinking: {e}")

        return response


def create_thinking_wrapper(
    enable_thinking: bool = True,
    compact_mode: bool = False,
    progress_callback: Optional[callable] = None,
) -> ThinkingPromptWrapper:
    """
    创建思考提示词包装器实例

    Args:
        enable_thinking: 是否启用思考引导
        compact_mode: 是否使用紧凑模式
        progress_callback: 进度回调函数

    Returns:
        ThinkingPromptWrapper实例
    """
    return ThinkingPromptWrapper(
        enable_thinking=enable_thinking,
        compact_mode=compact_mode,
        progress_callback=progress_callback,
    )
