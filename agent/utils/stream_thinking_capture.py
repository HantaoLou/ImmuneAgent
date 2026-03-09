"""
GLM-4.5 流式思考捕获增强模块

提供增强的流式调用方法，确保所有GLM-4.5调用都能捕获并推送思考过程
"""

from typing import List, Optional, Callable, Any, Dict
from langchain_core.messages import BaseMessage, HumanMessage
import time
import re


def stream_with_thinking_capture(
    zhipu_client: Any,
    messages: List[Dict[str, str]],
    model: str = "glm-4.5",
    temperature: float = 0.1,
    progress_callback: Optional[Callable] = None,
    enable_thinking_prompt: bool = True,
    **kwargs,
) -> str:
    """
    带思考捕获的流式调用方法

    Args:
        zhipu_client: ZhipuAI客户端实例
        messages: 消息列表（已转换为zhipu格式）
        model: 模型名称
        temperature: 温度参数
        progress_callback: 进度回调函数
        enable_thinking_prompt: 是否启用思考引导提示词
        **kwargs: 其他参数

    Returns:
        完整的响应内容
    """
    start_time = time.time()
    accumulated_content = ""
    chunk_count = 0

    # 注入思考引导提示词
    if enable_thinking_prompt and messages:
        thinking_instruction = """
【重要】在回答问题时，请先简要展示你的思考过程，然后给出答案。
格式：💭 思考：[你的思考过程] ... 💡 答案：[你的答案]
"""
        if messages[0]["role"] == "system":
            messages[0]["content"] += "\n\n" + thinking_instruction
        else:
            messages.insert(0, {"role": "system", "content": thinking_instruction})

    # 报告开始
    if progress_callback:
        try:
            user_msg = ""
            for msg in messages:
                if msg["role"] == "user":
                    user_msg = msg["content"][:100]
                    break

            progress_callback(
                event_type="llm_thinking",
                message=f"🤔 开始思考: {user_msg}",
                details={
                    "phase": "streaming_start",
                    "model": model,
                    "temperature": temperature,
                },
            )
        except Exception as e:
            print(f"[StreamThinkingCapture] Error reporting start: {e}")

    try:
        # 流式调用
        response = zhipu_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
            **kwargs,
        )

        # 处理流式响应
        for chunk in response:
            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    chunk_content = delta.content
                    accumulated_content += chunk_content
                    chunk_count += 1

                    # 每3个chunk报告一次
                    if progress_callback and chunk_count % 3 == 0:
                        try:
                            recent_text = accumulated_content[-150:]
                            progress_callback(
                                event_type="llm_streaming",
                                message=f"💭 {recent_text}",
                                details={
                                    "phase": "streaming_progress",
                                    "chunk_count": chunk_count,
                                    "total_length": len(accumulated_content),
                                    "elapsed_seconds": round(
                                        time.time() - start_time, 2
                                    ),
                                },
                            )
                        except Exception as e:
                            print(f"[StreamThinkingCapture] Error reporting chunk: {e}")

        # 报告完成
        if progress_callback:
            try:
                thinking, answer = _extract_thinking_and_answer(accumulated_content)

                if thinking:
                    progress_callback(
                        event_type="llm_thinking",
                        message=f"🧠 思考过程:\n{thinking[:300]}",
                        details={
                            "phase": "thinking_complete",
                            "thinking_length": len(thinking),
                        },
                    )

                if answer:
                    progress_callback(
                        event_type="llm_thinking",
                        message=f"💡 最终答案:\n{answer[:200]}",
                        details={
                            "phase": "answer_complete",
                            "answer_length": len(answer),
                            "total_time": round(time.time() - start_time, 2),
                            "total_chunks": chunk_count,
                        },
                    )
                else:
                    progress_callback(
                        event_type="llm_thinking",
                        message=f"✅ 思考完成 ({len(accumulated_content)} 字符)",
                        details={
                            "phase": "streaming_complete",
                            "total_length": len(accumulated_content),
                            "total_chunks": chunk_count,
                            "total_time": round(time.time() - start_time, 2),
                        },
                    )
            except Exception as e:
                print(f"[StreamThinkingCapture] Error reporting completion: {e}")

        return accumulated_content

    except Exception as e:
        if progress_callback:
            try:
                progress_callback(
                    event_type="error",
                    message=f"❌ LLM调用失败: {str(e)[:200]}",
                    details={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )
            except:
                pass
        raise


def invoke_with_thinking_capture(
    zhipu_client: Any,
    messages: List[Dict[str, str]],
    model: str = "glm-4.5",
    temperature: float = 0.1,
    progress_callback: Optional[Callable] = None,
    enable_thinking_prompt: bool = True,
    simulate_streaming: bool = True,
    **kwargs,
) -> str:
    """
    带思考捕获的非流式调用方法（可模拟流式输出）

    Args:
        zhipu_client: ZhipuAI客户端实例
        messages: 消息列表（已转换为zhipu格式）
        model: 模型名称
        temperature: 温度参数
        progress_callback: 进度回调函数
        enable_thinking_prompt: 是否启用思考引导提示词
        simulate_streaming: 是否模拟流式输出
        **kwargs: 其他参数

    Returns:
        响应内容
    """
    start_time = time.time()

    # 注入思考引导提示词
    if enable_thinking_prompt and messages:
        thinking_instruction = """
【重要】在回答问题时，请先简要展示你的思考过程，然后给出答案。
格式：💭 思考：[你的思考过程] ... 💡 答案：[你的答案]
"""
        if messages[0]["role"] == "system":
            messages[0]["content"] += "\n\n" + thinking_instruction
        else:
            messages.insert(0, {"role": "system", "content": thinking_instruction})

    # 报告开始
    if progress_callback:
        try:
            user_msg = ""
            for msg in messages:
                if msg["role"] == "user":
                    user_msg = msg["content"][:100]
                    break

            progress_callback(
                event_type="llm_thinking",
                message=f"🤔 开始思考: {user_msg}",
                details={
                    "phase": "invoke_start",
                    "model": model,
                    "temperature": temperature,
                },
            )
        except Exception as e:
            print(f"[InvokeThinkingCapture] Error reporting start: {e}")

    try:
        # 非流式调用
        response = zhipu_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False,
            **kwargs,
        )

        # 提取内容
        content = ""
        if hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content or ""

        # 报告思考过程
        if progress_callback:
            thinking, answer = _extract_thinking_and_answer(content)

            # 模拟流式输出
            if simulate_streaming:
                if thinking:
                    thinking_chunks = _split_into_chunks(thinking, chunk_size=100)
                    for i, chunk in enumerate(thinking_chunks):
                        progress_callback(
                            event_type="llm_streaming",
                            message=f"💭 {chunk}",
                            details={
                                "phase": "thinking_progress",
                                "chunk_number": i + 1,
                                "total_chunks": len(thinking_chunks),
                                "elapsed_seconds": round(time.time() - start_time, 2),
                            },
                        )
                        time.sleep(0.05)

                if answer:
                    progress_callback(
                        event_type="llm_thinking",
                        message=f"💡 最终答案:\n{answer[:200]}",
                        details={
                            "phase": "answer_complete",
                            "answer_length": len(answer),
                            "total_time": round(time.time() - start_time, 2),
                        },
                    )
            else:
                if thinking:
                    progress_callback(
                        event_type="llm_thinking",
                        message=f"🧠 思考过程:\n{thinking[:300]}",
                        details={
                            "phase": "thinking_complete",
                            "thinking_length": len(thinking),
                        },
                    )

                if answer:
                    progress_callback(
                        event_type="llm_thinking",
                        message=f"💡 最终答案:\n{answer[:200]}",
                        details={
                            "phase": "answer_complete",
                            "answer_length": len(answer),
                            "total_time": round(time.time() - start_time, 2),
                        },
                    )

        return content

    except Exception as e:
        if progress_callback:
            try:
                progress_callback(
                    event_type="error",
                    message=f"❌ LLM调用失败: {str(e)[:200]}",
                    details={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )
            except:
                pass
        raise


def _extract_thinking_and_answer(content: str) -> tuple:
    """
    从内容中提取思考过程和答案

    Returns:
        (thinking, answer) 元组
    """
    thinking = ""
    answer = ""

    # 紧凑格式匹配
    compact_thinking_match = re.search(
        r"💭\s*思考[：:]\s*(.*?)(?=💡\s*答案|$)",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if compact_thinking_match:
        thinking = compact_thinking_match.group(1).strip()

    compact_answer_match = re.search(
        r"💡\s*答案[：:]\s*(.*?)$",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if compact_answer_match:
        answer = compact_answer_match.group(1).strip()

    # 如果没有找到特定格式，尝试简单分割
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

    # 如果都没找到，返回全部内容作为答案
    if not thinking and not answer:
        answer = content

    return thinking, answer


def _split_into_chunks(text: str, chunk_size: int = 100) -> List[str]:
    """
    将文本分割成小块（用于模拟流式输出）
    """
    chunks = []
    sentences = re.split(r"[。！？.\n]", text)

    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
