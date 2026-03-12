"""
Simplified GeneralQA Subgraph - Core: Deep Research + X-Masters

流程设计：
1. N0: 简单预处理 (问题分类、领域检测)
2. Deep Research: 深度知识检索与研究
3. X-Masters: 多路径推理 + 批评 + 综合选择
4. N8: 答案格式化与输出

核心理念：
- Deep Research 负责提供充分的知识
- X-Masters 负责多路径推理和验证
- 移除冗余的中间节点，让LLM自由发挥

使用方式：
    USE_SIMPLIFIED_GRAPH=true
"""

from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, START, END
import json
import os
import time

# Import state
from .state import GeneralQAState

# Import X-Masters
try:
    from agent.nodes.subagents.x_masters.graph import (
        build_graph as build_xmasters_graph,
    )

    XMASTERS_AVAILABLE = True
except ImportError:
    XMASTERS_AVAILABLE = False
    print("Warning: X-Masters not available")

# Import Deep Research
try:
    from agent.nodes.subagents.deep_research.deep_researcher import (
        graph as deep_research_graph,
    )

    DEEP_RESEARCH_AVAILABLE = True
except ImportError:
    DEEP_RESEARCH_AVAILABLE = False
    print("Warning: Deep Research not available")

# Import LLM factory
from agent.utils.llm_factory import create_bioinformatics_llm, is_llm_available

# Try importing langchain
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: langchain not available")

# Use llm_factory's availability check (which properly checks zhipuai)
LLM_AVAILABLE = is_llm_available()


# ===================== Configuration =====================

# 是否启用Deep Research (默认启用)
ENABLE_DEEP_RESEARCH = os.getenv("ENABLE_DEEP_RESEARCH", "true").lower() == "true"

# X-Masters solver数量
XMASTERS_NUM_SOLVERS = int(os.getenv("XMASTERS_NUM_SOLVERS", "3"))

# ===================== LLM Helper =====================

_llm_instance = None


def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        if not LLM_AVAILABLE:
            print(f"[graph_simplified] LLM unavailable, cannot create LLM")
            return None
        _llm_instance = create_bioinformatics_llm(temperature=0.2)
        print(f"[graph_simplified] LLM created: {type(_llm_instance).__name__}")
    return _llm_instance


def _call_llm(prompt: str, node_name: str = "") -> Optional[str]:
    llm = _get_llm()
    if llm is None:
        return None

    try:
        start_time = time.time()
        response = llm.invoke([HumanMessage(content=prompt)])
        elapsed = time.time() - start_time
        print(f"  [{node_name}] LLM call completed in {elapsed:.1f}s")

        if hasattr(response, "content"):
            return response.content.strip()
        return str(response).strip()
    except Exception as e:
        print(f"  [{node_name}] LLM error: {e}")
        return None


def _call_llm_simple(prompt: str, node_name: str = "") -> Optional[str]:
    return _call_llm(prompt, node_name)


def _parse_json(response: str) -> Optional[Dict]:
    """Parse JSON from response"""
    if not response:
        return None

    import re

    try:
        return json.loads(response)
    except:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None


# ===================== N0: Simple Preprocessing =====================

N0_PROMPT = """分析这个生物医学问题，提取关键信息。

问题:
{user_input}

输出JSON格式:
{{
    "cleaned_text": "清理后的问题文本",
    "question_type": "multiple_choice|true_false|calculation|explanation|short_answer",
    "answer_format": "single_letter|true/false|number|text|sequence",
    "options": ["A选项文本", "B选项文本", ...] 或 [] 如果没有选项,
    "domain": "genetics|immunology|biochemistry|bioinformatics|clinical|microbiology|general",
    "key_terms": ["关键词1", "关键词2", ...]
}}

规则:
- True/False判断题 → question_type: "true_false", answer_format: "true/false"
- 多选题 → question_type: "multiple_choice", answer_format: "single_letter"
- 需要数值计算 → question_type: "calculation", answer_format: "number"
- 提取实际选项文本，不要只提取标签
"""


def n0_preprocess_node(state: GeneralQAState) -> GeneralQAState:
    """
    N0: 简单预处理
    只做问题分类和关键信息提取，不做复杂分析
    """
    print("\n" + "=" * 60)
    print("N0: 简单预处理 (Simplified)")
    print("=" * 60)

    if not state.user_input or not state.user_input.strip():
        state.error_message = "Empty input"
        return state

    # Get prompt
    prompt = N0_PROMPT.format(user_input=state.user_input)

    # Call LLM
    response = _call_llm_simple(prompt, "N0")

    if response is None:
        state.error_message = "N0 LLM call failed"
        return state

    # Parse response
    result = _parse_json(response)

    if result is None:
        # Fallback
        state.cleaned_text = state.user_input
        state.question_type_label = "Multiple Choice"
        state.answer_format_label = "Single Choice"
        state.core_domains = ["general"]
        return state

    # Update state
    state.cleaned_text = result.get("cleaned_text", state.user_input)
    state.question_type_label = result.get("question_type", "Multiple Choice")
    state.answer_format_label = result.get("answer_format", "Single Choice")
    state.question_options = result.get("options", [])
    state.core_domains = [result.get("domain", "general")]
    state.core_keywords = result.get("key_terms", [])

    print(f"  [OK] 问题类型: {state.question_type_label}")
    print(f"  [OK] 领域: {state.core_domains}")
    print(
        f"  [OK] 选项数: {len(state.question_options) if state.question_options else 0}"
    )

    return state


# ===================== Deep Research Integration =====================


def deep_research_node(state: GeneralQAState) -> GeneralQAState:
    """
    Deep Research: 深度知识检索
    使用deep_research子图获取充分的知识背景
    """
    print("\n" + "=" * 60)
    print("[Deep Research] 深度知识检索")
    print("=" * 60)

    if not DEEP_RESEARCH_AVAILABLE:
        print("  [WARN] Deep Research不可用，跳过")
        state.domain_knowledge_map = {"general": {"facts": [], "context": ""}}
        return state

    try:
        # 构建研究问题
        research_question = state.cleaned_text

        # 添加选项上下文（如果有）
        if state.question_options:
            options_text = "\n".join(
                [
                    f"{chr(65 + i)}. {opt}"
                    for i, opt in enumerate(state.question_options)
                ]
            )
            research_question = f"""
问题: {state.cleaned_text}

选项:
{options_text}

请深入研究这个问题，检索相关知识和文献，为回答问题提供充分的知识背景。
"""

        print(f"  [INFO] 研究问题: {state.cleaned_text[:100]}...")

        # 调用Deep Research
        from agent.nodes.subagents.deep_research.configuration import Configuration

        config = Configuration(
            max_researcher_iterations=2,  # 限制迭代次数以提高速度
            max_concurrent_research_units=2,
        )

        # 运行deep research
        result = deep_research_graph.invoke(
            {"messages": [HumanMessage(content=research_question)]},
            config={"configurable": config.__dict__},
        )

        # 提取研究结果
        if result:
            # 获取最终报告或消息
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    research_context = last_message.content
                else:
                    research_context = str(last_message)
            else:
                research_context = ""

            # 存储到state
            domain = state.core_domains[0] if state.core_domains else "general"
            state.domain_knowledge_map = {
                domain: {
                    "context": research_context,
                    "facts": [],
                    "source": "deep_research",
                }
            }
            state.deep_research_result = {"context": research_context, "success": True}

            print(
                f"  [OK] Deep Research完成，获取 {len(research_context)} 字符的知识背景"
            )
        else:
            print("  [WARN] Deep Research返回空结果")
            state.domain_knowledge_map = {"general": {"facts": [], "context": ""}}

    except Exception as e:
        print(f"  [FAIL] Deep Research失败: {e}")
        state.domain_knowledge_map = {"general": {"facts": [], "context": ""}}

    return state


# ===================== X-Masters Integration =====================


def xmasters_node(state: GeneralQAState) -> GeneralQAState:
    """
    X-Masters: 多路径推理
    使用X-Masters进行多解法、批评、综合、选择
    """
    print("\n" + "=" * 60)
    print("[X-Masters] X-Masters: 多路径推理")
    print("=" * 60)

    if not XMASTERS_AVAILABLE:
        print("  [WARN] X-Masters不可用，使用单路径推理")
        return _fallback_inference(state)

    try:
        # 构建问题
        problem = state.cleaned_text

        # 添加选项
        if state.question_options:
            options_text = "\n".join(
                [
                    f"{chr(65 + i)}. {opt}"
                    for i, opt in enumerate(state.question_options)
                ]
            )
            problem = f"""
问题: {state.cleaned_text}

选项:
{options_text}
"""

        # 添加Deep Research上下文
        if state.domain_knowledge_map:
            domain = state.core_domains[0] if state.core_domains else "general"
            domain_data = state.domain_knowledge_map.get(domain, {})
            context = domain_data.get("context", "")
            if context:
                problem += f"""

背景知识:
{context[:4000]}
"""

        print(f"  [INFO] 问题长度: {len(problem)} 字符")
        print(f"  [RUN] 使用 {XMASTERS_NUM_SOLVERS} 个Solver")

        # 构建X-Masters图
        xmasters_graph = build_xmasters_graph().compile()

        # 运行X-Masters
        from agent.utils.llm_factory import get_current_llm_config

        llm_config = get_current_llm_config()

        result = xmasters_graph.invoke(
            {
                "problem": problem,
                "num_solvers": XMASTERS_NUM_SOLVERS,
                "llm": llm_config.get("model", ""),
                "source": llm_config.get("provider", ""),
                "temperature": 0.7,
                "timeout_seconds": 300,
            }
        )

        # 提取最终答案
        final_answer = result.get("final_answer", "")

        if final_answer:
            state.final_answer = _extract_answer(final_answer, state)
            state.core_conclusion = final_answer
            print(f"  [OK] X-Masters完成")
            print(f"  [OK] 答案: {state.final_answer}")
        else:
            print("  [WARN] X-Masters返回空答案，使用备用推理")
            return _fallback_inference(state)

    except Exception as e:
        print(f"  [FAIL] X-Masters失败: {e}")
        return _fallback_inference(state)

    return state


def _fallback_inference(state: GeneralQAState) -> GeneralQAState:
    """
    备用推理：当X-Masters不可用时的单路径推理
    """
    print("\n  使用备用单路径推理...")

    # 构建推理提示
    prompt = f"""基于以下知识回答问题。

问题: {state.cleaned_text}

"""

    # 添加选项
    if state.question_options:
        options_text = "\n".join(
            [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(state.question_options)]
        )
        prompt += f"选项:\n{options_text}\n\n"

    # 添加知识背景
    if state.domain_knowledge_map:
        domain = state.core_domains[0] if state.core_domains else "general"
        domain_data = state.domain_knowledge_map.get(domain, {})
        context = domain_data.get("context", "")
        if context:
            prompt += f"背景知识:\n{context[:3000]}\n\n"

    prompt += """
输出JSON格式:
{
    "analysis": "分析过程",
    "answer": "最终答案 (多选题为单个字母如A/B/C，True/False题为True或False)"
}
"""

    response = _call_llm_simple(prompt, "Fallback")

    if response:
        result = _parse_json(response)
        if result:
            state.final_answer = _extract_answer(result.get("answer", ""), state)
            state.core_conclusion = result.get("analysis", "")
            print(f"  [OK] 备用推理完成，答案: {state.final_answer}")
        else:
            state.final_answer = response[:100]
            state.core_conclusion = response
    else:
        state.error_message = "推理失败"

    return state


def _extract_answer(raw_answer: str, state: GeneralQAState) -> str:
    """
    从原始回答中提取最终答案
    """
    if not raw_answer:
        return ""

    raw_answer = raw_answer.strip()

    # True/False问题
    if (
        state.answer_format_label == "true/false"
        or state.question_type_label == "true_false"
    ):
        if "true" in raw_answer.lower():
            return "True"
        elif "false" in raw_answer.lower():
            return "False"

    # 多选题
    if state.question_options:
        # 尝试提取选项字母
        import re

        match = re.search(r"\b([A-E])\b", raw_answer.upper())
        if match:
            return match.group(1)

        # 如果回答中包含完整选项文本，匹配到对应字母
        for i, opt in enumerate(state.question_options):
            if opt.lower() in raw_answer.lower():
                return chr(65 + i)

    # 数值计算题 - 提取数字
    if (
        state.answer_format_label == "number"
        or state.question_type_label == "calculation"
    ):
        import re

        numbers = re.findall(r"[-+]?\d*\.?\d+", raw_answer)
        if numbers:
            return numbers[-1]  # 返回最后一个数字

    return raw_answer[:200]  # 默认返回前200字符


# ===================== N8: Answer Formatting =====================


def n8_format_node(state: GeneralQAState) -> GeneralQAState:
    """
    N8: 答案格式化
    确保答案格式符合要求
    """
    print("\n" + "=" * 60)
    print("N8: 答案格式化")
    print("=" * 60)

    if not state.final_answer:
        print("  [WARN] 无答案可格式化")
        return state

    # 验证答案格式
    answer = state.final_answer.strip()
    format_label = state.answer_format_label or "Single Choice"

    # True/False验证
    if format_label == "true/false":
        if answer.lower() not in ["true", "false"]:
            # 尝试纠正
            if "true" in answer.lower():
                answer = "True"
            elif "false" in answer.lower():
                answer = "False"

    # 多选题验证
    elif format_label == "Single Choice" and state.question_options:
        import re

        match = re.search(r"^[A-Ea-e]$", answer)
        if not match:
            # 尝试从文本中提取
            match = re.search(r"\b([A-Ea-e])\b", answer.upper())
            if match:
                answer = match.group(1).upper()

    state.final_answer = answer
    state.format_valid_label = "Valid"

    print(f"  [OK] 最终答案: {state.final_answer}")
    print(f"  [OK] 格式: {format_label}")

    return state


# ===================== Routing Functions =====================


def should_skip_deep_research(state: GeneralQAState) -> str:
    """判断是否跳过Deep Research"""
    if not ENABLE_DEEP_RESEARCH:
        return "skip"

    # 简单问题可以跳过
    if state.question_type_label == "true_false":
        # True/False问题通常不需要深度研究
        if len(state.cleaned_text) < 200:
            return "skip"

    return "research"


# ===================== Build Graph =====================


def build_simplified_general_qa_graph():
    """
    构建简化版General QA子图

    流程:
    START → N0(预处理) → Deep Research → X-Masters → N8(格式化) → END
                           ↑              ↑
                        (可选跳过)     (备用推理)
    """

    graph = StateGraph(GeneralQAState)

    # 添加节点
    graph.add_node("n0_preprocess", n0_preprocess_node)
    graph.add_node("deep_research", deep_research_node)
    graph.add_node("xmasters", xmasters_node)
    graph.add_node("n8_format", n8_format_node)

    # 定义边
    graph.add_edge(START, "n0_preprocess")

    # 条件路由：是否跳过Deep Research
    graph.add_conditional_edges(
        "n0_preprocess",
        should_skip_deep_research,
        {
            "research": "deep_research",
            "skip": "xmasters",  # 跳过Deep Research直接推理
        },
    )

    # Deep Research → X-Masters
    graph.add_edge("deep_research", "xmasters")

    # X-Masters → N8
    graph.add_edge("xmasters", "n8_format")

    # N8 → END
    graph.add_edge("n8_format", END)

    return graph.compile()


# ===================== Main Entry Point =====================


def run_simplified_general_qa(user_input: str, **kwargs) -> Dict[str, Any]:
    """
    运行简化版General QA流程

    Args:
        user_input: 问题文本
        **kwargs: 额外参数

    Returns:
        包含final_answer的结果字典
    """
    print("\n" + "=" * 60)
    print("[START] 简化版 GENERAL QA 流程")
    print("   核心: Deep Research + X-Masters")
    print("=" * 60)

    start_time = time.time()

    # 初始化state
    state = GeneralQAState(user_input=user_input)

    # 应用额外参数
    for key, value in kwargs.items():
        if hasattr(state, key):
            setattr(state, key, value)

    # 运行图
    graph = build_simplified_general_qa_graph()

    try:
        result_state = graph.invoke(state)

        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print(f"[SUCCESS] 完成，耗时 {elapsed:.1f}秒")
        print("=" * 60)

        return {
            "final_answer": result_state.final_answer,
            "conclusion": result_state.core_conclusion,
            "duration": elapsed,
            "success": True,
            "nodes_executed": ["n0", "deep_research", "xmasters", "n8"],
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] 流程错误: {e}")

        return {
            "final_answer": None,
            "error": str(e),
            "duration": elapsed,
            "success": False,
        }


# ===================== Export =====================

__all__ = [
    "build_simplified_general_qa_graph",
    "run_simplified_general_qa",
    "ENABLE_DEEP_RESEARCH",
    "XMASTERS_NUM_SOLVERS",
]
