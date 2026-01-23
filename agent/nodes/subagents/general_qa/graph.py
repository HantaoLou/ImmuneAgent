"""
General QA Agent 子图

负责处理用户的普通问答请求，使用LLM提供科学、严谨的回答。
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import sys
import json
import re
from pathlib import Path

from .prompt import GENERAL_QA_SYSTEM_PROMPT, get_general_qa_user_prompt

# 导入主图状态（用于状态映射）
# 添加agent目录到路径（支持从子图目录导入）
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import GlobalState

# LLM相关导入（使用公共LLM工厂）
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from utils.llm_factory import create_reasoning_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    create_reasoning_llm = None
    HumanMessage = None
    SystemMessage = None
    print("警告：langchain相关库未安装，普通问答功能将不可用")


# ---------------------- General QA 状态模型 ----------------------
class GeneralQAState(BaseModel):
    """General QA Agent 子图状态"""
    user_input: str = Field(description="用户原始输入（问题）")
    answer: Optional[str] = Field(default=None, description="LLM生成的回答")
    confidence: Optional[str] = Field(default=None, description="回答的置信度说明")
    related_topics: List[str] = Field(default_factory=list, description="相关问题或话题")
    sources_suggested: List[str] = Field(default_factory=list, description="建议的参考资料或研究方向")


# ---------------------- LLM实例化（使用公共LLM工厂） ----------------------
def _get_llm():
    """
    获取推理模型实例（用于普通问答）
    
    使用公共LLM工厂创建推理模型，优先使用推理性能好的模型。
    
    Returns:
        LLM实例，如果都不可用则返回None
    """
    if not LLM_AVAILABLE or create_reasoning_llm is None:
        return None
    
    # 使用推理模型（用于普通问答，温度稍高以获得更自然的回答）
    return create_reasoning_llm(temperature=0.3)


# ---------------------- 节点1：普通问答处理节点 ----------------------
def general_qa_answer_node(state: GeneralQAState) -> GeneralQAState:
    """
    普通问答处理节点
    
    使用LLM回答用户的普通问题，遵循科学性和严谨性原则。
    
    Args:
        state: General QA 子图状态
    
    Returns:
        更新后的状态（包含回答）
    """
    user_input = state.user_input
    
    # 使用LLM生成回答及相关信息
    llm = _get_llm()
    if llm is not None:
        result = _generate_answer_with_llm(user_input, llm)
        if result:
            state.answer = result.get("answer", "")
            state.confidence = result.get("confidence", "")
            state.related_topics = result.get("related_topics", [])
            state.sources_suggested = result.get("sources_suggested", [])
            print(f"✓ 已生成完整回答（使用LLM）")
            if state.confidence:
                print(f"  置信度：{state.confidence}")
            if state.related_topics:
                print(f"  相关问题：{len(state.related_topics)}个")
            if state.sources_suggested:
                print(f"  参考资料：{len(state.sources_suggested)}个")
        else:
            # LLM失败时的降级方案
            fallback_result = _generate_fallback_answer(user_input)
            state.answer = fallback_result["answer"]
            state.confidence = fallback_result.get("confidence", "由于LLM不可用，无法评估置信度")
            state.related_topics = fallback_result.get("related_topics", [])
            state.sources_suggested = fallback_result.get("sources_suggested", [])
            print(f"⚠ 使用降级方案生成回答")
    else:
        # LLM不可用时的降级方案
        fallback_result = _generate_fallback_answer(user_input)
        state.answer = fallback_result["answer"]
        state.confidence = fallback_result.get("confidence", "由于LLM不可用，无法评估置信度")
        state.related_topics = fallback_result.get("related_topics", [])
        state.sources_suggested = fallback_result.get("sources_suggested", [])
        print(f"⚠ LLM不可用，使用降级方案")
    
    return state


def _generate_answer_with_llm(user_input: str, llm) -> Optional[Dict[str, Any]]:
    """
    使用LLM生成回答及相关信息
    
    Args:
        user_input: 用户输入的问题
        llm: LLM实例
    
    Returns:
        包含回答、置信度、相关问题、参考资料的字典，如果失败返回None
    """
    # 使用集中的提示词模板
    system_prompt = GENERAL_QA_SYSTEM_PROMPT
    user_prompt = get_general_qa_user_prompt(user_input)
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # 尝试解析JSON格式的响应
        result = _parse_llm_response(response_text)
        
        return result
        
    except Exception as e:
        # 检查是否是认证错误（API Key 错误）
        error_str = str(e).lower()
        if "authentication" in error_str or "api key" in error_str or "401" in error_str:
            print(f"⚠ LLM API Key 认证失败，将使用降级方案: {type(e).__name__}")
            print(f"  提示：请检查环境变量中的 API Key 是否正确配置")
        elif "rate limit" in error_str or "429" in error_str:
            print(f"⚠ LLM API 调用频率限制，将使用降级方案: {type(e).__name__}")
        else:
            print(f"⚠ LLM生成回答失败，将使用降级方案: {type(e).__name__}: {str(e)[:100]}")
        
        # 不在测试环境中打印完整堆栈（避免输出过多）
        import os
        if os.getenv("DEBUG_LLM_ERRORS", "false").lower() == "true":
            import traceback
            traceback.print_exc()
        
        return None


def _parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    解析LLM返回的响应，提取结构化信息
    
    Args:
        response_text: LLM返回的文本
    
    Returns:
        包含answer、confidence、related_topics、sources_suggested的字典
    """
    # 方法1：尝试直接解析整个响应为JSON
    try:
        result = json.loads(response_text.strip())
        if isinstance(result, dict):
            return {
                "answer": result.get("answer", response_text),
                "confidence": result.get("confidence", "中等置信度"),
                "related_topics": result.get("related_topics", []),
                "sources_suggested": result.get("sources_suggested", [])
            }
    except json.JSONDecodeError:
        pass
    
    # 方法2：尝试提取JSON代码块
    json_block_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'\{[^{}]*"answer"[^{}]*\}',  # 包含answer字段的JSON对象
    ]
    
    for pattern in json_block_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict) and "answer" in result:
                    return {
                        "answer": result.get("answer", response_text),
                        "confidence": result.get("confidence", "中等置信度"),
                        "related_topics": result.get("related_topics", []),
                        "sources_suggested": result.get("sources_suggested", [])
                    }
            except json.JSONDecodeError:
                continue
    
    # 方法3：尝试提取嵌套的JSON对象（更健壮的正则）
    # 查找最外层的JSON对象
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(response_text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    json_str = response_text[start_idx:i+1]
                    result = json.loads(json_str)
                    if isinstance(result, dict) and "answer" in result:
                        return {
                            "answer": result.get("answer", response_text),
                            "confidence": result.get("confidence", "中等置信度"),
                            "related_topics": result.get("related_topics", []),
                            "sources_suggested": result.get("sources_suggested", [])
                        }
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    
    # 方法4：如果JSON解析失败，尝试从文本中提取信息
    # 或者直接使用原始文本作为回答
    print("警告：无法解析JSON格式，使用文本提取方式")
    return {
        "answer": response_text,
        "confidence": "中等置信度（无法从响应中提取置信度信息）",
        "related_topics": _extract_related_topics_from_text(response_text),
        "sources_suggested": _extract_sources_from_text(response_text)
    }


def _extract_related_topics_from_text(text: str) -> List[str]:
    """
    从文本中提取相关问题（降级方案）
    
    Args:
        text: 文本内容
    
    Returns:
        相关问题列表
    """
    # 简单的关键词提取（可以后续优化）
    topics = []
    
    # 查找"相关问题"、"相关话题"等关键词后的内容
    patterns = [
        r'相关问题[：:]\s*(.+?)(?:\n|$)',
        r'相关话题[：:]\s*(.+?)(?:\n|$)',
        r'related topics[：:]\s*(.+?)(?:\n|$)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # 分割并清理
            items = [item.strip() for item in re.split(r'[，,、]', match) if item.strip()]
            topics.extend(items)
    
    return topics[:5] if topics else []


def _extract_sources_from_text(text: str) -> List[str]:
    """
    从文本中提取参考资料（降级方案）
    
    Args:
        text: 文本内容
    
    Returns:
        参考资料列表
    """
    sources = []
    
    # 查找"参考资料"、"建议"等关键词后的内容
    patterns = [
        r'参考资料[：:]\s*(.+?)(?:\n|$)',
        r'建议[：:]\s*(.+?)(?:\n|$)',
        r'建议参考[：:]\s*(.+?)(?:\n|$)',
        r'sources[：:]\s*(.+?)(?:\n|$)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # 分割并清理
            items = [item.strip() for item in re.split(r'[，,、]', match) if item.strip()]
            sources.extend(items)
    
    return sources[:5] if sources else []


def _generate_fallback_answer(user_input: str) -> Dict[str, Any]:
    """
    降级方案：当LLM不可用时生成简单回答
    
    Args:
        user_input: 用户输入的问题
    
    Returns:
        包含回答、置信度、相关问题、参考资料的字典
    """
    return {
        "answer": f"""感谢您的问题："{user_input}"

由于当前LLM服务不可用，我无法提供详细的科学回答。建议您：
1. 查阅相关的学术文献或专业资料
2. 咨询相关领域的专家
3. 在配置LLM API Key后重新提问以获得更专业的回答

对于科研相关问题，建议参考：
- 相关领域的学术数据库（如PubMed、arXiv等）
- 专业期刊和会议论文
- 领域专家的研究成果""",
        "confidence": "无法评估（LLM服务不可用）",
        "related_topics": [
            "相关领域的学术研究",
            "专业文献检索",
            "专家咨询"
        ],
        "sources_suggested": [
            "PubMed - 生物医学文献数据库",
            "arXiv - 科学预印本数据库",
            "相关领域的专业期刊",
            "领域专家的研究成果",
            "学术会议论文"
        ]
    }


# ---------------------- 状态映射函数 =====================
def general_qa_input_mapper(global_state: GlobalState) -> GeneralQAState:
    """
    主图→子图的状态映射
    
    将主图的 GlobalState 映射为 GeneralQAState，提取子图需要的信息。
    
    Args:
        global_state: 主图的全局状态
    
    Returns:
        GeneralQAState: 子图状态
    """
    return GeneralQAState(
        user_input=global_state.user_input,
        answer=None,  # 将在子图中生成
        confidence=None,
        related_topics=[],
        sources_suggested=[]
    )


def general_qa_output_mapper(subgraph_output: GeneralQAState | dict, global_state: GlobalState) -> GlobalState:
    """
    子图→主图的状态映射
    
    将子图的 GeneralQAState 结果同步回主图的 GlobalState。
    
    Args:
        subgraph_output: 子图输出的状态（可能是 GeneralQAState 对象或字典）
        global_state: 主图的全局状态（将被更新）
    
    Returns:
        GlobalState: 更新后的主图状态
    """
    
    # 处理字典格式的状态（LangGraph 可能返回字典）
    if isinstance(subgraph_output, dict):
        subgraph_output = GeneralQAState(**subgraph_output)
    
    # 将回答存储到 merged_result 中
    if not global_state.merged_result:
        global_state.merged_result = {}
    
    if subgraph_output.answer:
        global_state.merged_result["general_qa_answer"] = subgraph_output.answer
    
    if subgraph_output.confidence:
        global_state.merged_result["general_qa_confidence"] = subgraph_output.confidence
    
    if subgraph_output.related_topics:
        global_state.merged_result["general_qa_related_topics"] = subgraph_output.related_topics
    
    if subgraph_output.sources_suggested:
        global_state.merged_result["general_qa_sources"] = subgraph_output.sources_suggested
    
    # 返回更新后的全局状态
    return global_state


# ---------------------- 构建 General QA Agent 子图 ----------------------
def build_general_qa_subgraph():
    """
    构建普通问答Agent子图
    
    使用公共LLM工厂创建LLM实例，优先使用通义千问，其次使用其他模型。
    
    Returns:
        编译后的子图
    """
    graph = StateGraph(GeneralQAState)
    
    # 添加节点
    graph.add_node("answer_question", general_qa_answer_node)
    
    # 定义流转规则
    graph.add_edge(START, "answer_question")
    graph.add_edge("answer_question", END)
    
    return graph.compile()

