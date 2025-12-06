import uuid

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from common.factory import get_default_model, get_reasoning_model
from common.prompts import AntibodyPrompt
from common.util.retrieval_utils import remove_think_tags
from schema.common_schemas import QueryExpansion
from usecases.antibody.antibody_config import get_antibody_runnable_config
from usecases.antibody.state.state import RetrievalState
from usecases.retrieval.tools import retrieve, web_search_node


def query_rewriter(state: RetrievalState, config: RunnableConfig):
    """Agent节点 - 查询优化"""
    prompt = ChatPromptTemplate.from_template(AntibodyPrompt.QUERY_EXPANSION_PROMPT)
    reasoning_model = get_reasoning_model(config)
    structured_model = reasoning_model.with_structured_output(QueryExpansion)
    runnable = prompt | structured_model
    response = runnable.invoke({"query": state.original_question})
    optimized_questions = response.queries
    print("优化后的查询:")
    for i, query in enumerate(optimized_questions, 1):
        print(f"  {i}. {query}")
    state.optimized_questions = optimized_questions
    return state


tools = [retrieve, web_search_node]
tool_node = ToolNode(tools)


def retrieval_agent(state: RetrievalState, config: RunnableConfig):
    """检索代理节点 - 直接并行调用两个工具"""
    query = ";".join(state.optimized_questions or [state.original_question])

    # 按照图片方式手动构造包含多个工具调用的AIMessage
    from langchain_core.messages import AIMessage

    message_with_multiple_tool_calls = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "retrieve",
                "args": {"query": state.optimized_questions, "config": config},
                "id": "tool_call_id_1",
                "type": "tool_call",
            },
            {
                "name": "web_search_node",
                "args": {"query": state.optimized_questions},
                "id": "tool_call_id_2",
                "type": "tool_call",
            },
        ],
    )

    # 直接调用ToolNode，自动并行执行两个工具
    result = tool_node.invoke({"messages": [message_with_multiple_tool_calls]})

    # 合并工具结果
    tool_results = []
    for msg in result.get("messages", []):
        if hasattr(msg, "content") and msg.content:
            tool_results.append(msg.content)

    context = "\n\n".join(tool_results)

    state.context = context
    state.optimized_question = query
    return state


def generate(state: RetrievalState, config: RunnableConfig):
    """生成节点 - 只负责生成最终答案"""
    query = state.optimized_questions
    context = state.context
    input = ""
    for i, q in enumerate(query):
        input += f"**Requirement {i + 1}**: {q}\n\n"
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", AntibodyPrompt.SYSTEMT_PLAN_GENERATION_PROMPT),
            ("user", AntibodyPrompt.USER_PLAN_GENERATION_PROMPT),
        ]
    )

    model = get_default_model(config)
    chain = planner_prompt | model | StrOutputParser() | remove_think_tags

    plan_response = chain.invoke({"input": input, "context": context})

    # 输出到控制台
    print("\n===== Input =====\n")
    print(input)
    print("\n===== Context =====\n")
    print(context)
    print("\n===== Generated Plan =====\n")
    print(plan_response)
    state.optimized_question = input
    state.generated_plan = plan_response
    return state


def create_rag_graph():
    """创建简化的RAG图 - 直接并行工具调用"""
    workflow = StateGraph(RetrievalState)

    # 添加节点
    workflow.add_node("query_rewriter", query_rewriter)
    workflow.add_node("retrieval_agent", retrieval_agent)
    workflow.add_node("generate", generate)

    # 设置入口点
    workflow.set_entry_point("query_rewriter")
    workflow.add_edge("query_rewriter", "retrieval_agent")
    workflow.add_edge("retrieval_agent", "generate")
    workflow.add_edge("generate", END)

    graph = workflow.compile()

    # 打印流程图
    try:
        print("\n===== LangGraph工作流程图 =====")
        print(graph.get_graph().draw_mermaid())
    except Exception as e:
        print(f"生成流程图时出错: {str(e)}")

    return graph


def run_rag_graph(original_question: str, config: RunnableConfig):
    """运行RAG图"""
    graph = create_rag_graph()

    initial_state = RetrievalState(
        original_question=original_question,
        optimized_question="",
        optimized_questions=[],
        generated_plan="",
        context="",
    )

    # 配置
    config = get_antibody_runnable_config(uuid.uuid4())

    # 让图自然流转
    final_state = None
    for event in graph.stream(initial_state, config):
        print(f"当前节点: {list(event.keys())}")
        final_state = event

    generated_plan = final_state.get("generate", {})

    print(f"\n流程完成")
    print(f"问题: {original_question}")
    print(
        f"上下文长度: {len(final_state.get('generate', {}).get('context', '')) if final_state else 0} 字符"
    )

    return generated_plan


def complete_rag_pipeline(original_question: str, config: RunnableConfig):
    """完整的RAG流程"""
    print("=== LangGraph RAG检索流程开始 ===")

    result = run_rag_graph(original_question, config)

    print(f"完整结果: {result}")

    return result


# 测试完整流程
if __name__ == "__main__":
    original_question = "Design antibodies targeting protein-protein interactions, specifically inhibitors of p53-MDM2 interaction"
    from usecases._debug import get_debug_runnable_config

    rc = get_debug_runnable_config()
    final_state = complete_rag_pipeline(original_question, rc)
