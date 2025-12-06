import uuid

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from common.factory import get_default_model, get_reasoning_model
from common.prompts import CellPrompt
from common.util.retrieval_utils import remove_think_tags
from common.util.word_exporter import export_planning_to_word
from schema.common_schemas import QueryExpansion
from usecases.cell.cell_config import get_cell_runnable_config
from usecases.cell.state.state import ParallelPlanState, RetrievalState
from usecases.retrieval.tools import retrieve, web_search_node


def query_rewriter(state: RetrievalState, config: RunnableConfig):
    """Agent节点 - 查询优化"""
    prompt = ChatPromptTemplate.from_template(CellPrompt.QUERY_EXPANSION_PROMPT)
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


def route_to_parallel_plans(state: RetrievalState):
    """路由函数 - 返回Send列表用于并行处理"""
    print(f"\n===== 并行分发节点 =====")
    print(f"查询数量: {len(state.optimized_questions)}")

    # 为每个查询创建Send对象，发送到generate_single_plan节点
    sends = []
    for query in state.optimized_questions:
        send_obj = Send(
            "generate_single_plan",
            {
                "original_question": state.original_question,
                "optimized_questions": state.optimized_questions,
                "context": state.context,
                "query": query,
                "individual_plans": [],
            },
        )
        sends.append(send_obj)

    print(f"创建了 {len(sends)} 个Send对象")
    return sends


def generate_single_plan(state, config: RunnableConfig):
    """单个计划生成节点 - 作为Send API的接收端"""
    print(f"\n===== 生成单个计划 =====")

    # Send API发送的是字典，直接访问键值
    query = state["query"]
    context = state["context"]

    print(f"处理查询: {query}")

    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CellPrompt.SYSTEMT_PLAN_GENERATION_PROMPT),
            ("user", CellPrompt.USER_PLAN_GENERATION_PROMPT),
        ]
    )

    model = get_default_model(config)
    chain = planner_prompt | model | StrOutputParser() | remove_think_tags

    plan_response = chain.invoke(
        {
            "objective": [query],  # 传入单个查询作为列表
            "context": context,
        }
    )

    print(f"生成计划长度: {len(plan_response)} 字符")

    # 返回到individual_plans列表中（会被reducer自动合并）
    return {"individual_plans": [plan_response]}


def integrate_plans(state: ParallelPlanState, config: RunnableConfig):
    """整合多个并行计划"""
    print(f"\n===== 整合并行计划 =====")
    print(f"收到计划数量: {len(state.individual_plans)}")

    if len(state.individual_plans) == 1:
        # 如果只有一个计划，直接使用
        integrated_plan = state.individual_plans[0]
    else:
        # 如果有多个计划，使用LLM整合
        # 构建结构化的参数字典
        integration_params = {
            # 分别构建每个研究目标
            "objective_1": state.optimized_questions[0]
            if len(state.optimized_questions) > 0
            else "",
            "objective_2": state.optimized_questions[1]
            if len(state.optimized_questions) > 1
            else "",
            "objective_3": state.optimized_questions[2]
            if len(state.optimized_questions) > 2
            else "",
            "objective_4": state.optimized_questions[3]
            if len(state.optimized_questions) > 3
            else "",
            # 分别构建每个分析计划
            "plan_1": state.individual_plans[0]
            if len(state.individual_plans) > 0
            else "",
            "plan_2": state.individual_plans[1]
            if len(state.individual_plans) > 1
            else "",
            "plan_3": state.individual_plans[2]
            if len(state.individual_plans) > 2
            else "",
            "plan_4": state.individual_plans[3]
            if len(state.individual_plans) > 3
            else "",
        }

        integration_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CellPrompt.INTEGRATION_SYSTEM_PROMPT),
                ("user", CellPrompt.INTEGRATION_USER_PROMPT),
            ]
        )
        model = get_default_model(config)
        integration_chain = (
            integration_prompt | model | StrOutputParser() | remove_think_tags
        )
        integrated_plan = integration_chain.invoke(integration_params)

    print(f"整合后计划长度: {len(integrated_plan)} 字符")

    # 构建完整的objective用于日志
    objective = ""
    for i, q in enumerate(state.optimized_questions):
        objective += f"**Requirement {i + 1}**: {q}\n\n"

    print(f"\n===== 发送到LLM的提示词参数 =====")
    print(f"Objective: {objective}")
    print(f"Context: {state.context}")
    state.generated_plan = integrated_plan
    return state


def create_parallel_rag_graph():
    """创建支持并行处理的RAG图"""
    workflow = StateGraph(ParallelPlanState)

    # 添加节点
    workflow.add_node("query_rewriter", query_rewriter)
    workflow.add_node("retrieval_agent", retrieval_agent)
    workflow.add_node("generate_single_plan", generate_single_plan)
    workflow.add_node("integrate_plans", integrate_plans)

    # 设置入口点和边
    workflow.set_entry_point("query_rewriter")
    workflow.add_edge("query_rewriter", "retrieval_agent")

    # 直接从retrieval_agent使用conditional_edges进行并行分发
    workflow.add_conditional_edges(
        "retrieval_agent",
        route_to_parallel_plans,
        ["generate_single_plan"],  # 指定可能的目标节点
    )

    # 单个计划 -> 整合
    workflow.add_edge("generate_single_plan", "integrate_plans")
    workflow.add_edge("integrate_plans", END)

    graph = workflow.compile()

    # 打印流程图
    try:
        print("\n===== 并行LangGraph工作流程图 =====")
        print(graph.get_graph().draw_mermaid())
    except Exception as e:
        print(f"生成流程图时出错: {str(e)}")

    return graph


def run_parallel_rag_graph(original_question: str, config: RunnableConfig):
    """运行并行RAG图"""
    graph = create_parallel_rag_graph()
    config = get_cell_runnable_config(uuid.uuid4())

    initial_state = ParallelPlanState(
        original_question=original_question,
        optimized_questions=[],
        generated_plan="",
        context="",
        query="",
        individual_plans=[],
    )

    # 让图自然流转
    final_state = None
    for event in graph.stream(initial_state, config):
        print(f"当前节点: {list(event.keys())}")
        final_state = event

    if final_state and "integrate_plans" in final_state:
        generated_plan = final_state["integrate_plans"]
        # 导出Word文档
        try:
            output_path = export_planning_to_word(
                original_question=generated_plan["original_question"],
                optimized_queries=generated_plan["optimized_questions"],
                context=generated_plan["context"],
                individual_plans=generated_plan["individual_plans"],
                integrated_plan=generated_plan["generated_plan"],
                output_dir="D:\\PartTimeJob\\antibody_gen\\output",
            )
            print(f"\n===== Word文档已导出 =====")
            print(f"文件路径: {output_path}")
        except Exception as e:
            print(f"\n===== Word文档导出失败 =====")
            print(f"错误信息: {str(e)}")
    else:
        print("警告: 未找到integrate_plans结果")
        generated_plan = {"generated_plan": ""}

    print(f"\n并行流程完成")
    print(f"问题: {original_question}")

    return generated_plan


def complete_rag_pipeline(original_question: str, config: RunnableConfig):
    """完整的RAG流程"""
    print("=== LangGraph RAG检索流程开始 ===")
    result = run_parallel_rag_graph(original_question, config)  # 并行版本
    print(f"完整结果: {result}")

    return result
