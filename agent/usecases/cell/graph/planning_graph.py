import os
import uuid
from typing import List

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import SimpleJsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt

from common.constants import STANDARDIZED_WORKING_DIRECTORY
from common.factory import get_default_model, get_reasoning_model
from common.prompts import CellPrompt
from common.util.retrieval_utils import remove_think_tags
from usecases.cell.cell_config import get_cell_runnable_config
from usecases.cell.graph.retrieval_graph import create_parallel_rag_graph
from usecases.cell.state.state import State
from usecases.cell.tool.planning_tools import (
    figure_analysis_tools,
    metabcr_tool,
)
from usecases.cell.tool.rds_tools import (
    bcr_standardize_tool,
    process_csv_to_standard_tool,
    rds_standardize_tool,
)


def refine_planning_node(state: State, config: RunnableConfig):
    """精化计划节点 - 让用户确认或修改生成的计划"""
    # 确定要显示的计划内容
    current_plan = (
        state.refine_plan if state.refine_plan != "" else state.generated_plan
    )

    refine_info = {
        "task": "计划确认与精化",
        "description": "请检查生成的计划，如需修改请提供新的计划内容，如无需修改请输入:y/yes/confirm",
    }

    # 显示当前计划并等待用户确认或修改
    user_feedback = interrupt(refine_info)
    if user_feedback == "":
        state.plan_confirmed = False
        return state

    refine_plan = ""
    plan_confirmed = True
    # 处理用户反馈
    if user_feedback.strip().lower() in ["yes", "y", "confirm"]:
        # 用户确认当前计划，标记为已确认
        print("用户确认了当前计划")
        refine_plan = current_plan
    else:
        # 用户提供了修改意见，需要重新精化
        print(f"用户修改了计划: {user_feedback}")
        refine_plan_prompt = ChatPromptTemplate.from_template(
            CellPrompt.REFINE_PLAN_PROMPT
        )
        model = get_default_model(config)
        refine_plan_chain = (
            refine_plan_prompt | model | StrOutputParser() | remove_think_tags
        )
        refine_plan = refine_plan_chain.invoke(
            {"original_plan": current_plan, "user_feedback": user_feedback}
        )
        plan_confirmed = False
    state.refine_plan = refine_plan
    state.plan_confirmed = plan_confirmed
    return state


def input_standardize_file_node(state: State, config: RunnableConfig) -> State:
    """
    标准化计划节点 - 用户上传并标准化RDS或CSV文件
    """
    standardization_info = {
        "task": "文件标准化节点",
        "description": "请上传需要标准化的RDS文件或CSV文件（文件路径必须以.rds或.csv结尾），如果跳过文件标准化节点请输入:skip",
    }

    # 等待用户上传文件
    uploaded_file_path = interrupt(standardization_info)

    # 验证输入是否为空
    if not uploaded_file_path or not uploaded_file_path.strip():
        return state.model_copy(update={"standardize_status": "failed"})

    # 检查是否跳过
    user_input = uploaded_file_path.strip()
    if user_input.lower() == "skip":
        print("用户选择跳过文件标准化节点，跳转到BCR文件处理")
        return state.model_copy(update={"standardize_status": "bcr_processing"})

    # 验证文件类型（必须是RDS或CSV文件）
    file_path = user_input.lower()
    if not (file_path.endswith(".rds") or file_path.endswith(".csv")):
        print(f"文件类型验证失败: {uploaded_file_path}，必须是.rds或.csv文件")
        return state.model_copy(update={"standardize_status": "failed"})

    print(f"文件验证成功: {uploaded_file_path}")

    # 更新状态 - 文件路径有效，准备进入标准化
    return state.model_copy(
        update={
            "standardized_files": uploaded_file_path,
            "standardize_status": "processing",
        }
    )


def standardize_file_node(state: State, config: RunnableConfig) -> State:
    """
    文件标准化执行节点 - 执行RDS或CSV文件的标准化处理
    """
    # 获取文件路径
    file_path = state.standardized_files
    if not file_path:
        return state.model_copy(update={"standardize_status": "failed"})

    # 获取combine_fields输入
    combine_fields_info = {
        "task": "字段组合输入",
        "description": f"请输入需要组合的字段名，用逗号分隔（例如：field1,field2,field3，或者rownames）\n文件路径：{file_path}",
    }

    combine_fields_input = interrupt(combine_fields_info)

    # 验证输入
    if not combine_fields_input or not combine_fields_input.strip():
        # 如果是RDS文件，给默认值rownames
        if file_path.lower().endswith(".rds"):
            combine_fields_input = "rownames"
            print("combine_fields输入为空，RDS文件使用默认值：rownames")
        else:
            print("combine_fields输入为空，标准化失败")
            return state.model_copy(update={"standardize_status": "failed"})

    # 解析字段列表
    combine_fields = [
        field.strip()
        for field in combine_fields_input.strip().split(",")
        if field.strip()
    ]
    if not combine_fields:
        print("combine_fields格式错误，标准化失败")
        return state.model_copy(update={"standardize_status": "failed"})

    try:
        work_directory = config["configurable"]["work_directory"]
        # 根据文件类型调用相应的标准化工具
        if file_path.lower().endswith(".rds"):
            result = rds_standardize_tool(file_path, combine_fields, work_directory)
        elif file_path.lower().endswith(".csv"):
            result = bcr_standardize_tool(file_path, combine_fields, work_directory)
        else:
            print(f"不支持的文件类型: {file_path}")
            return state.model_copy(update={"standardize_status": "failed"})

        print(f"文件标准化成功: {result}")

        # 更新状态
        return state.model_copy(
            update={
                "combine_fields": combine_fields_input.strip(),
                "standardize_status": "completed",
            }
        )

    except Exception as e:
        print(f"文件标准化失败: {str(e)}")
        return state.model_copy(update={"standardize_status": "failed"})


def input_bcr_file_node(state: State, config: RunnableConfig) -> State:
    """
    BCR文件输入节点：用户上传CSV文件
    """
    file_info = {
        "task": "BCR文件输入节点",
        "description": "请上传CSV格式的BCR文件，如果跳过BCR文件处理请输入:skip",
        "requirements": "文件必须是CSV格式",
    }

    # 等待用户上传文件
    uploaded_files_input = interrupt(file_info)

    if not uploaded_files_input or not uploaded_files_input.strip():
        print("未提供文件路径")
        return state.model_copy(update={"bcr_input_valid": False})

    user_input = uploaded_files_input.strip()

    # 检查是否跳过
    if user_input.lower() == "skip":
        print("用户选择跳过BCR文件处理，跳转到模型选择")
        return state.model_copy(update={"bcr_skip": True, "bcr_input_valid": True})

    # 检查文件是否为CSV格式
    if not user_input.lower().endswith(".csv"):
        print("错误：文件必须是CSV格式")
        return state.model_copy(update={"bcr_input_valid": False})

    print(f"BCR文件上传成功: {user_input}")
    return state.model_copy(
        update={"bcr_file_path": user_input, "bcr_input_valid": True, "bcr_skip": False}
    )


def bcr_file_standard_node(state: State, config: RunnableConfig) -> State:
    """
    BCR文件标准化节点：处理CSV文件并生成标准格式
    """
    if not state.bcr_file_path:
        print("错误：未找到BCR文件路径")
        return state.model_copy(update={"bcr_input_valid": False})

    input_info = {
        "task": "BCR文件标准化节点",
        "description": "请输入BCR文件处理参数",
        "format": "请按以下格式输入，用逗号分隔：bcr_code字段名,heavy字段名,light字段名,抗原序列,实验信息",
        "example": "barcode,heavy_chain,light_chain,ATCGATCG,experiment_001",
    }

    # 等待用户输入参数
    user_input = interrupt(input_info)

    if not user_input or not user_input.strip():
        print("未提供输入参数")
        return state.model_copy(update={"bcr_input_valid": False})

    user_input = user_input.strip()

    # 解析输入参数
    try:
        parts = [part.strip() for part in user_input.split(",")]
        if len(parts) != 5:
            print("错误：输入格式不正确，需要5个参数，用逗号分隔")
            return state.model_copy(update={"bcr_input_valid": False})

        bcr_code, heavy, light, variant_seq, experiment = parts

        # 验证字段是否为空
        if not all([bcr_code, heavy, light, variant_seq, experiment]):
            print("错误：所有字段都不能为空")
            return state.model_copy(update={"bcr_input_valid": False})

        print(f"开始处理BCR文件: {state.bcr_file_path}")
        print(
            f"参数: bcr_code={bcr_code}, heavy={heavy}, light={light}, variant_seq={variant_seq}, experiment={experiment}"
        )
        work_directory = config["configurable"]["work_directory"]
        # 调用处理工具
        result = process_csv_to_standard_tool(
            csv_file_path=state.bcr_file_path,
            bar_code=bcr_code,
            heavy=heavy,
            light=light,
            variant_seq=variant_seq,
            experiment=experiment,
            output_path=work_directory,
        )

        print(f"BCR文件处理完成: {result}")
        return state.model_copy(
            update={"standardized_files": result, "bcr_input_valid": True}
        )

    except Exception as e:
        print(f"BCR文件处理失败: {str(e)}")
        return state.model_copy(update={"bcr_input_valid": False})


def select_metabcr_model_node(state: State, config: RunnableConfig) -> State:
    """
    用户输入节点：模型选择路径 - 上传数据文件并智能选择模型
    """

    # 智能模型选择提示词
    selection_model_prompt = ChatPromptTemplate.from_template(
        CellPrompt.SELECTION_MODEL_PROMPT
    )
    reasoning_model = get_reasoning_model(config)
    runnable = selection_model_prompt | reasoning_model | SimpleJsonOutputParser()

    try:
        response = runnable.invoke({"refine_plan": state.refine_plan})
        selected_model = response if isinstance(response, list) else []
        print(f"从计划中提取的模型: {selected_model}")
    except Exception as e:
        print(f"模型选择出错: {e}")
        selected_model = []
    state.selected_model = selected_model
    return state


def user_input_strategy_node(state: State, config: RunnableConfig) -> State:
    """
    用户输入节点：分析策略路径 - 上传数据文件并智能选择分析工具
    """
    strategy_info = {
        "task": "策略选择节点 - 执行R语言分析",
        "description": "请提供需要进行R语言分析的RDS数据文件",
    }

    # 等待用户上传文件
    uploaded_files_input = interrupt(strategy_info)

    # 验证输入是否为空
    is_valid_input = bool(uploaded_files_input and uploaded_files_input.strip())

    # 只有输入有效时才更新strategy_upload_files
    update_dict = {"strategy_input_valid": is_valid_input}
    if is_valid_input:
        update_dict["strategy_upload_files"] = uploaded_files_input
    return state.model_copy(update=update_dict)


def execute_metabcr_node(state: State, config: RunnableConfig) -> State:
    """
    智能模型选择节点 - 执行Meta-BCR模型预测
    """
    model_info = {
        "task": "智能模型选择节点 - 执行Meta-BCR模型预测",
        "description": "请提供需要进行Meta-BCR模型预测的数据文件，如果跳过此步骤请输入：skip",
    }

    # 等待用户上传文件
    uploaded_files_input = interrupt(model_info)

    # 验证输入是否为空
    if not uploaded_files_input or not uploaded_files_input.strip():
        return state.model_copy(update={"metabcr_input_valid": False})

    user_input = uploaded_files_input.strip()

    # 检查是否跳过
    if user_input.lower() == "skip":
        print("用户选择跳过Meta-BCR模型预测，跳转到文件整合节点")
        return state.model_copy(
            update={"metabcr_input_valid": True, "metabcr_skip": True}
        )

    reasoning_model = get_reasoning_model(config)
    print(f"调用Metabcr处理文件: {user_input}")
    try:
        react_agent = create_react_agent(reasoning_model, [metabcr_tool])

        # 调用ReAct Agent
        message = f"请分析这个CSV文件: {user_input}，输出结果请保存到{config['configurable']['work_directory']}目录下"

        result = react_agent.invoke({"messages": [HumanMessage(content=message)]})
        print(f"Metabcr处理结果: {result}")

        # 提取 ToolMessage 的 content (工具执行结果)
        metabcr_result_str = ""
        if isinstance(result, dict) and "messages" in result:
            for message in result["messages"]:
                if (
                    hasattr(message, "content")
                    and hasattr(message, "name")
                    and getattr(message, "name", None) == "metabcr_tool"
                ):
                    metabcr_result_str = message.content
                    break

        print(f"提取的MetaBCR结果: {metabcr_result_str}")

        # 返回字典格式的状态更新
        state.metabcr_result = metabcr_result_str
        state.metabcr_input_valid = False  # 设置为False，让其返回自身节点
        state.metabcr_skip = False
    except Exception as e:
        error_msg = f"处理出错: {str(e)}"
        print(f"调用Metabcr出错: {error_msg}")
        state.metabcr_input_valid = False  # 设置为False，让其返回自身节点
        state.metabcr_skip = False
    return state


def input_integrate_rds_bcr_node(state: State, config: RunnableConfig) -> State:
    """
    RDS和BCR文件整合输入节点 - 用户输入RDS文件和CSV文件路径
    """
    file_info = {
        "task": "RDS和BCR文件整合节点",
        "description": "请输入RDS文件和CSV文件路径，用逗号分隔（例如：/path/to/file.rds,/path/to/file.csv）。如果跳过此步骤，请输入：skip",
        "requirements": "第一个文件必须是RDS格式，第二个文件必须是CSV格式。",
    }

    # 等待用户输入文件路径
    user_input = interrupt(file_info)

    # 验证输入是否为空
    if not user_input or not user_input.strip():
        print("未提供文件路径")
        return state.model_copy(update={"rds_bcr_input_valid": False})

    user_input = user_input.strip()

    # 检查是否跳过
    if user_input.lower() == "skip":
        print("用户选择跳过RDS和BCR文件整合，跳转到策略选择节点")
        return state.model_copy(
            update={
                "rds_bcr_input_valid": True,
                "rds_file_path": "",
                "bcr_file_path": "",
            }
        )

    # 解析文件路径（用逗号分隔）
    file_paths = [path.strip() for path in user_input.split(",")]

    # 验证是否提供了两个文件路径
    if len(file_paths) != 2:
        print("错误：必须提供两个文件路径，用逗号分隔")
        return state.model_copy(update={"rds_bcr_input_valid": False})

    rds_path, csv_path = file_paths

    # 验证文件类型
    if not rds_path.lower().endswith(".rds"):
        print(f"错误：第一个文件必须是RDS格式，当前文件：{rds_path}")
        return state.model_copy(update={"rds_bcr_input_valid": False})

    if not csv_path.lower().endswith(".csv"):
        print(f"错误：第二个文件必须是CSV格式，当前文件：{csv_path}")
        return state.model_copy(update={"rds_bcr_input_valid": False})

    # 验证文件是否存在
    import os

    if not os.path.exists(rds_path):
        print(f"错误：RDS文件不存在：{rds_path}")
        return state.model_copy(update={"rds_bcr_input_valid": False})

    if not os.path.exists(csv_path):
        print(f"错误：CSV文件不存在：{csv_path}")
        return state.model_copy(update={"rds_bcr_input_valid": False})

    print(f"文件验证成功 - RDS文件：{rds_path}，CSV文件：{csv_path}")

    # 更新状态
    return state.model_copy(
        update={
            "rds_file_path": rds_path,
            "bcr_file_path": csv_path,
            "rds_bcr_input_valid": True,
        }
    )


def integrate_rds_bcr_node(state: State, config: RunnableConfig) -> State:
    """
    RDS和BCR文件整合节点 - 执行文件整合操作
    """
    print("=== 执行RDS和BCR文件整合 ===")

    # 从state中获取文件路径
    rds_file_path = state.rds_file_path
    bcr_file_path = state.bcr_file_path

    if not rds_file_path or not bcr_file_path:
        print("错误：缺少必要的文件路径")
        return state.model_copy(update={"rds_bcr_input_valid": False})

    print(f"RDS文件路径: {rds_file_path}")
    print(f"BCR文件路径: {bcr_file_path}")

    try:
        work_directory = config["configurable"]["work_directory"]
        # 调用整合工具
        from tool.rds_tools import integrate_rds_bcr_data_tool

        result = integrate_rds_bcr_data_tool(
            bcr_file_path, rds_file_path, work_directory
        )
        print(f"整合结果: {result}")

        # 整合完成，重置输入状态以便返回到输入节点
        return state.model_copy(
            update={
                "rds_bcr_input_valid": False,  # 重置为False，返回到输入节点
                "rds_file_path": "",
                "bcr_file_path": "",
            }
        )

    except Exception as e:
        print(f"整合过程中发生错误: {str(e)}")
        return state.model_copy(update={"rds_bcr_input_valid": False})


def strategy_selection_node(state: State, config: RunnableConfig) -> State:
    """
    策略选择节点 - 使用ReAct Agent执行R语言分析
    """
    print("=== 执行分析策略 ===")

    strategy_info = {
        "task": "策略选择节点 - 执行R语言分析",
        "description": "请根据提供的RDS数据文件，选择合适的分析工具执行分析。输入'skip'结束分析。",
    }

    # 等待用户输入
    user_prompt = interrupt(strategy_info)

    # 检查用户是否输入skip
    if user_prompt and user_prompt.lower().strip() == "skip":
        state.should_end = True
        return state

    reasoning_model = get_reasoning_model(config)
    try:
        react_agent = create_react_agent(reasoning_model, figure_analysis_tools)

        # 调用ReAct Agent
        message = f"""作为专业的单细胞数据分析专家，请对以下RDS文件进行深入分析：

## 输入文件：
{state.strategy_upload_files}
## 输出目录：
{config["configurable"]["work_directory"]}

## 分析任务：
{user_prompt}

## 请确保：
1. 仔细检查数据质量和完整性
2. 生成高质量的可视化图表
3. 提供详细的分析结果和解释
4. 将所有输出文件保存到指定目录
5. 如遇到问题，请提供清晰的错误信息和建议解决方案"""

        result = react_agent.invoke({"messages": [HumanMessage(content=message)]})

        print(f"R分析完成: {result}")

    except Exception as e:
        print(f"执行R分析时出错: {e}")

    return state


def should_continue_refine(state: State) -> str:
    """判断是否需要继续精化计划"""
    if state.plan_confirmed == True:
        return "continue"
    else:
        return "refine"


def should_retry_strategy_input(state: State) -> str:
    """判断是否需要重新输入策略文件"""
    if state.strategy_input_valid:
        return "continue"
    else:
        return "retry"


def should_retry_metabcr_input(state: State) -> str:
    """判断MetaBCR节点的流向"""
    if hasattr(state, "metabcr_skip") and state.metabcr_skip:
        return "skip_to_integrate"  # 用户选择跳过，流转到文件整合节点
    else:
        return "retry"  # 成功或失败都返回自身节点


def should_skip_or_continue_standardization(state: State) -> str:
    """判断是否跳过标准化、继续标准化或重新输入"""
    if state.standardize_status == "failed":
        return "retry_input"  # 输入验证失败，重新输入
    elif state.standardize_status in ["skipped", "completed"]:
        return "skip"  # 已跳过或已完成，跳转到下一节点
    elif state.standardize_status == "processing":
        return "standardize"  # 需要执行标准化
    elif state.standardize_status == "bcr_processing":
        return "bcr_input"  # 跳转到BCR节点
    else:
        return "retry_input"  # 默认重新输入


def should_retry_file_standardization(state: State) -> str:
    """判断文件标准化是否需要重试"""
    if state.standardize_status == "completed":
        return "back_to_input"  # 标准化完成，回到输入节点
    else:
        return "retry"  # 标准化失败，重试


def should_retry_bcr_input(state: State) -> str:
    """
    判断BCR文件输入是否有效
    """
    if state.bcr_input_valid and state.bcr_skip:
        return "skip_to_model"  # 跳过BCR处理，直接到模型选择
    elif state.bcr_input_valid:
        return "continue"  # 继续BCR文件标准化
    else:
        return "retry"  # 重新输入


def should_continue_from_bcr_standard(state: State) -> str:
    """
    判断BCR标准化节点的流转
    """
    if state.bcr_input_valid and state.standardized_files:
        return "back_to_bcr_input"  # 处理完成，回到BCR输入
    else:
        return "retry"  # 处理失败，重试


def should_continue_from_integrate_input(state: State) -> str:
    """
    判断RDS和BCR文件整合输入节点的流转
    使用单一变量rds_bcr_input_valid控制流转：
    - True且有文件路径：流转到整合节点
    - True且无文件路径：用户选择跳过，流转到策略选择
    - False：输入无效，重新输入
    """
    if not state.rds_bcr_input_valid:
        return "retry"  # 输入无效，重新输入
    elif state.rds_file_path and state.bcr_file_path:
        return "integrate"  # 文件有效，流转到整合节点
    else:
        return "skip_to_strategy"  # 用户选择跳过，流转到策略选择节点


def should_continue_strategy_selection(state: State) -> str:
    """
    判断策略选择节点是否应该结束
    """
    if state.should_end:
        return "end"
    else:
        return "continue"


def create_planning_graph():
    """
    创建完整的规划图
    """
    # 创建内存检查点保存器
    memory = MemorySaver()

    # 创建检索子图
    retrieval_subgraph = create_parallel_rag_graph()

    # 创建StateGraph
    workflow = StateGraph(State)
    # 添加节点
    workflow.add_node("retrieval", retrieval_subgraph)
    workflow.add_node("refine_planning", refine_planning_node)
    workflow.add_node("input_standardize_file", input_standardize_file_node)
    workflow.add_node("standardize_file", standardize_file_node)
    workflow.add_node("input_bcr_file", input_bcr_file_node)
    workflow.add_node("bcr_file_standard", bcr_file_standard_node)
    workflow.add_node("select_metabcr_model", select_metabcr_model_node)
    workflow.add_node("input_integrate_rds_bcr", input_integrate_rds_bcr_node)
    workflow.add_node("integrate_rds_bcr", integrate_rds_bcr_node)
    workflow.add_node("user_input_strategy", user_input_strategy_node)
    workflow.add_node("execute_metabcr", execute_metabcr_node)
    workflow.add_node("strategy_selection", strategy_selection_node)

    # 设置入口点：从检索开始
    workflow.add_edge(START, "retrieval")

    # 从检索直接流转到refine_planning进行计划确认
    workflow.add_edge("retrieval", "refine_planning")

    # 从refine_planning添加条件边
    workflow.add_conditional_edges(
        "refine_planning",
        should_continue_refine,
        {"continue": "input_standardize_file", "refine": "refine_planning"},
    )

    # 从input_standardize_file添加条件边处理文件输入和标准化流转
    workflow.add_conditional_edges(
        "input_standardize_file",
        should_skip_or_continue_standardization,
        {
            "skip": "select_metabcr_model",  # 跳过标准化，直接到下一节点
            "standardize": "standardize_file",  # 执行标准化
            "bcr_input": "input_bcr_file",  # 跳转到BCR文件输入
            "retry_input": "input_standardize_file",  # 重新输入
        },
    )

    # 从standardize_file添加条件边处理标准化结果
    workflow.add_conditional_edges(
        "standardize_file",
        should_retry_file_standardization,
        {
            "back_to_input": "input_standardize_file",  # 标准化完成，回到输入节点
            "retry": "standardize_file",  # 标准化失败，重试
        },
    )

    # 从input_bcr_file添加条件边处理BCR文件输入
    workflow.add_conditional_edges(
        "input_bcr_file",
        should_retry_bcr_input,
        {
            "continue": "bcr_file_standard",  # BCR文件有效，进入标准化
            "skip_to_model": "select_metabcr_model",  # 跳过BCR处理，直接到模型选择
            "retry": "input_bcr_file",  # BCR文件无效，重新输入
        },
    )

    # 从bcr_file_standard添加条件边处理BCR标准化结果
    workflow.add_conditional_edges(
        "bcr_file_standard",
        should_continue_from_bcr_standard,
        {
            "back_to_bcr_input": "input_bcr_file",  # 处理完成，回到BCR输入节点
            "retry": "bcr_file_standard",  # 处理失败，重试
        },
    )

    # 添加边：模型选择路径
    workflow.add_edge("select_metabcr_model", "execute_metabcr")

    # 从execute_metabcr添加条件边处理MetaBCR流向
    workflow.add_conditional_edges(
        "execute_metabcr",
        should_retry_metabcr_input,
        {
            "skip_to_integrate": "input_integrate_rds_bcr",  # 用户选择跳过，流转到文件整合节点
            "retry": "execute_metabcr",  # 成功或失败都返回自身节点
        },
    )

    # 从input_integrate_rds_bcr添加条件边处理RDS和BCR文件整合
    workflow.add_conditional_edges(
        "input_integrate_rds_bcr",
        should_continue_from_integrate_input,
        {
            "integrate": "integrate_rds_bcr",  # 文件有效，流转到整合节点
            "skip_to_strategy": "strategy_selection",  # 跳过，直接到策略选择
            "retry": "input_integrate_rds_bcr",  # 重新输入
        },
    )

    # 从integrate_rds_bcr返回到input_integrate_rds_bcr
    workflow.add_edge("integrate_rds_bcr", "input_integrate_rds_bcr")

    # 添加边：策略分析路径 - 使用条件边处理重新输入
    workflow.add_conditional_edges(
        "user_input_strategy",
        should_retry_strategy_input,
        {"continue": "strategy_selection", "retry": "user_input_strategy"},
    )

    # 添加条件边：策略选择节点的流转逻辑
    workflow.add_conditional_edges(
        "strategy_selection",
        should_continue_strategy_selection,
        {"continue": "strategy_selection", "end": END},
    )

    # 编译图
    app = workflow.compile(checkpointer=memory)
    try:
        print("\n===== 工作流程图代码 =====")
        print("可以将以下代码复制到任意Mermaid编辑器中查看图形:")
        print(app.get_graph().draw_mermaid())
    except Exception as e:
        print(f"生成Mermaid代码时出错: {str(e)}")
    return app


def run_planning_graph(user_question, config: RunnableConfig):
    """
    运行规划图的示例函数
    """
    # 创建图
    app = create_planning_graph()
    # 初始状态
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
    # 配置
    _uuid = uuid.uuid4()
    work_directory = os.path.join(STANDARDIZED_WORKING_DIRECTORY, str(_uuid))

    os.makedirs(work_directory, exist_ok=True)
    config = get_cell_runnable_config(_uuid, work_directory)

    # 先运行到第一次中断，支持subgraph
    _ = list(app.stream(initial_state, config, subgraphs=True))

    # 持续运行直到工作流完成
    while True:
        # 检查工作流状态
        state = app.get_state(config)

        # 如果工作流已完成（没有下一个节点），退出循环
        if not state.next:
            break

        # 如果有下一个节点，说明需要继续执行（可能被中断）
        print(state.interrupts[0].value)
        user_input = input("> ").strip()

        # 使用用户输入恢复工作流，支持subgraph
        _ = list(app.stream(Command(resume=user_input), config, subgraphs=True))

    # 获取最终状态
    final_state = app.get_state(config).values
    print(f"\n=== 工作流执行完成 ===")
    print(f"最终状态: {final_state}")
