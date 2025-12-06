import asyncio
from typing import Annotated

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from common.factory import get_default_model
from common.prompts import AntibodyPrompt
from common.util.mcp_utils import mcp_tool_async
from common.util.retrieval_utils import remove_think_tags
from usecases.antibody.state.state import PlanState


def metabcr_tool(state: PlanState):
    """异步执行MetaBCR工具函数"""
    input_file = state.csv_path
    return asyncio.run(
        mcp_tool_async("metabcr", "metabcr", {"input_file_path": input_file})
    )


@tool
def metabcr_analysis_tool(input_file_path: Annotated[str, "文件的完整路径"]) -> str:
    """使用MetaBCR工具分析抗体数据。需要提供文件的完整路径。"""
    try:
        result = asyncio.run(
            mcp_tool_async("metabcr", "metabcr", {"input_file_path": input_file_path})
        )
        return f"MetaBCR分析完成: {result}"
    except Exception as e:
        return f"MetaBCR分析失败: {str(e)}"


@tool
def fdg_analysis_tool(input_file_path: Annotated[str, "文件的完整路径"]) -> str:
    """使用FDG工具进行蛋白质稳定性分析。需要提供CSV文件的完整路径。"""
    try:
        result = asyncio.run(
            mcp_tool_async("fdg", "fdg", {"input_file_path": input_file_path})
        )
        return f"FDG分析完成: {result}"
    except Exception as e:
        return f"FDG分析失败: {str(e)}"


@tool
def alphafold3_analysis_tool(
    input_file_path: Annotated[str, "文件的完整路径"],
    antigen_name: Annotated[str, "抗原名称"] = None,
) -> str:
    """使用AlphaFold3进行蛋白质结构预测。需要提供文件的完整路径。"""
    try:
        result = asyncio.run(
            mcp_tool_async(
                "af3",
                "alphafold3",
                {"input_file_path": input_file_path, "antigen_name": antigen_name},
            )
        )
        return f"AlphaFold3分析完成: {result}"
    except Exception as e:
        return f"AlphaFold3分析失败: {str(e)}"


def refine_plan_tool(state: PlanState, config: RunnableConfig):
    """使用模型优化计划的节点"""
    refine_plan_prompt = ChatPromptTemplate.from_template(
        AntibodyPrompt.PLAN_EVALUATION_PROMPT
    )
    model = get_default_model(config)
    runnable = refine_plan_prompt | model | StrOutputParser() | remove_think_tags
    refined_plan = runnable.invoke(
        {"plan": state.generated_plan, "context": state.context}
    )

    print(f"\n===== 优化后的研究计划 =====\n{refined_plan}")
    return refined_plan


# 用于PlanState的工具节点（保持原有逻辑）
tool_node = ToolNode([metabcr_tool, refine_plan_tool])

# 用于ReAct Agent的分析工具（符合LangChain工具规范）
analysis_tool_node = [
    metabcr_analysis_tool,
    fdg_analysis_tool,
    alphafold3_analysis_tool,
]
