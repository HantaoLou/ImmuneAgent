import asyncio

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from common.util.mcp_utils import mcp_tool_async


@tool
def metabcr_tool(input_file: str, output_file: str):
    """异步执行MetaBCR工具函数 - 适配0.1.7版本API"""
    return asyncio.run(
        mcp_tool_async(
            "metabcr",
            "metabcr",
            {"input_file_path": input_file, "output_file_path": output_file},
        )
    )


@tool
def figure2_analysis_tool(input_file: str, base_dir: str):
    """执行Figure2 RSV分析"""
    return asyncio.run(
        mcp_tool_async(
            "r_analysis",
            "run_figure2_analysis",
            {"input_file": input_file, "base_dir": base_dir},
        )
    )


@tool
def figure3_analysis_tool(input_file: str, base_dir: str):
    """执行Figure3 RSV分析"""
    return asyncio.run(
        mcp_tool_async(
            "r_analysis",
            "run_figure3_analysis",
            {"input_file": input_file, "base_dir": base_dir},
        )
    )


@tool
def figure4_analysis_tool(input_file: str, base_dir: str):
    """执行Figure4 RSV分析（轨迹分析，耗时较长）"""
    return asyncio.run(
        mcp_tool_async(
            "r_analysis",
            "run_figure4_analysis",
            {"input_file": input_file, "base_dir": base_dir},
        )
    )


@tool
def figure5_analysis_tool(input_file: str, base_dir: str):
    """执行Figure5 RSV分析"""
    return asyncio.run(
        mcp_tool_async(
            "r_analysis",
            "run_figure5_analysis",
            {"input_file": input_file, "base_dir": base_dir},
        )
    )


@tool
def all_figures_analysis_tool(input_file: str, base_dir: str):
    """执行所有Figure分析（包括Figure2-5的完整分析）"""
    return asyncio.run(
        mcp_tool_async(
            "r_analysis",
            "run_all_figures_analysis",
            {"input_file": input_file, "base_dir": base_dir},
        )
    )


# 工具列表
figure_analysis_tools = [
    figure2_analysis_tool,
    figure3_analysis_tool,
    figure4_analysis_tool,
    figure5_analysis_tool,
    all_figures_analysis_tool,
]
