import asyncio
from typing import List

from langchain_core.tools import tool

from common.util.mcp_utils import mcp_tool_async


# @tool
def bcr_standardize_tool(
    bcr_file_path: str, combine_fields: List[str], output_path: str
):
    """
    标准化BCR文件 - 组合字段生成combine_barcode

    Args:
        bcr_file_path (str): BCR文件路径
        combine_fields (List[str]): 需要组合的字段名列表
        output_path (str): 输出文件路径


    Returns:
        str: 输出文件路径
    """
    params = {
        "bcr_file_path": bcr_file_path,
        "combine_fields": combine_fields,
        "output_path": output_path,
    }
    return asyncio.run(mcp_tool_async("r_standardize", "run_bcr_standardize", params))


# @tool
def rds_standardize_tool(
    rds_file_path: str, combine_fields: List[str], output_path: str
):
    """
    标准化RDS文件 - 组合字段生成combine_barcode

    Args:
        rds_file_path (str): RDS文件路径
        combine_fields (List[str]): 需要组合的字段名列表


    Returns:
        str: 执行结果信息，包含成功消息或错误信息
    """
    params = {
        "rds_file_path": rds_file_path,
        "combine_fields": combine_fields,
        "output_path": output_path,
    }
    return asyncio.run(mcp_tool_async("r_standardize", "run_rds_standardize", params))


# @tool
def extract_bcr_info_tool(bcr_file_path: str):
    """
    从BCR文件中提取包含条形码信息的字段名、重链序列信息的字段名、轻链序列信息的字段名

    Args:
        bcr_file_path (str): BCR文件路径

    Returns:
        str: 执行结果信息，成功时包含提取的字段信息（bar_code、Heavy、Light字段列表），失败时包含错误信息
    """
    params = {"bcr_file_path": bcr_file_path}
    return asyncio.run(mcp_tool_async("r_standardize", "run_extract_bcr_info", params))


# @tool
def process_csv_to_standard_tool(
    csv_file_path: str,
    bar_code: str,
    heavy: str,
    light: str,
    variant_seq: str,
    experiment: str,
    output_path: str = None,
):
    """
    处理CSV文件，提取指定字段并生成标准格式的新CSV文件

    Args:
        csv_file_path: 输入CSV文件路径
        bar_code: 条形码字段名
        heavy: 重链字段名
        light: 轻链字段名
        variant_seq: 变异序列值（固定值）
        experiment: 实验值（固定值）
        output_path: 输出文件路径，如果为None则自动生成

    Returns:
        输出文件路径
    """
    params = {
        "csv_file_path": csv_file_path,
        "bar_code": bar_code,
        "heavy": heavy,
        "light": light,
        "variant_seq": variant_seq,
        "experiment": experiment,
        "output_path": output_path,
    }
    return asyncio.run(
        mcp_tool_async("r_standardize", "run_process_csv_to_standard", params)
    )


# @tool
def integrate_rds_bcr_data_tool(
    bcr_file_path: str, rds_file_path: str, output_path: str
):
    """
    整合BCR文件和RDS文件

    Args:
        bcr_file_path (str): BCR文件路径
        rds_file_path (str): RDS文件路径
        output_path (str): 输出文件路径


    Returns:
        str: 执行结果信息，包含成功消息或错误信息
    """
    params = {
        "bcr_file_path": bcr_file_path,
        "rds_file_path": rds_file_path,
        "output_path": output_path,
    }
    return asyncio.run(
        mcp_tool_async("r_standardize", "run_integrate_rds_bcr_data", params)
    )


# 工具列表
data_tools = [
    bcr_standardize_tool,
    rds_standardize_tool,
    extract_bcr_info_tool,
    process_csv_to_standard_tool,
    integrate_rds_bcr_data_tool,
]
