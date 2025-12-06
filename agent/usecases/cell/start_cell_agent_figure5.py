# 测试Figure5 RSV分析工具
import asyncio
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# 直接导入mcp_tool_async避免工具包装器的事件循环问题
from common.util.mcp_utils import mcp_tool_async


async def test_figure5():
    """测试Figure5 RSV分析工具"""
    # 使用项目中的测试数据文件
    # test_file = "D:\\PartTimeJob\\antibody_gen\\mcp_r\\data\\rds\\King_Tonsil_RSVbinding0225_vclean.rds"  # 需要替换为实际的RDS文件路径
    test_file = "D:\\data\\test_data_20251001\\fluBcells.rds"

    # 设置输出目录 - 在当前项目目录下创建输出文件夹
    base_dir = "D:\\data\\test_output_figure5"

    print(f"测试Figure5 RSV分析工具")
    print(f"输入文件: {test_file}")
    print(f"输出目录: {base_dir}")

    # 确保输出目录存在
    os.makedirs(base_dir, exist_ok=True)

    # 直接调用MCP工具避免事件循环冲突
    try:
        print("正在调用Figure5分析工具...")
        result = await mcp_tool_async(
            "r_analysis",
            "run_figure5_analysis",
            {"input_file": test_file, "base_dir": base_dir},
        )
        print(f"分析完成！结果: {result}")
    except Exception as e:
        print(f"调用MCP工具时出错: {e}")
        result = None

    return result


if __name__ == "__main__":
    asyncio.run(test_figure5())
