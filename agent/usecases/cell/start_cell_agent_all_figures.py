# 测试所有Figure RSV分析工具
import asyncio
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from usecases.cell.tool.planning_tools import all_figures_analysis_tool_async


async def test_all_figures():
    """测试所有Figure RSV分析工具（2-5）"""
    # 使用项目中的测试数据文件
    test_file = "/path/to/rsv_data.rds"  # 需要替换为实际的RDS文件路径

    print(f"测试所有Figure RSV分析工具，文件: {test_file}")
    print("注意：包含Figure4轨迹分析，可能需要较长时间")

    result = await all_figures_analysis_tool_async(test_file)
    print(f"结果: {result}")

    return result


if __name__ == "__main__":
    asyncio.run(test_all_figures())
