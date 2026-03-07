# -*- coding: utf-8 -*-
"""
IterativeOpenCodeExecutor 使用示例

这个示例展示如何使用迭代式执行器完成任务。
"""

import asyncio
import os
from datetime import datetime

# 添加项目路径
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))

from coding_agent import (
    IterativeOpenCodeExecutor,
    IterativeOpenCodeExecutorSync,
    OpenCodeConfig,
    IterationStatus,
)


async def demo_async():
    """异步执行示例"""
    print("=" * 70)
    print("IterativeOpenCodeExecutor 异步执行示例")
    print("=" * 70)
    
    # 从环境变量加载配置
    config = OpenCodeConfig.from_env()
    
    # 创建执行器
    executor = IterativeOpenCodeExecutor(
        config=config,
        max_iterations=3,  # 最多迭代 3 次
        early_stop_on_success=True,  # 成功时提前退出
    )
    
    # 准备输入数据
    input_data = {
        "session_id": f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "input_files": [
            # 可以添加远程数据文件路径
            # "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv",
        ],
        "params": {
            "threshold": 0.6,
            "model": "xgboost",
        },
        "user_input": "执行 TCR 抗体结合预测分析",
        "task_type": "tcr_binding_prediction",
        "mcp_tools": ["nettcr"],
    }
    
    print(f"\n会话 ID: {input_data['session_id']}")
    print(f"最大迭代次数: {executor.max_iterations}")
    
    # 执行
    print("\n开始执行...")
    result = await executor.execute(input_data)
    
    # 打印结果
    print("\n" + "=" * 70)
    print("执行结果")
    print("=" * 70)
    print(f"总迭代次数: {result.total_iterations}")
    print(f"最终状态: {result.final_status.value}")
    print(f"总执行时间: {result.total_execution_time_ms}ms")
    print(f"输出目录: {result.final_output_dir}")
    print(f"输出文件数: {len(result.final_output_files)}")
    
    if result.final_output_files:
        print("\n输出文件:")
        for f in result.final_output_files:
            print(f"  - {f}")
    
    # 打印迭代历史
    print("\n迭代历史:")
    for iter_result in result.iteration_history:
        print(f"  迭代 {iter_result.iteration}:")
        print(f"    状态: {iter_result.status.value}")
        print(f"    质量分数: {iter_result.quality_score:.2f}")
        print(f"    执行时间: {iter_result.execution_time_ms}ms")
    
    # 检查是否成功
    if result.is_success():
        print("\n✅ 执行成功!")
    else:
        print("\n❌ 执行失败")
        if result.final_summary.get("error"):
            print(f"错误: {result.final_summary['error']}")


def demo_sync():
    """同步执行示例"""
    print("\n" + "=" * 70)
    print("IterativeOpenCodeExecutorSync 同步执行示例")
    print("=" * 70)
    
    # 从环境变量加载配置
    config = OpenCodeConfig.from_env()
    
    # 创建同步执行器
    executor = IterativeOpenCodeExecutorSync(
        config=config,
        max_iterations=2,  # 最多迭代 2 次
    )
    
    # 准备输入数据
    input_data = {
        "session_id": f"sync_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "user_input": "简单测试任务",
        "params": {"test": True},
    }
    
    print(f"\n会话 ID: {input_data['session_id']}")
    
    # 执行（同步方式）
    print("\n开始执行（同步）...")
    result = executor.execute(input_data)
    
    # 打印结果
    print(f"\n执行完成: {result.final_status.value}")
    print(f"总迭代次数: {result.total_iterations}")


def demo_custom_criteria():
    """自定义评估标准示例"""
    from coding_agent import EvaluationCriteria
    
    print("\n" + "=" * 70)
    print("自定义评估标准示例")
    print("=" * 70)
    
    # 创建自定义评估标准
    criteria = EvaluationCriteria(
        required_output_files=[
            "final_predictions.csv",
            "evaluation_report.json",
        ],
        min_quality_score=0.7,  # 更高的质量阈值
        early_stop_on_success=True,
    )
    
    # 创建执行器
    config = OpenCodeConfig.from_env()
    executor = IterativeOpenCodeExecutor(
        config=config,
        max_iterations=5,
        evaluation_criteria=criteria,
    )
    
    print(f"必需文件: {criteria.required_output_files}")
    print(f"最低质量分数: {criteria.min_quality_score}")
    
    # ... 执行代码省略 ...


def main():
    """主函数"""
    print("\n请选择示例:")
    print("  1. 异步执行示例")
    print("  2. 同步执行示例")
    print("  3. 自定义评估标准示例")
    
    try:
        choice = input("\n请输入选择 (1/2/3, 默认 1): ").strip() or "1"
    except EOFError:
        choice = "1"
    
    if choice == "2":
        demo_sync()
    elif choice == "3":
        demo_custom_criteria()
    else:
        asyncio.run(demo_async())


if __name__ == "__main__":
    main()

