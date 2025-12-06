#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure4相关工具测试文件

测试生信分析模块化MCP服务器中的Figure4相关工具
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bioinformatics_mcp_server import (
    pseudotime_trajectory_analysis,
    pseudotime_celltype_boxplot_analysis,
    trajectory_polynomial_regression_analysis,
    trajectory_supplementary_analysis
)

# 测试参数配置
INPUT_RDS_FILE = r"D:\data\test_data_20251001\Age_Bcells.rds"
OUTPUT_DIR = r"D:\data\test_data_20251001"

def test_pseudotime_trajectory_analysis():
    """测试单细胞B细胞伪时间轨迹分析和UMAP可视化"""
    print("测试: 单细胞B细胞伪时间轨迹分析和UMAP可视化")
    result = pseudotime_trajectory_analysis(
        input_file=INPUT_RDS_FILE,
        base_dir=OUTPUT_DIR,
        num_dim=50,
        cluster_resolution=0.001,
        min_gene_cells=3,
        root_celltype="Naive"
    )
    print(f"结果: {result}")
    print("-" * 50)

def test_pseudotime_celltype_boxplot_analysis():
    """测试伪时间与细胞类型分布箱线图分析"""
    print("测试: 伪时间与细胞类型分布箱线图分析")
    result = pseudotime_celltype_boxplot_analysis(
        input_file=INPUT_RDS_FILE,
        base_dir=OUTPUT_DIR,
        celltype_column=""  # 空值表示自动检测
    )
    print(f"结果: {result}")
    print("-" * 50)

def test_trajectory_polynomial_regression_analysis():
    """测试轨迹多项式回归分析和特征基因模块评分"""
    print("测试: 轨迹多项式回归分析和特征基因模块评分")
    result = trajectory_polynomial_regression_analysis(
        input_file=INPUT_RDS_FILE,
        base_dir=OUTPUT_DIR
    )
    print(f"结果: {result}")
    print("-" * 50)

def test_trajectory_supplementary_analysis():
    """测试轨迹分析补充图形生成和转录标记分析"""
    print("测试: 轨迹分析补充图形生成和转录标记分析")
    result = trajectory_supplementary_analysis(
        input_file=INPUT_RDS_FILE,
        base_dir=OUTPUT_DIR
    )
    print(f"结果: {result}")
    print("-" * 50)

def run_all_figure4_tests():
    """运行所有Figure4相关工具测试"""
    print("=" * 60)
    print("开始运行Figure4相关工具测试")
    print("=" * 60)
    
    # 注意：Figure4的分析有依赖关系，建议按顺序执行
    print("注意: Figure4分析建议按顺序执行，轨迹分析需要先运行")
    
    # 首先运行轨迹分析，生成CDS对象
    test_pseudotime_trajectory_analysis()
    
    # 然后运行依赖轨迹分析结果的其他分析
    test_pseudotime_celltype_boxplot_analysis()
    test_trajectory_polynomial_regression_analysis()
    test_trajectory_supplementary_analysis()
    
    print("=" * 60)
    print("Figure4相关工具测试完成")
    print("=" * 60)

if __name__ == "__main__":
    run_all_figure4_tests()