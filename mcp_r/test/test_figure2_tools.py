#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure2相关工具测试文件

测试生信分析模块化MCP服务器中的Figure2相关工具
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bioinformatics_mcp_server import (
    antigen_binding_prediction_visualization,
    bcell_celltype_distribution_analysis,
    binding_prediction_interval_distribution_analysis,
    differential_gene_expression_volcano_analysis,
    umap_dimensionality_reduction_visualization,
    bcell_marker_gene_dotplot_analysis
)

# 测试参数配置
INPUT_RDS_FILE = r"D:\data\test_data_20251001\Age_Bcells.rds"
OUTPUT_DIR = r"D:\data\test_data_20251001"

def test_antigen_binding_prediction_visualization():
    """测试单细胞B细胞抗原结合预测可视化分析"""
    print("测试: 单细胞B细胞抗原结合预测可视化分析")
    result = antigen_binding_prediction_visualization(
        input_file=INPUT_RDS_FILE,
        base_dir=OUTPUT_DIR,
        binding_threshold=0.5
    )
    print(f"结果: {result}")
    print("-" * 50)

def test_bcell_celltype_distribution_analysis():
    """测试单细胞B细胞亚群类型分布可视化分析"""
    print("测试: 单细胞B细胞亚群类型分布可视化分析")
    result = bcell_celltype_distribution_analysis(
        input_file=INPUT_RDS_FILE,
        base_dir=OUTPUT_DIR
    )
    print(f"结果: {result}")
    print("-" * 50)

def test_binding_prediction_interval_distribution_analysis():
    """测试单细胞抗原结合预测值区间分布统计分析"""
    print("测试: 单细胞抗原结合预测值区间分布统计分析")
    result = binding_prediction_interval_distribution_analysis(
        input_file=INPUT_RDS_FILE,
        base_dir=OUTPUT_DIR,
        interval_step=0.1,
        data_min=0.0,
        data_max=1.0
    )
    print(f"结果: {result}")
    print("-" * 50)

def test_differential_gene_expression_volcano_analysis():
    """测试单细胞差异表达基因分析和火山图可视化"""
    print("测试: 单细胞差异表达基因分析和火山图可视化")
    result = differential_gene_expression_volcano_analysis(
        input_file=INPUT_RDS_FILE,
        base_dir=OUTPUT_DIR,
        logfc_threshold=0.25,
        min_pct=0.2,
        analysis_strategy="both"
    )
    print(f"结果: {result}")
    print("-" * 50)

def test_umap_dimensionality_reduction_visualization():
    """测试单细胞B细胞UMAP降维和细胞亚群可视化分析"""
    print("测试: 单细胞B细胞UMAP降维和细胞亚群可视化分析")
    result = umap_dimensionality_reduction_visualization(
        input_file=INPUT_RDS_FILE,
        base_dir=OUTPUT_DIR
    )
    print(f"结果: {result}")
    print("-" * 50)

def test_bcell_marker_gene_dotplot_analysis():
    """测试B细胞亚群特异性标记基因表达点图分析"""
    print("测试: B细胞亚群特异性标记基因表达点图分析")
    result = bcell_marker_gene_dotplot_analysis(
        input_file=INPUT_RDS_FILE,
        base_dir=OUTPUT_DIR,
        min_pct=0.1,
        min_expression=0.25
    )
    print(f"结果: {result}")
    print("-" * 50)

def run_all_figure2_tests():
    """运行所有Figure2相关工具测试"""
    print("=" * 60)
    print("开始运行Figure2相关工具测试")
    print("=" * 60)
    
    test_antigen_binding_prediction_visualization()
    test_bcell_celltype_distribution_analysis()
    test_binding_prediction_interval_distribution_analysis()
    test_differential_gene_expression_volcano_analysis()
    test_umap_dimensionality_reduction_visualization()
    test_bcell_marker_gene_dotplot_analysis()
    
    print("=" * 60)
    print("Figure2相关工具测试完成")
    print("=" * 60)

if __name__ == "__main__":
    run_all_figure2_tests()