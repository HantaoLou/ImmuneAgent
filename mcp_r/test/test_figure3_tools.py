#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure3相关工具测试文件

测试生信分析模块化MCP服务器中的Figure3相关工具
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bioinformatics_mcp_server import (
    antigen_binding_neutralization_density_visualization,
    bcell_celltype_umap_visualization,
    bcell_marker_gene_expression_dotplot,
    differential_gene_correlation_analysis,
    prediction_value_density_visualization
)

# 测试参数配置
INPUT_RDS_FILE = r"D:\data\test_data_20251001\Age_Bcells.rds"
OUTPUT_DIR = r"D:\data\test_data_20251001"

def test_antigen_binding_neutralization_density_visualization():
    """测试单细胞抗原结合和中和预测密度图可视化分析"""
    print("测试: 单细胞抗原结合和中和预测密度图可视化分析")
    result = antigen_binding_neutralization_density_visualization(
        input_file=INPUT_RDS_FILE,
        base_dir=OUTPUT_DIR,
        prediction_keywords="neut,bind,average,predict,output",
        na_strategy="exclude_cells",
        feature_priority="neutralization_first"
    )
    print(f"结果: {result}")
    print("-" * 50)

def test_bcell_celltype_umap_visualization():
    """测试单细胞B细胞类型UMAP空间分布可视化分析"""
    print("测试: 单细胞B细胞类型UMAP空间分布可视化分析")
    result = bcell_celltype_umap_visualization(
        input_file=INPUT_RDS_FILE,
        base_dir=OUTPUT_DIR,
        celltype_column="CellType"
    )
    print(f"结果: {result}")
    print("-" * 50)

def test_bcell_marker_gene_expression_dotplot():
    """测试B细胞亚群标记基因表达模式点图可视化分析"""
    print("测试: B细胞亚群标记基因表达模式点图可视化分析")
    result = bcell_marker_gene_expression_dotplot(
        input_file=INPUT_RDS_FILE,
        base_dir=OUTPUT_DIR,
        celltype_column="CellType"
    )
    print(f"结果: {result}")
    print("-" * 50)

def test_differential_gene_correlation_analysis():
    """测试差异表达基因相关性分析和散点图可视化"""
    print("测试: 差异表达基因相关性分析和散点图可视化")
    
    # 首先检查是否有Figure2差异表达分析的结果文件
    deg_file1 = os.path.join(OUTPUT_DIR, "output", "Figure2", "files", "DEG_broad_control.csv")
    deg_file2 = os.path.join(OUTPUT_DIR, "output", "Figure2", "files", "DEG_specific_control.csv")
    
    # 如果文件不存在，先运行差异表达分析
    if not (os.path.exists(deg_file1) and os.path.exists(deg_file2)):
        print("DEG文件不存在，先运行差异表达分析生成DEG文件...")
        
        # 导入Figure2的差异表达分析函数
        from bioinformatics_mcp_server import differential_gene_expression_volcano_analysis
        
        # 运行差异表达分析
        deg_result = differential_gene_expression_volcano_analysis(
            input_file=INPUT_RDS_FILE,
            base_dir=OUTPUT_DIR,
            logfc_threshold=0.25,
            min_pct=0.2,
            analysis_strategy="both"
        )
        print(f"差异表达分析结果: {deg_result}")
        
        # 再次检查文件是否生成
        if not (os.path.exists(deg_file1) and os.path.exists(deg_file2)):
            print("跳过测试: 差异表达分析未能生成所需的DEG文件")
            print("-" * 50)
            return
    
    # 运行相关性分析
    result = differential_gene_correlation_analysis(
        deg_file1=deg_file1,
        deg_file2=deg_file2,
        base_dir=OUTPUT_DIR,
        dataset1_name="Broad_Reactive",
        dataset2_name="Specific_Reactive",
        p_value_threshold=0.05,
        min_common_genes=10,
        highlight_genes="ITGAX,FGR,FCRL4,FCRL5,CD68,TNFRSF1B,JCHAIN,MZB1,XBP1,MARCKSL1"
    )
    print(f"结果: {result}")
    print("-" * 50)

def test_prediction_value_density_visualization():
    """测试预测值UMAP密度图可视化分析"""
    print("测试: 预测值UMAP密度图可视化分析")
    result = prediction_value_density_visualization(
        input_file=INPUT_RDS_FILE,
        base_dir=OUTPUT_DIR,
        prediction_keywords="bind,predict,output,average,score",
        prediction_threshold=0.5
    )
    print(f"结果: {result}")
    print("-" * 50)

def run_all_figure3_tests():
    """运行所有Figure3相关工具测试"""
    print("=" * 60)
    print("开始运行Figure3相关工具测试")
    print("=" * 60)
    
    # test_antigen_binding_neutralization_density_visualization()
    # test_bcell_celltype_umap_visualization()
    # test_bcell_marker_gene_expression_dotplot()
    test_differential_gene_correlation_analysis()
    # test_prediction_value_density_visualization()
    
    print("=" * 60)
    print("Figure3相关工具测试完成")
    print("=" * 60)

if __name__ == "__main__":
    run_all_figure3_tests()