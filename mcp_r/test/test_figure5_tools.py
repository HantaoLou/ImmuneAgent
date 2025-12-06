#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure5相关工具测试文件

测试生信分析模块化MCP服务器中的Figure5相关工具
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bioinformatics_mcp_server import (
    bcr_isotype_distribution_shm_analysis,
    neutralizing_antibody_shm_comparison_analysis
)

# 测试参数配置
INPUT_RDS_FILE = r"D:\data\test_data_20251001\Age_Bcells.rds"
OUTPUT_DIR = r"D:\data\test_data_20251001"

def test_bcr_isotype_distribution_shm_analysis():
    """测试B细胞受体同型分布和体细胞超突变率分析"""
    print("测试: B细胞受体同型分布和体细胞超突变率分析")
    result = bcr_isotype_distribution_shm_analysis(
        input_file=INPUT_RDS_FILE,
        base_dir=OUTPUT_DIR,
        binding_threshold=0.5
    )
    print(f"结果: {result}")
    print("-" * 50)

def test_neutralizing_antibody_shm_comparison_analysis():
    """测试中和抗体与非中和抗体SHM率比较分析"""
    print("测试: 中和抗体与非中和抗体SHM率比较分析")
    result = neutralizing_antibody_shm_comparison_analysis(
        input_file=INPUT_RDS_FILE,
        base_dir=OUTPUT_DIR,
        binding_threshold=0.5
    )
    print(f"结果: {result}")
    print("-" * 50)

def run_all_figure5_tests():
    """运行所有Figure5相关工具测试"""
    print("=" * 60)
    print("开始运行Figure5相关工具测试")
    print("=" * 60)
    
    test_bcr_isotype_distribution_shm_analysis()
    test_neutralizing_antibody_shm_comparison_analysis()
    
    print("=" * 60)
    print("Figure5相关工具测试完成")
    print("=" * 60)

if __name__ == "__main__":
    run_all_figure5_tests()