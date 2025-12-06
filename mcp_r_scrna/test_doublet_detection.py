#!/usr/bin/env python3
"""
DoubletFinder测试文件

DoubletFinder生物信息学背景：
DoubletFinder是一种用于单细胞RNA测序数据中检测doublets（双细胞）的计算方法。
Doublets是指在单细胞捕获过程中意外捕获的两个或多个细胞，它们会产生混合的转录组信号，
可能导致错误的细胞类型注释和下游分析结果的偏差。

DoubletFinder工作原理：
1. 人工生成doublets：通过随机组合现有单细胞的表达谱来模拟真实doublets
2. 计算doublet分数：使用PCA降维和k近邻算法计算每个细胞的doublet概率
3. 分类细胞：基于doublet分数阈值将细胞分类为singlet或doublet

关键参数说明：
- expected_doublet_rate: 预期doublet形成率，通常为0.08 (8%)
- pN: 人工doublets的比例，默认0.25 (25%)
- pK: PC邻域大小参数，影响doublet检测的敏感性，默认0.09
- dims: 用于分析的主成分数量，默认20

测试策略：
本测试使用真实的scRNA-seq数据，测试不同参数组合下DoubletFinder的性能，
验证doublet检测的准确性和输出文件的完整性。
"""

import os
import sys
import json
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from scrna_mcp_server import run_doublet_detection


def validate_input_file(input_rds):
    """验证输入RDS文件"""
    print(f"验证输入文件: {input_rds}")
    
    if not os.path.exists(input_rds):
        raise FileNotFoundError(f"输入文件不存在: {input_rds}")
    
    file_size = os.path.getsize(input_rds)
    print(f"文件大小: {file_size / (1024*1024):.2f} MB")
    
    if file_size == 0:
        raise ValueError("输入文件为空")
    
    print("✓ 输入文件验证通过")


def analyze_results(result, test_name):
    """分析DoubletFinder结果"""
    print(f"\n=== {test_name} 结果分析 ===")
    print(f"执行状态: {result.get('status', 'unknown')}")
    print(f"消息: {result.get('message', 'No message')}")
    
    generated_files = result.get('generated_files', [])
    print(f"生成文件数量: {len(generated_files)}")
    
    # 检查预期的输出文件
    expected_files = [
        'doublet_assignments.csv',
        'doublet_statistics.csv', 
        'seurat_singlets.rds',
        'umap_before_doublet_removal.pdf',
        'umap_doublets_labeled.pdf',
        'umap_doublet_scores.pdf',
        'umap_after_doublet_removal.pdf',
        'doublet_score_distribution.pdf'
    ]
    
    found_files = []
    for expected_file in expected_files:
        found = any(expected_file in file_path for file_path in generated_files)
        status = "✓" if found else "✗"
        print(f"  {status} {expected_file}")
        if found:
            found_files.append(expected_file)
    
    print(f"找到预期文件: {len(found_files)}/{len(expected_files)}")
    
    # 分析doublet统计信息
    if result.get('status') == 'success':
        doublet_stats_file = None
        for file_path in generated_files:
            if 'doublet_statistics.csv' in file_path:
                doublet_stats_file = file_path
                break
        
        if doublet_stats_file and os.path.exists(doublet_stats_file):
            print(f"\n--- Doublet统计信息 ---")
            try:
                with open(doublet_stats_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[:10]:  # 显示前10行
                        print(f"  {line.strip()}")
            except Exception as e:
                print(f"读取统计文件失败: {e}")
    
    return result.get('status') == 'success'


def test_doublet_detection_default():
    """测试用例1: 默认参数doublet检测"""
    print("\n" + "="*60)
    print("测试用例1: 默认参数doublet检测")
    print("="*60)
    
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    try:
        validate_input_file(input_rds)
        
        print("\n执行DoubletFinder (默认参数)...")
        result = run_doublet_detection(
            input_rds=input_rds,
            expected_doublet_rate=0.08,  # 8% doublet率
            pN=0.25,                     # 25% 人工doublets
            pK=0.09,                     # 默认PC邻域大小
            dims=20                      # 20个主成分
        )
        
        success = analyze_results(result, "默认参数doublet检测")
        return success
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_doublet_detection_high_rate():
    """测试用例2: 高doublet率检测"""
    print("\n" + "="*60)
    print("测试用例2: 高doublet率检测")
    print("="*60)
    
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    try:
        print("\n执行DoubletFinder (高doublet率)...")
        result = run_doublet_detection(
            input_rds=input_rds,
            expected_doublet_rate=0.15,  # 15% 高doublet率
            pN=0.25,                     # 25% 人工doublets
            pK=0.09,                     # 默认PC邻域大小
            dims=20                      # 20个主成分
        )
        
        success = analyze_results(result, "高doublet率检测")
        return success
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_doublet_detection_low_pk():
    """测试用例3: 低pK值检测（更敏感）"""
    print("\n" + "="*60)
    print("测试用例3: 低pK值检测（更敏感）")
    print("="*60)
    
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    try:
        print("\n执行DoubletFinder (低pK值)...")
        result = run_doublet_detection(
            input_rds=input_rds,
            expected_doublet_rate=0.08,  # 8% doublet率
            pN=0.25,                     # 25% 人工doublets
            pK=0.05,                     # 更小的PC邻域，更敏感
            dims=20                      # 20个主成分
        )
        
        success = analyze_results(result, "低pK值检测")
        return success
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_doublet_detection_high_dims():
    """测试用例4: 高维度分析"""
    print("\n" + "="*60)
    print("测试用例4: 高维度分析")
    print("="*60)
    
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    try:
        print("\n执行DoubletFinder (高维度)...")
        result = run_doublet_detection(
            input_rds=input_rds,
            expected_doublet_rate=0.08,  # 8% doublet率
            pN=0.30,                     # 30% 人工doublets
            pK=0.09,                     # 默认PC邻域大小
            dims=30                      # 30个主成分，更多信息
        )
        
        success = analyze_results(result, "高维度分析")
        return success
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


if __name__ == "__main__":
    print("DoubletFinder测试开始")
    print("=" * 80)
    
    # 运行所有测试用例
    test_results = []
    
    test_results.append(("默认参数doublet检测", test_doublet_detection_default()))
    test_results.append(("高doublet率检测", test_doublet_detection_high_rate()))
    test_results.append(("低pK值检测", test_doublet_detection_low_pk()))
    test_results.append(("高维度分析", test_doublet_detection_high_dims()))
    
    # 测试总结
    print("\n" + "="*80)
    print("DoubletFinder测试总结")
    print("="*80)
    
    successful_tests = 0
    for test_name, success in test_results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{status}: {test_name}")
        if success:
            successful_tests += 1
    
    print(f"\n测试完成: {successful_tests}/{len(test_results)} 个测试用例成功")
    
    if successful_tests == len(test_results):
        print("\n🎉 所有测试用例执行成功！")
        print("\nDoubletFinder生物信息学建议：")
        print("1. Doublet检测是scRNA-seq数据质控的重要步骤")
        print("2. 不同的pK值会影响检测敏感性，建议根据数据特点调整")
        print("3. Expected doublet rate应根据实验条件和细胞捕获方法设定")
        print("4. 检测后的singlet细胞可用于下游聚类和差异表达分析")
        print("5. 建议保存doublet分数信息，用于后续质量评估")
    else:
        print(f"\n⚠️  {len(test_results) - successful_tests} 个测试用例失败，请检查错误信息")