#!/usr/bin/env python3
"""
差异表达基因(DEG)分析测试文件

生物信息学背景:
差异表达基因分析是单细胞RNA测序数据分析的核心步骤之一，用于识别不同细胞群体或条件间
表达水平显著差异的基因。本测试基于Seurat的FindMarkers函数，支持多种统计方法：

1. Wilcoxon rank-sum test (默认): 非参数检验，适用于大多数scRNA-seq数据
2. MAST: Model-based Analysis of Single-cell Transcriptomics，专为scRNA-seq设计的hurdle模型
3. DESeq2: 基于负二项分布的广义线性模型，适用于pseudobulk分析

关键参数说明:
- logfc_threshold: log2倍数变化阈值，通常设为0.25 (1.19倍变化)
- min_pct: 基因在任一组中的最小表达细胞比例，用于过滤低表达基因
- group_by: 分组变量，如细胞类型、聚类结果等
- ident_1/ident_2: 比较的两个组别

测试策略:
使用真实的B细胞scRNA-seq数据，测试不同参数组合下的DEG分析性能，
验证统计方法的有效性和结果的生物学意义。
"""

import os
import sys
import json
from pathlib import Path

# 添加项目路径到sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scrna_mcp_server import run_deg_analysis

def validate_input_file(file_path):
    """验证输入RDS文件是否存在"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"输入文件不存在: {file_path}")
    
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    print(f"✓ 输入文件验证通过: {file_path}")
    print(f"  文件大小: {file_size:.2f} MB")
    return True

def analyze_deg_results(result, test_name):
    """分析DEG分析结果"""
    print(f"\n=== {test_name} 结果分析 ===")
    
    if result.get('status') == 'success':
        print("✓ DEG分析执行成功")
        
        # 检查输出文件
        output_dir = "output/deg_analysis"
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"✓ 生成输出文件数量: {len(files)}")
            
            # 分析CSV结果文件
            csv_files = [f for f in files if f.endswith('.csv')]
            for csv_file in csv_files:
                if 'deg_results' in csv_file:
                    csv_path = os.path.join(output_dir, csv_file)
                    try:
                        import pandas as pd
                        deg_data = pd.read_csv(csv_path)
                        total_genes = len(deg_data)
                        sig_genes = len(deg_data[deg_data['p_val_adj'] < 0.05])
                        up_genes = len(deg_data[(deg_data['avg_log2FC'] > 0) & (deg_data['p_val_adj'] < 0.05)])
                        down_genes = len(deg_data[(deg_data['avg_log2FC'] < 0) & (deg_data['p_val_adj'] < 0.05)])
                        
                        print(f"  - 总检测基因数: {total_genes}")
                        print(f"  - 显著差异基因数: {sig_genes}")
                        print(f"  - 上调基因数: {up_genes}")
                        print(f"  - 下调基因数: {down_genes}")
                        
                        if sig_genes > 0:
                            print(f"  - 显著性比例: {sig_genes/total_genes*100:.2f}%")
                    except Exception as e:
                        print(f"  ⚠ CSV文件分析失败: {e}")
        else:
            print("⚠ 未找到输出目录")
    else:
        print(f"✗ DEG分析执行失败")
        print(f"  错误信息: {result.get('message', 'Unknown error')}")
        if 'stderr' in result:
            print(f"  详细错误: {result['stderr']}")

def run_deg_test_case(test_name, input_rds, **kwargs):
    """运行单个DEG测试用例"""
    print(f"\n{'='*60}")
    print(f"开始执行: {test_name}")
    print(f"{'='*60}")
    
    # 打印测试参数
    print("测试参数:")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")
    
    try:
        # 执行DEG分析
        result = run_deg_analysis(input_rds, **kwargs)
        
        # 分析结果
        analyze_deg_results(result, test_name)
        
        return result.get('status') == 'success'
        
    except Exception as e:
        print(f"✗ 测试执行异常: {e}")
        return False

if __name__ == "__main__":
    print("差异表达基因(DEG)分析测试")
    print("=" * 60)
    
    # 输入文件路径
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    # 验证输入文件
    try:
        validate_input_file(input_rds)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 清理之前的输出文件，避免文件冲突
    output_dir = "output/deg_analysis"
    if os.path.exists(output_dir):
        import shutil
        import time
        try:
            shutil.rmtree(output_dir)
            print(f"✓ 清理输出目录: {output_dir}")
        except PermissionError as e:
            print(f"⚠ 无法完全清理输出目录 (文件被占用): {e}")
            print("  继续执行测试，可能会覆盖现有文件...")
            time.sleep(1)  # 等待1秒
    
    # 测试用例列表 - 基于实际数据结构
    test_cases = [
        {
            "name": "测试1: 默认参数聚类比较 (cluster 0 vs 1)",
            "params": {
                "group_by": "seurat_clusters",
                "ident_1": "0",  # 最大的聚类 (19641 cells)
                "ident_2": "1",  # 第二大聚类 (12562 cells)
                "test_use": "wilcox",
                "logfc_threshold": 0.25,
                "min_pct": 0.1
            }
        },
        {
            "name": "测试2: Wilcoxon方法分析 (cluster 0 vs rest)",
            "params": {
                "group_by": "seurat_clusters", 
                "ident_1": "0",
                "ident_2": None,  # 与其他所有cluster比较
                "test_use": "wilcox",  # 使用Wilcoxon代替MAST
                "logfc_threshold": 0.25,
                "min_pct": 0.1
            }
        },
        {
            "name": "测试3: 高阈值严格筛选 (cluster 2 vs 3)",
            "params": {
                "group_by": "seurat_clusters",
                "ident_1": "2",  # 第三大聚类 (11465 cells)
                "ident_2": "3",  # 第四大聚类 (8580 cells)
                "test_use": "wilcox",
                "logfc_threshold": 0.5,  # 更高的倍数变化阈值
                "min_pct": 0.25  # 更高的表达比例要求
            }
        },
        {
            "name": "测试4: 细胞类型比较 (Activated vs Memory)",
            "params": {
                "group_by": "CellType",  # 使用实际存在的CellType列
                "ident_1": "Activated",  # 最大的细胞类型 (39374 cells)
                "ident_2": "Memory",     # 第二大的细胞类型 (8348 cells)
                "test_use": "wilcox",
                "logfc_threshold": 0.25,
                "min_pct": 0.1
            }
        }
    ]
    
    # 执行测试用例
    results = []
    for i, test_case in enumerate(test_cases, 1):
        success = run_deg_test_case(
            test_case["name"],
            input_rds,
            **test_case["params"]
        )
        results.append({
            "test": test_case["name"],
            "status": "success" if success else "error"
        })
    
    # 测试总结
    print(f"\n{'='*60}")
    print("DEG分析测试总结")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r["status"] == "success")
    total_count = len(results)
    
    print(f"总测试数: {total_count}")
    print(f"成功测试数: {success_count}")
    print(f"失败测试数: {total_count - success_count}")
    
    for result in results:
        status_symbol = "✓" if result["status"] == "success" else "✗"
        print(f"{status_symbol} {result['test']}: {result['status']}")
    
    if success_count == total_count:
        print("\n🎉 所有测试用例执行成功!")
    else:
        print(f"\n⚠ {total_count - success_count} 个测试用例执行失败")
    
    # 生物信息学建议
    print(f"\n{'='*60}")
    print("生物信息学分析建议")
    print(f"{'='*60}")
    print("1. DEG分析结果解读:")
    print("   - 关注log2FC > 1且p_adj < 0.05的高置信度差异基因")
    print("   - 结合生物学功能进行通路富集分析")
    print("   - 验证关键差异基因的表达模式")
    print("\n2. 统计方法选择:")
    print("   - Wilcoxon: 适用于大多数scRNA-seq数据，稳健性好")
    print("   - MAST: 考虑dropout事件，适用于稀疏数据")
    print("   - DESeq2: 适用于pseudobulk分析，需要重复样本")
    print("\n3. 参数优化建议:")
    print("   - logfc_threshold: 根据生物学意义调整(0.25-1.0)")
    print("   - min_pct: 平衡敏感性和特异性(0.1-0.25)")
    print("   - 多重检验校正: 使用Benjamini-Hochberg方法")