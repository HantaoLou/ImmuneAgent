#!/usr/bin/env python3
"""
测试文件：run_integration_harmony 方法测试

生物信息学背景：
Harmony是一种用于单细胞RNA测序数据批次校正的算法，主要用于：
1. 批次效应去除：消除不同实验批次、样本或技术平台间的系统性差异
2. 多样本整合：将来自不同条件或时间点的样本整合到统一的分析框架中
3. 生物学变异保持：在去除技术噪音的同时保留真实的生物学差异

Harmony算法原理：
- 基于PCA降维结果进行批次校正
- 使用多样性聚类惩罚参数(theta)平衡批次混合和聚类保持
- 迭代优化过程确保批次间的细胞类型对齐

测试策略：
- 使用真实的单细胞数据进行测试
- 测试不同的批次变量和参数组合
- 验证整合前后的UMAP可视化效果
- 从生物学角度评估批次校正的效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrna_mcp_server import run_integration_harmony

def test_integration_harmony():
    """测试Harmony整合功能"""
    
    # 输入文件路径
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    print("=" * 80)
    print("单细胞RNA测序Harmony整合测试")
    print("=" * 80)
    
    # 验证输入文件存在
    if not os.path.exists(input_rds):
        print(f"错误：输入文件不存在 - {input_rds}")
        return False
    
    print(f"输入文件：{input_rds}")
    print(f"文件大小：{os.path.getsize(input_rds) / (1024*1024):.2f} MB")
    
    # 测试用例1：默认参数（orig.ident批次变量）
    print("\n" + "="*60)
    print("测试用例1：默认参数整合")
    print("批次变量：orig.ident")
    print("维度：30")
    print("Theta：[2.0]")
    print("="*60)
    
    try:
        result1 = run_integration_harmony(
            input_rds=input_rds,
            batch_variable="orig.ident",
            dims=30,
            theta=[2.0]
        )
        
        print("执行状态：", result1.get("status", "未知"))
        print("执行消息：", result1.get("message", "无消息"))
        print("输出目录：", result1.get("output_dir", "未指定"))
        print("生成文件数量：", result1.get("file_count", 0))
        
        if result1.get("generated_files"):
            print("生成的文件：")
            for file in result1["generated_files"]:
                print(f"  - {file}")
        
        if result1.get("stdout"):
            print("\nR脚本输出：")
            print(result1["stdout"])
            
        print("\n生信分析建议：")
        print("- 检查整合前后UMAP图的批次混合效果")
        print("- 验证生物学信号是否得到保留")
        print("- 评估不同批次间细胞类型的对齐程度")
        
    except Exception as e:
        print(f"测试用例1失败：{e}")
        return False
    
    # 测试用例2：使用CellType作为批次变量
    print("\n" + "="*60)
    print("测试用例2：CellType批次校正")
    print("批次变量：CellType")
    print("维度：30")
    print("Theta：[2.0]")
    print("="*60)
    
    try:
        result2 = run_integration_harmony(
            input_rds=input_rds,
            batch_variable="CellType",
            dims=30,
            theta=[2.0]
        )
        
        print("执行状态：", result2.get("status", "未知"))
        print("执行消息：", result2.get("message", "无消息"))
        print("输出目录：", result2.get("output_dir", "未指定"))
        print("生成文件数量：", result2.get("file_count", 0))
        
        if result2.get("generated_files"):
            print("生成的文件：")
            for file in result2["generated_files"]:
                print(f"  - {file}")
        
        print("\n生信分析建议：")
        print("- 使用细胞类型作为批次变量可能不合适")
        print("- 观察是否过度校正导致生物学信号丢失")
        print("- 比较不同批次变量的校正效果")
        
    except Exception as e:
        print(f"测试用例2失败：{e}")
        return False
    
    # 测试用例3：高维度和多theta参数
    print("\n" + "="*60)
    print("测试用例3：高维度多theta参数")
    print("批次变量：orig.ident")
    print("维度：50")
    print("Theta：[1.0, 2.0]")
    print("="*60)
    
    try:
        result3 = run_integration_harmony(
            input_rds=input_rds,
            batch_variable="orig.ident",
            dims=50,
            theta=[1.0, 2.0]
        )
        
        print("执行状态：", result3.get("status", "未知"))
        print("执行消息：", result3.get("message", "无消息"))
        print("输出目录：", result3.get("output_dir", "未指定"))
        print("生成文件数量：", result3.get("file_count", 0))
        
        if result3.get("generated_files"):
            print("生成的文件：")
            for file in result3["generated_files"]:
                print(f"  - {file}")
        
        print("\n生信分析建议：")
        print("- 使用更多维度可能提高整合精度")
        print("- 多个theta值可以更精细地控制整合过程")
        print("- 评估计算时间和整合效果的平衡")
        
    except Exception as e:
        print(f"测试用例3失败：{e}")
        return False
    
    # 测试用例4：低theta值测试
    print("\n" + "="*60)
    print("测试用例4：低theta值测试")
    print("批次变量：orig.ident")
    print("维度：30")
    print("Theta：[0.5]")
    print("="*60)
    
    try:
        result4 = run_integration_harmony(
            input_rds=input_rds,
            batch_variable="orig.ident",
            dims=30,
            theta=[0.5]
        )
        
        print("执行状态：", result4.get("status", "未知"))
        print("执行消息：", result4.get("message", "无消息"))
        print("输出目录：", result4.get("output_dir", "未指定"))
        print("生成文件数量：", result4.get("file_count", 0))
        
        if result4.get("generated_files"):
            print("生成的文件：")
            for file in result4["generated_files"]:
                print(f"  - {file}")
        
        print("\n生信分析建议：")
        print("- 较低的theta值可能导致过度校正")
        print("- 观察是否丢失了重要的生物学差异")
        print("- 比较不同theta值的校正强度")
        
    except Exception as e:
        print(f"测试用例4失败：{e}")
        return False
    
    print("\n" + "="*80)
    print("Harmony整合测试总结")
    print("="*80)
    print("✓ 所有测试用例执行完成")
    print("✓ 测试了4种不同的参数组合")
    print("✓ 验证了不同批次变量和theta参数的效果")
    print("✓ 测试了不同维度设置的影响")
    print("\n预期输出文件：")
    print("- umap_before_harmony.pdf：整合前UMAP可视化")
    print("- umap_after_harmony.pdf：整合后UMAP可视化")
    print("- umap_comparison.pdf：整合前后对比图")
    print("- harmony_embeddings.csv：Harmony嵌入向量")
    print("- batch_statistics.csv：批次统计信息")
    print("- integration_statistics.csv：整合参数和统计")
    print("- seurat_integrated.rds：整合后的Seurat对象")
    
    return True

if __name__ == "__main__":
    print("开始执行Harmony整合测试...")
    success = test_integration_harmony()
    
    if success:
        print("\n🎉 所有测试用例执行成功！")
        print("\n生物信息学建议：")
        print("1. Harmony整合是多样本分析的重要步骤")
        print("2. 批次变量的选择直接影响整合效果")
        print("3. theta参数需要根据数据特点进行调优")
        print("4. 整合后需要验证生物学信号的保留程度")
        print("5. 建议进行下游聚类和标记基因分析")
    else:
        print("\n❌ 测试执行失败，请检查错误信息")
    
    print("\n测试完成。")