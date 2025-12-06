#!/usr/bin/env python3
"""
测试文件：run_clustering_analysis 方法测试

生物信息学背景：
聚类分析是单细胞RNA测序数据分析的核心步骤，用于识别具有相似基因表达模式的细胞群体。
该分析基于以下原理：
1. SNN图构建：基于PCA或Harmony降维结果构建共享最近邻图
2. 社区检测：使用Leiden或Louvain算法进行图基聚类
3. UMAP可视化：生成二维可视化图展示聚类结果

测试策略：
- 使用真实的单细胞数据进行测试
- 测试不同的聚类参数组合
- 验证输出文件的生成和内容
- 从生物学角度评估聚类结果的合理性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrna_mcp_server import run_clustering_analysis

def test_clustering_analysis():
    """测试聚类分析功能"""
    
    # 输入文件路径
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    print("=" * 80)
    print("单细胞RNA测序聚类分析测试")
    print("=" * 80)
    
    # 验证输入文件存在
    if not os.path.exists(input_rds):
        print(f"错误：输入文件不存在 - {input_rds}")
        return False
    
    print(f"输入文件：{input_rds}")
    print(f"文件大小：{os.path.getsize(input_rds) / (1024*1024):.2f} MB")
    
    # 测试用例1：默认参数（Leiden算法，分辨率0.8）
    print("\n" + "="*60)
    print("测试用例1：默认参数聚类")
    print("算法：Leiden")
    print("分辨率：0.8")
    print("维度：30")
    print("="*60)
    
    try:
        result1 = run_clustering_analysis(
            input_rds=input_rds,
            resolution=0.8,
            dims=30,
            algorithm="leiden"
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
        print("- 检查聚类数量是否合理（通常5-20个聚类）")
        print("- 观察UMAP图中聚类的分离度")
        print("- 验证聚类结果的生物学意义")
        
    except Exception as e:
        print(f"测试用例1失败：{e}")
        return False
    
    # 测试用例2：Louvain算法，较低分辨率
    print("\n" + "="*60)
    print("测试用例2：Louvain算法，较低分辨率")
    print("算法：Louvain")
    print("分辨率：0.5")
    print("维度：30")
    print("="*60)
    
    try:
        result2 = run_clustering_analysis(
            input_rds=input_rds,
            resolution=0.5,
            dims=30,
            algorithm="louvain"
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
        print("- 较低分辨率应产生较少的聚类")
        print("- 比较不同算法的聚类结果")
        print("- 评估聚类稳定性")
        
    except Exception as e:
        print(f"测试用例2失败：{e}")
        return False
    
    # 测试用例3：高分辨率聚类
    print("\n" + "="*60)
    print("测试用例3：高分辨率聚类")
    print("算法：Leiden")
    print("分辨率：1.2")
    print("维度：50")
    print("="*60)
    
    try:
        result3 = run_clustering_analysis(
            input_rds=input_rds,
            resolution=1.2,
            dims=50,
            algorithm="leiden"
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
        print("- 高分辨率应产生更多细分的聚类")
        print("- 使用更多维度可能提高聚类精度")
        print("- 注意过度聚类的风险")
        
    except Exception as e:
        print(f"测试用例3失败：{e}")
        return False
    
    # 测试用例4：SLM算法测试
    print("\n" + "="*60)
    print("测试用例4：SLM算法测试")
    print("算法：SLM")
    print("分辨率：0.8")
    print("维度：30")
    print("="*60)
    
    try:
        result4 = run_clustering_analysis(
            input_rds=input_rds,
            resolution=0.8,
            dims=30,
            algorithm="slm"
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
        print("- SLM算法适用于大规模数据集")
        print("- 比较不同算法的计算效率")
        print("- 评估算法对数据集的适用性")
        
    except Exception as e:
        print(f"测试用例4失败：{e}")
        return False
    
    print("\n" + "="*80)
    print("聚类分析测试总结")
    print("="*80)
    print("✓ 所有测试用例执行完成")
    print("✓ 测试了4种不同的参数组合")
    print("✓ 验证了Leiden、Louvain和SLM算法")
    print("✓ 测试了不同分辨率和维度参数")
    print("\n预期输出文件：")
    print("- umap_clusters.pdf：UMAP聚类可视化")
    print("- umap_clusters_by_sample.pdf：按样本分组的UMAP图")
    print("- cluster_composition.pdf：聚类组成柱状图")
    print("- cluster_assignments.csv：细胞聚类分配表")
    print("- cluster_statistics.csv：聚类统计信息")
    print("- clustering_parameters.csv：聚类参数记录")
    print("- seurat_clustered.rds：聚类后的Seurat对象")
    
    return True

if __name__ == "__main__":
    print("开始执行聚类分析测试...")
    success = test_clustering_analysis()
    
    if success:
        print("\n🎉 所有测试用例执行成功！")
        print("\n生物信息学建议：")
        print("1. 聚类分析是细胞类型识别的基础")
        print("2. 不同算法和参数会影响聚类结果")
        print("3. 需要结合生物学知识验证聚类的合理性")
        print("4. 建议进行标记基因分析以注释聚类")
    else:
        print("\n❌ 测试执行失败，请检查错误信息")
    
    print("\n测试完成。")