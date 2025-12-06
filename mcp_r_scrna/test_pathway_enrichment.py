#!/usr/bin/env python3
"""
通路富集分析(Pathway Enrichment)测试文件

生物信息学背景:
通路富集分析是单细胞RNA测序数据分析的重要下游分析步骤，用于解释差异表达基因的生物学功能。
本测试基于clusterProfiler包，支持多种富集分析方法：

1. GO富集分析 (Gene Ontology):
   - BP (Biological Process): 生物学过程
   - MF (Molecular Function): 分子功能  
   - CC (Cellular Component): 细胞组分

2. KEGG通路分析 (Kyoto Encyclopedia of Genes and Genomes):
   - 代谢通路
   - 信号转导通路
   - 疾病相关通路

3. GSEA (Gene Set Enrichment Analysis):
   - 基于基因表达排序的富集分析
   - 不需要预设阈值，考虑所有基因

关键参数说明:
- organism: 物种选择 ("human"/"mouse")，影响注释数据库
- ontology: GO本体选择 ("BP"/"MF"/"CC")
- pvalue_cutoff: P值阈值，控制富集显著性
- qvalue_cutoff: FDR校正后的Q值阈值，控制假阳性率

测试策略:
1. 首先运行DEG分析生成差异表达基因列表
2. 使用DEG结果进行通路富集分析
3. 验证富集结果的生物学意义和统计显著性
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# 添加项目路径到sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scrna_mcp_server import run_deg_analysis, run_pathway_enrichment


def validate_input_file(file_path):
    """验证输入文件是否存在且为RDS格式"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"输入文件不存在: {file_path}")
    
    if not file_path.lower().endswith('.rds'):
        raise ValueError(f"输入文件必须是RDS格式: {file_path}")
    
    # 检查文件大小
    file_size = os.path.getsize(file_path)
    print(f"✓ 输入文件验证通过: {file_path}")
    print(f"  文件大小: {file_size / (1024*1024):.1f} MB")


def run_deg_for_pathway_enrichment(input_rds):
    """运行DEG分析，为pathway enrichment准备输入数据"""
    print("\n" + "="*60)
    print("步骤1: 运行DEG分析生成差异表达基因列表")
    print("="*60)
    
    # DEG分析参数 - 选择有生物学意义的比较
    deg_params = {
        "group_by": "seurat_clusters",
        "ident_1": "0",  # 最大的聚类
        "ident_2": "1",  # 第二大聚类
        "test_use": "wilcox",
        "logfc_threshold": 0.25,
        "min_pct": 0.1
    }
    
    print("DEG分析参数:")
    for key, value in deg_params.items():
        print(f"  {key}: {value}")
    
    try:
        # 执行DEG分析
        deg_result = run_deg_analysis(input_rds, **deg_params)
        
        if deg_result.get('status') == 'success':
            print("✓ DEG分析执行成功")
            
            # 查找生成的DEG CSV文件
            output_dir = os.path.join(project_root, "output", "deg_analysis")
            if os.path.exists(output_dir):
                csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and 'deg_' in f]
                if csv_files:
                    deg_csv_path = os.path.join(output_dir, csv_files[0])
                    print(f"✓ DEG结果文件: {deg_csv_path}")
                    
                    # 分析DEG结果
                    try:
                        import pandas as pd
                        deg_data = pd.read_csv(deg_csv_path)
                        total_genes = len(deg_data)
                        sig_genes = len(deg_data[deg_data['p_val_adj'] < 0.05])
                        print(f"  - 总检测基因数: {total_genes}")
                        print(f"  - 显著差异基因数: {sig_genes}")
                        
                        if sig_genes > 0:
                            return deg_csv_path
                        else:
                            print("⚠ 警告: 未发现显著差异基因，但继续进行通路分析")
                            return deg_csv_path
                    except Exception as e:
                        print(f"⚠ 无法分析DEG结果: {e}")
                        return deg_csv_path
                else:
                    raise FileNotFoundError("未找到DEG结果CSV文件")
            else:
                raise FileNotFoundError("DEG输出目录不存在")
        else:
            raise RuntimeError(f"DEG分析失败: {deg_result.get('message', '未知错误')}")
            
    except Exception as e:
        print(f"✗ DEG分析执行失败: {e}")
        raise


def analyze_pathway_results(result, test_name):
    """分析pathway enrichment结果"""
    print(f"\n=== {test_name} 结果分析 ===")
    
    if result.get('status') == 'success':
        print("✓ Pathway enrichment分析执行成功")
        
        # 检查输出文件
        output_dir = "output/pathway_enrichment"
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"✓ 生成输出文件数量: {len(files)}")
            
            # 分析富集结果文件
            csv_files = [f for f in files if f.endswith('.csv')]
            pdf_files = [f for f in files if f.endswith('.pdf')]
            
            print(f"  - CSV结果文件: {len(csv_files)}")
            print(f"  - PDF图形文件: {len(pdf_files)}")
            
            # 分析具体的富集结果
            for csv_file in csv_files:
                if 'enrichment' in csv_file.lower():
                    csv_path = os.path.join(output_dir, csv_file)
                    try:
                        import pandas as pd
                        enrich_data = pd.read_csv(csv_path)
                        if len(enrich_data) > 0:
                            sig_pathways = len(enrich_data[enrich_data['p.adjust'] < 0.05])
                            print(f"  - {csv_file}: {len(enrich_data)} 个通路, {sig_pathways} 个显著")
                        else:
                            print(f"  - {csv_file}: 无富集结果")
                    except Exception as e:
                        print(f"  - {csv_file}: 无法解析 ({e})")
        else:
            print("⚠ 输出目录不存在")
    else:
        print(f"✗ Pathway enrichment分析失败: {result.get('message', '未知错误')}")


def run_pathway_test_case(test_name, input_rds, deg_csv, **kwargs):
    """运行单个pathway enrichment测试用例"""
    print(f"\n{'='*60}")
    print(f"步骤2: {test_name}")
    print(f"{'='*60}")
    
    # 打印测试参数
    print("Pathway enrichment参数:")
    print(f"  input_rds: {input_rds}")
    print(f"  deg_csv: {deg_csv}")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")
    
    try:
        # 执行pathway enrichment分析
        result = run_pathway_enrichment(input_rds, deg_csv, **kwargs)
        
        # 分析结果
        analyze_pathway_results(result, test_name)
        
        return result.get('status') == 'success'
        
    except Exception as e:
        print(f"✗ 测试执行异常: {e}")
        return False


if __name__ == "__main__":
    print("通路富集分析(Pathway Enrichment)测试")
    print("=" * 60)
    
    # 输入文件路径
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    # 验证输入文件
    try:
        validate_input_file(input_rds)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 清理之前的输出文件
    for output_subdir in ["deg_analysis", "pathway_enrichment"]:
        output_dir = f"output/{output_subdir}"
        if os.path.exists(output_dir):
            import shutil
            try:
                shutil.rmtree(output_dir)
                print(f"✓ 清理输出目录: {output_dir}")
            except PermissionError as e:
                print(f"⚠ 无法完全清理输出目录 (文件被占用): {e}")
    
    try:
        # 步骤1: 运行DEG分析
        deg_csv_path = run_deg_for_pathway_enrichment(input_rds)
        
        # 步骤2: 运行pathway enrichment测试用例
        test_cases = [
            {
                "name": "GO生物学过程富集分析 (人类)",
                "params": {
                    "organism": "human",
                    "ontology": "BP",
                    "pvalue_cutoff": 0.05,
                    "qvalue_cutoff": 0.2
                }
            },
            {
                "name": "GO分子功能富集分析 (人类)",
                "params": {
                    "organism": "human", 
                    "ontology": "MF",
                    "pvalue_cutoff": 0.05,
                    "qvalue_cutoff": 0.2
                }
            },
            {
                "name": "GO细胞组分富集分析 (人类)",
                "params": {
                    "organism": "human",
                    "ontology": "CC", 
                    "pvalue_cutoff": 0.05,
                    "qvalue_cutoff": 0.2
                }
            },
            {
                "name": "严格阈值GO富集分析",
                "params": {
                    "organism": "human",
                    "ontology": "BP",
                    "pvalue_cutoff": 0.01,  # 更严格的P值阈值
                    "qvalue_cutoff": 0.1    # 更严格的Q值阈值
                }
            }
        ]
        
        # 执行测试用例
        results = []
        for i, test_case in enumerate(test_cases, 1):
            success = run_pathway_test_case(
                test_case["name"],
                input_rds,
                deg_csv_path,
                **test_case["params"]
            )
            results.append({
                "test": test_case["name"],
                "status": "success" if success else "error"
            })
        
        # 测试总结
        print(f"\n{'='*60}")
        print("Pathway Enrichment测试总结")
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
        print("1. 通路富集结果解读:")
        print("   - 关注p.adjust < 0.05且富集倍数高的通路")
        print("   - 结合基因比例(GeneRatio)评估富集强度")
        print("   - 验证富集基因的生物学相关性")
        print("\n2. GO本体选择策略:")
        print("   - BP: 关注生物学功能和调控过程")
        print("   - MF: 关注蛋白质分子功能")
        print("   - CC: 关注亚细胞定位和组织结构")
        print("\n3. 参数优化建议:")
        print("   - pvalue_cutoff: 平衡敏感性和特异性(0.01-0.05)")
        print("   - qvalue_cutoff: 控制假阳性率(0.1-0.2)")
        print("   - 考虑基因集大小和背景基因数量")
        print("\n4. 下游分析方向:")
        print("   - 通路网络分析和功能模块识别")
        print("   - 关键调控基因的表达验证")
        print("   - 与表型数据的关联分析")
        
    except Exception as e:
        print(f"\n✗ 测试流程执行失败: {e}")
        sys.exit(1)