"""
测试 filter_by_vdj_genes 工具

此测试文件用于测试 filter_by_vdj_genes 工具的功能，
该工具用于按 V/D/J 基因使用模式过滤序列。
"""

import sys
import os
import json
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入要测试的函数
from airr_mcp_server import filter_by_vdj_genes


def test_filter_by_vdj_genes():
    """测试按 V/D/J 基因使用模式过滤序列"""
    
    # 设置测试参数
    # 注意：repertoire_id 需要是有效的 ID，可以先通过 search_airr_repertoires 获取
    # 这里使用一个示例 ID，实际使用时需要替换为有效的 ID
    repertoire_id = "6173719481891549676-242ac11c-0001-012"
    
    # 设置基因过滤条件
    v_gene = "IGHV3-53"  # V 基因家族或等位基因
    j_gene = "IGHJ6"     # J 基因家族或等位基因
    d_gene = None        # 不指定 D 基因
    combination_logic = "AND"  # 如何组合过滤器 - AND 或 OR
    
    # 调用函数
    result = filter_by_vdj_genes(
        repertoire_id=repertoire_id,
        v_gene=v_gene,
        d_gene=d_gene,
        j_gene=j_gene,
        combination_logic=combination_logic
    )
    
    # 打印结果
    print("\n===== filter_by_vdj_genes 测试结果 =====")
    print(f"状态: {result.get('status')}")
    
    # 如果过滤成功，打印详细信息
    if result.get('status') == 'success':
        print(f"过滤的序列数量: {result.get('filtered_sequences')}")
        print(f"总序列数量: {result.get('total_sequences')}")
        print(f"百分比: {result.get('percentage')}%")
        
        # 打印基因使用统计
        gene_stats = result.get('gene_usage_stats', {})
        
        # 打印 V 基因分布
        v_dist = gene_stats.get('v_gene_distribution', {})
        if v_dist:
            print("\nV 基因分布 (前 5 个):")
            for i, (gene, count) in enumerate(v_dist.items()):
                if i < 5:
                    print(f"  {gene}: {count}")
                else:
                    break
        
        # 打印 J 基因分布
        j_dist = gene_stats.get('j_gene_distribution', {})
        if j_dist:
            print("\nJ 基因分布 (前 5 个):")
            for i, (gene, count) in enumerate(j_dist.items()):
                if i < 5:
                    print(f"  {gene}: {count}")
                else:
                    break
        
        # 打印过滤参数
        filter_params = result.get('filter_parameters', {})
        print("\n过滤参数:")
        for key, value in filter_params.items():
            print(f"  {key}: {value}")
        
        print(f"\n下载可用: {result.get('download_available')}")
        print(f"仓库: {result.get('repository')}")
    else:
        # 如果过滤失败，打印错误信息
        print(f"错误: {result.get('error')}")
        print(f"消息: {result.get('message')}")
    
    # 返回结果以便可能的进一步处理
    return result


if __name__ == "__main__":
    # 执行测试
    test_result = test_filter_by_vdj_genes()
    
    # 可选：将结果保存到文件
    # with open("filter_result.json", "w") as f:
    #     json.dump(test_result, f, indent=2)