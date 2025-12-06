#!/usr/bin/env python3
"""
简单测试文件 - 验证IgBLAST MCP服务器工具方法功能
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from igblast_mcp_server import analyze_vdj_batch, extract_cdr3_from_airr


def test_analyze_vdj_batch():
    """测试V(D)J分析功能"""
    print("=" * 50)
    print("测试 analyze_vdj_batch 功能")
    print("=" * 50)
    
    # 测试序列 - 使用真实的抗体重链序列
    test_sequences = [
        {
            "id": "test_seq_1",
            "sequence": "CAGGTGCAGCTGGTGGAGTCTGGGGGAGGCGTGGTCCAGCCTGGGAGGTCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCTTCAGTAGCTATGCAATGAGCTGGGTCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGGTCTCAGCTATTAGTGGTAGTGGTGGTAGCACATACTACGCAGACTCCGTGAAGGGCCGGTTCACCATCTCCAGAGACAATTCCAAGAACACGCTGTATCTGCAAATGAACAGCCTGAGAGCCGAGGACACGGCCGTATATTACTGTGCGAGAGA"
        },
        {
            "id": "test_seq_2", 
            "sequence": "GAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCTTCAGTAGCTACGCCATGAGCTGGGTCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGGTCTCAGCTATTAGTGGTAGTGGTGGTAGCACATACTACGCAGACTCCGTGAAGGGCCGGTTCACCATCTCCAGAGACAATTCCAAGAACACGCTGTATCTGCAAATGAACAGCCTGAGAGCCGAGGACACGGCCGTATATTACTGTGCGAGAGA"
        }
    ]
    
    # 调用分析函数
    result = analyze_vdj_batch(
        sequences=test_sequences,
        organism="human",
        receptor_type="Ig",
        locus="IGH"
    )
    
    # 输出结果
    print(f"状态: {result.get('status')}")
    print(f"总序列数: {result.get('total_sequences')}")
    print(f"处理序列数: {result.get('processed')}")
    print(f"格式: {result.get('format')}")
    print(f"处理时间: {result.get('processing_time_ms', 0):.2f} ms")
    
    if result.get('status') == 'error':
        print(f"错误信息: {result.get('message')}")
        print(f"错误类型: {result.get('error_type')}")
        if result.get('stderr'):
            print(f"标准错误: {result.get('stderr')}")
        return None
    
    # 显示部分结果
    results = result.get('results', [])
    if results:
        print(f"\n前{min(2, len(results))}个结果:")
        for i, res in enumerate(results[:2]):
            print(f"  序列 {i+1}:")
            print(f"    sequence_id: {res.get('sequence_id')}")
            print(f"    v_call: {res.get('v_call')}")
            print(f"    j_call: {res.get('j_call')}")
            print(f"    junction: {res.get('junction')}")
            print(f"    junction_aa: {res.get('junction_aa')}")
            print(f"    productive: {res.get('productive')}")
    
    return results


def test_extract_cdr3_from_airr(airr_results):
    """测试CDR3提取功能"""
    print("\n" + "=" * 50)
    print("测试 extract_cdr3_from_airr 功能")
    print("=" * 50)
    
    if not airr_results:
        print("没有AIRR结果可供测试CDR3提取")
        return
    
    # 调用CDR3提取函数
    result = extract_cdr3_from_airr(airr_results)
    
    # 输出结果
    print(f"状态: {result.get('status')}")
    print(f"CDR3总数: {result.get('total')}")
    
    if result.get('status') == 'error':
        print(f"错误信息: {result.get('message')}")
        return
    
    # 显示CDR3结果
    cdr3_results = result.get('cdr3_results', [])
    if cdr3_results:
        print(f"\nCDR3结果:")
        for i, cdr3 in enumerate(cdr3_results):
            print(f"  序列 {i+1}:")
            print(f"    sequence_id: {cdr3.get('sequence_id')}")
            print(f"    junction (CDR3核苷酸): {cdr3.get('junction')}")
            print(f"    junction_aa (CDR3氨基酸): {cdr3.get('junction_aa')}")
            print(f"    junction_length: {cdr3.get('junction_length')}")
            print(f"    productive: {cdr3.get('productive')}")
            print(f"    v_call: {cdr3.get('v_call')}")
            print(f"    j_call: {cdr3.get('j_call')}")


def main():
    """主测试函数"""
    print("开始测试IgBLAST MCP服务器工具方法...")
    
    try:
        # 测试V(D)J分析
        airr_results = test_analyze_vdj_batch()
        
        # 测试CDR3提取
        test_extract_cdr3_from_airr(airr_results)
        
        print("\n" + "=" * 50)
        print("测试完成!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()