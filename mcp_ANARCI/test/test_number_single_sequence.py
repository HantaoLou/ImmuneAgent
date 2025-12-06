#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试anarci_mcp_server.py中的number_single_sequence方法
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入要测试的函数
from anarci_mcp_server import number_single_sequence

def main():
    """测试number_single_sequence函数的功能"""
    
    # 测试用例1: 有效的抗体序列 (重链)
    test_antibody_heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTTVTVSS"
    
    # 测试用例2: 有效的抗体序列 (轻链)
    test_antibody_light = "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKRLIYAASSLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCLQQNSDPPTFGQGTKVEIK"
    
    # 测试用例3: 非抗体序列
    test_non_antibody = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
    
    # 测试不同的编号方案
    schemes = ["imgt", "chothia", "kabat", "martin", "aho", "wolfguy"]
    
    print("测试 number_single_sequence 函数\n")
    
    # 测试抗体重链序列
    print("测试抗体重链序列:")
    for scheme in schemes:
        result = number_single_sequence(test_antibody_heavy, scheme=scheme)
        print(f"  使用 {scheme} 方案:")
        print(f"  状态: {result['status']}")
        if result['status'] == 'success':
            print(f"  链类型: {result['chain_type']}")
            print(f"  编号方案: {result['scheme']}")
            # 只打印部分编号结果作为示例
            numbering_sample = result['numbering'][:5] if len(result['numbering']) > 5 else result['numbering']
            print(f"  编号示例 (前5个): {numbering_sample}")
        else:
            print(f"  消息: {result.get('message', 'No message')}")
        print()
    
    # 测试抗体轻链序列
    print("测试抗体轻链序列:")
    result = number_single_sequence(test_antibody_light)
    print(f"  状态: {result['status']}")
    if result['status'] == 'success':
        print(f"  链类型: {result['chain_type']}")
        print(f"  编号方案: {result['scheme']}")
        # 只打印部分编号结果作为示例
        numbering_sample = result['numbering'][:5] if len(result['numbering']) > 5 else result['numbering']
        print(f"  编号示例 (前5个): {numbering_sample}")
    else:
        print(f"  消息: {result.get('message', 'No message')}")
    print()
    
    # 测试非抗体序列
    print("测试非抗体序列:")
    result = number_single_sequence(test_non_antibody)
    print(f"  状态: {result['status']}")
    if result['status'] == 'success':
        print(f"  链类型: {result['chain_type']}")
        print(f"  编号方案: {result['scheme']}")
    else:
        print(f"  消息: {result.get('message', 'No message')}")
    print()

if __name__ == "__main__":
    main()