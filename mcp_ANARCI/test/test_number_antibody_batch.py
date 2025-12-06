#!/usr/bin/env python
"""
测试ANARCI抗体编号功能

简单测试number_antibody_batch方法的功能
"""

import sys
import os
import json
from pathlib import Path

# 添加父目录到Python路径
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# 导入要测试的函数
from anarci_mcp_server import number_antibody_batch

def main():
    """测试number_antibody_batch方法"""
    print("开始测试number_antibody_batch方法...")
    
    # 测试数据 - 抗体序列示例
    test_sequences = [
        {
            "id": "test_antibody_1",
            "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGILGAGKAGTTVIVSS"
        },
        {
            "id": "test_antibody_2",
            "sequence": "DIVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGYNYLDWYLQKPGQSPQLLIYLGSNRASGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCMQALQTPYTFGQGTKLEIKR"
        },
        {
            "id": "test_non_antibody",
            "sequence": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
        }
    ]
    
    # 调用函数
    result = number_antibody_batch(test_sequences)
    
    # 打印结果
    print("\n测试结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 简单验证
    if result["status"] == "success":
        print("\n测试成功!")
        print(f"处理了 {result['total']} 个序列")
        print(f"处理时间: {result['processing_time_ms']:.2f} 毫秒")
        
        # 检查每个序列的结果
        for i, seq_result in enumerate(result["results"]):
            print(f"\n序列 {i+1} ({seq_result['id']}):")
            if seq_result["numbered"]:
                print("  成功编号: 是")
                print(f"  编号方案: {seq_result['scheme']}")
                print(f"  域数量: {len(seq_result['numbering'])}")
            else:
                print("  成功编号: 否")
                print(f"  原因: {seq_result['message']}")
    else:
        print("\n测试失败!")
        print(f"错误信息: {result['message']}")

if __name__ == "__main__":
    main()