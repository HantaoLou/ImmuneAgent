"""
测试 get_airr_study_metadata 工具

此测试文件用于测试 get_airr_study_metadata 工具的功能，
该工具用于获取研究和样本的详细元数据。
"""

import sys
import os
import json
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入要测试的函数
from airr_mcp_server import get_airr_study_metadata


def test_get_airr_study_metadata():
    """测试获取研究和样本的详细元数据"""
    
    # 设置测试参数
    # 注意：study_id 需要是有效的 ID，这里使用从test_search_airr_repertoires获取的有效ID
    study_id = "SRP001460"  # 从iReceptor API获取的有效研究ID
    repository = "ireceptor"  # 指定使用ireceptor仓库，因为我们知道这个ID在该仓库中有效
    
    # 调用函数
    result = get_airr_study_metadata(
        study_id=study_id,
        repository=repository
    )
    
    # 打印结果
    print("\n===== get_airr_study_metadata 测试结果 =====")
    print(f"状态: {result.get('status')}")
    
    # 如果获取成功，打印详细信息
    if result.get('status') == 'success':
        study = result.get('study', {})
        print("\n研究信息:")
        print(f"  研究 ID: {study.get('study_id')}")
        print(f"  研究标题: {study.get('study_title')}")
        print(f"  研究类型: {study.get('study_type')}")
        print(f"  研究描述: {study.get('study_description')}")
        print(f"  关键词: {study.get('keywords')}")
        print(f"  发布 ID: {study.get('pub_ids')}")
        print(f"  受试者数量: {study.get('subjects')}")
        print(f"  样本数量: {study.get('samples')}")
        print(f"  库数量: {study.get('repertoires')}")
        
        # 打印样本信息
        samples = result.get('samples', [])
        print(f"\n样本数量: {len(samples)}")
        if samples:
            print("\n前 3 个样本信息:")
            for i, sample in enumerate(samples[:3]):
                print(f"  样本 {i+1}:")
                for key, value in sample.items():
                    print(f"    {key}: {value}")
        
        print(f"\n仓库: {result.get('repository')}")
    else:
        # 如果获取失败，打印错误信息
        print(f"错误: {result.get('error')}")
        print(f"消息: {result.get('message')}")
    
    # 返回结果以便可能的进一步处理
    return result


if __name__ == "__main__":
    # 执行测试
    test_result = test_get_airr_study_metadata()
    
    # 可选：将结果保存到文件
    # with open("study_metadata_result.json", "w") as f:
    #     json.dump(test_result, f, indent=2)