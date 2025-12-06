"""
测试 download_airr_sequences 工具

此测试文件用于测试 download_airr_sequences 工具的功能，
该工具用于从特定库下载 BCR 序列。
"""

import sys
import os
import json
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入要测试的函数
from airr_mcp_server import download_airr_sequences


def test_download_airr_sequences():
    """测试从特定库下载 BCR 序列"""
    
    # 设置测试参数
    # 注意：repertoire_id 需要是有效的 ID，可以先通过 search_airr_repertoires 获取
    # 这里使用从test_search_airr_repertoires获取的有效ID
    repertoire_id = "2"  # 从iReceptor API获取的有效repertoire_id
    
    # 设置过滤条件
    filters = {
        "v_call": "IGHV3-23",  # V 基因过滤器
        "productive": True  # 只下载有效序列
    }
    
    # 创建输出目录
    output_dir = str(Path(__file__).parent.parent / "output")
    print(f"\n使用自定义输出目录: {output_dir}")
    
    # 调用函数
    result = download_airr_sequences(
        repertoire_id=repertoire_id,
        filters=filters,
        format="airr",  # 输出格式：airr (TSV) 或 json
        max_sequences=100,  # 最大序列数量
        output_dir=output_dir  # 指定输出目录
    )
    
    # 打印结果
    print("\n===== download_airr_sequences 测试结果 =====")
    print(f"状态: {result.get('status')}")
    
    # 如果下载成功，打印详细信息
    if result.get('status') == 'success':
        print(f"文件路径: {result.get('file_path')}")
        print(f"格式: {result.get('format')}")
        print(f"下载的序列数量: {result.get('sequences_downloaded')}")
        print(f"仓库: {result.get('repository')}")
        print(f"与 IgBLAST 兼容: {result.get('compatible_with_igblast')}")
        print(f"应用的过滤条件: {result.get('filters_applied')}")
        
        # 可选：打印文件的前几行
        file_path = result.get('file_path')
        if file_path and os.path.exists(file_path):
            print("\n文件内容预览 (前 5 行):")
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i < 5:
                        print(line.strip())
                    else:
                        break
    else:
        # 如果下载失败，打印错误信息
        print(f"错误: {result.get('error')}")
        print(f"消息: {result.get('message')}")
    
    # 返回结果以便可能的进一步处理
    return result


if __name__ == "__main__":
    # 执行测试
    test_result = test_download_airr_sequences()
    
    # 可选：将结果保存到文件
    # with open("download_result.json", "w") as f:
    #     json.dump(test_result, f, indent=2)