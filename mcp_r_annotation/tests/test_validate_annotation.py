#!/usr/bin/env python3
"""
测试 validate_annotation 方法

这个测试脚本用于验证 annotation_mcp_server.py 中的 validate_annotation 方法
使用真实的单细胞RNA测序数据进行测试，确保方法的科学严谨性
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent.parent  # 获取mcp_r_annotation目录
sys.path.insert(0, str(current_dir))

# 导入需要测试的模块
from annotation_mcp_server import validate_annotation, load_config

def print_separator(title=""):
    """打印分隔线"""
    print("=" * 60)
    if title:
        print(f"{title}")
        print("=" * 60)

def print_result_summary(result):
    """打印结果摘要"""
    print(f"状态: {result.get('status', 'unknown')}")
    print(f"消息: {result.get('message', 'No message')}")
    
    if 'generated_files' in result:
        print(f"生成的文件数量: {len(result['generated_files'])}")
        if result['generated_files']:
            print("生成的文件:")
            for file_path in result['generated_files']:
                file_size = "未知大小"
                if os.path.exists(file_path):
                    size_bytes = os.path.getsize(file_path)
                    if size_bytes < 1024:
                        file_size = f"{size_bytes} B"
                    elif size_bytes < 1024 * 1024:
                        file_size = f"{size_bytes / 1024:.2f} KB"
                    else:
                        file_size = f"{size_bytes / (1024 * 1024):.2f} MB"
                print(f"  - {file_path} ({file_size})")
    
    if 'output_dir' in result:
        print(f"输出目录: {result['output_dir']}")

def test_validate_annotation():
    """测试 validate_annotation 方法"""
    
    print_separator("Validate Annotation 测试脚本")
    
    # 测试参数
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    annotation_column1 = "seurat_clusters"  # 使用聚类结果作为第一个注释列
    reference_dataset = "MonacoImmuneData"  # 使用Monaco免疫数据作为参考
    
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"脚本位置: {__file__}")
    print()
    
    # 检查输入文件
    print_separator("输入文件检查")
    print(f"检查输入文件: {input_rds}")
    if os.path.exists(input_rds):
        file_size = os.path.getsize(input_rds)
        print("✅ 输入文件存在")
        print(f"   文件大小: {file_size / (1024 * 1024):.2f} MB")
    else:
        print("❌ 输入文件不存在")
        return
    
    # 检查配置
    print_separator("配置检查")
    try:
        config = load_config()
        print("✅ 配置加载成功")
        print(f"   基础目录: {config.get('base_dir', 'Not set')}")
        print(f"   脚本目录: {config.get('scripts_dir', 'Not set')}")
        print(f"   输出目录: {config.get('output_dir', 'Not set')}")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return
    
    # 测试参数显示
    print_separator("测试参数")
    print(f"  输入文件: {input_rds}")
    print(f"  注释列1: {annotation_column1}")
    print(f"  注释列2: None (将使用SingleR进行验证)")
    print(f"  参考数据集: {reference_dataset}")
    print()
    
    # 执行测试
    print_separator("开始执行 Validate Annotation")
    print("🚀 开始执行 Validate Annotation...")
    print("注意: 这可能需要几分钟时间，请耐心等待...")
    print()
    
    try:
        # 调用 validate_annotation 方法
        result = validate_annotation(
            input_rds=input_rds,
            annotation_column1=annotation_column1,
            annotation_column2=None,  # 不提供第二个注释列，使用SingleR验证
            reference_dataset=reference_dataset
        )
        
        print_separator("执行结果")
        print_result_summary(result)
        
        # 详细输出信息
        if 'stdout' in result and result['stdout']:
            print_separator("R脚本输出信息")
            print(result['stdout'])
        
        # 错误信息
        if 'stderr' in result and result['stderr']:
            print_separator("错误信息")
            print(result['stderr'])
        
        # 成功或失败标识
        if result.get('status') == 'success':
            print_separator("✅ 测试完成 - 成功!")
        else:
            print_separator("❌ 测试完成 - 失败!")
            
    except Exception as e:
        print_separator("❌ 测试执行异常")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {str(e)}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()

def main():
    """主函数"""
    print("Validate Annotation 测试脚本")
    print("=" * 60)
    
    # 执行测试
    test_validate_annotation()

if __name__ == "__main__":
    main()