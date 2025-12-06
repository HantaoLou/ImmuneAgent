#!/usr/bin/env python3
"""
SingleR Annotation Test Script

测试 run_singler_annotation 方法的独立测试脚本
使用真实数据进行测试，确保生信分析的科学严谨性
"""

import os
import sys
import json
from pathlib import Path

# 添加当前目录到Python路径，以便导入annotation_mcp_server模块
current_dir = Path(__file__).parent.parent  # 获取mcp_r_annotation目录
sys.path.insert(0, str(current_dir))

from annotation_mcp_server import run_singler_annotation


def test_singler_annotation():
    """
    测试 SingleR 细胞类型注释功能
    
    使用指定的测试数据文件进行真实的生信分析测试
    """
    print("=" * 60)
    print("SingleR 细胞类型注释测试")
    print("=" * 60)
    
    # 测试参数
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    # 验证输入文件是否存在
    print(f"检查输入文件: {input_rds}")
    if not os.path.exists(input_rds):
        print(f"❌ 错误: 输入文件不存在: {input_rds}")
        return False
    else:
        print(f"✅ 输入文件存在")
        # 显示文件大小
        file_size = os.path.getsize(input_rds)
        print(f"   文件大小: {file_size / (1024*1024):.2f} MB")
    
    print("\n" + "-" * 40)
    print("测试参数:")
    print(f"  输入文件: {input_rds}")
    print(f"  参考数据集: HumanPrimaryCellAtlasData (默认)")
    print(f"  标签类型: label.main (默认)")
    print(f"  聚类列: seurat_clusters (默认)")
    print("-" * 40)
    
    try:
        print("\n🚀 开始执行 SingleR 注释...")
        print("注意: 这可能需要几分钟时间，请耐心等待...")
        
        # 调用 run_singler_annotation 方法
        result = run_singler_annotation(
            input_rds=input_rds,
            reference_dataset="HumanPrimaryCellAtlasData",
            label_type="label.main",
            cluster_column="seurat_clusters"
        )
        
        print("\n" + "=" * 60)
        print("执行结果:")
        print("=" * 60)
        
        # 打印执行状态
        status = result.get("status", "unknown")
        print(f"状态: {status}")
        
        if status == "success":
            print("✅ 执行成功!")
            
            # 打印消息
            message = result.get("message", "")
            if message:
                print(f"消息: {message}")
            
            # 打印生成的文件
            generated_files = result.get("generated_files", [])
            if generated_files:
                print(f"\n📁 生成的文件 ({len(generated_files)} 个):")
                for i, file_path in enumerate(generated_files, 1):
                    print(f"  {i}. {file_path}")
                    # 检查文件是否真实存在
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        print(f"     ✅ 文件存在 ({file_size} bytes)")
                    else:
                        print(f"     ❌ 文件不存在")
            else:
                print("📁 未生成文件")
            
            # 打印输出目录
            output_dir = result.get("output_dir", "")
            if output_dir:
                print(f"\n📂 输出目录: {output_dir}")
            
            # 打印标准输出（如果有）
            stdout = result.get("stdout", "")
            if stdout:
                print(f"\n📋 R脚本输出:")
                print("-" * 30)
                print(stdout)
                print("-" * 30)
            
            # 打印注释摘要（如果有）
            annotation_summary = result.get("annotation_summary", {})
            if annotation_summary:
                print(f"\n📊 注释摘要:")
                print(json.dumps(annotation_summary, indent=2, ensure_ascii=False))
            
        else:
            print("❌ 执行失败!")
            
            # 打印错误消息
            message = result.get("message", "")
            if message:
                print(f"错误消息: {message}")
            
            # 打印标准错误输出（如果有）
            stderr = result.get("stderr", "")
            if stderr:
                print(f"\n🔍 错误详情:")
                print("-" * 30)
                print(stderr)
                print("-" * 30)
            
            # 打印标准输出（可能包含有用的调试信息）
            stdout = result.get("stdout", "")
            if stdout:
                print(f"\n📋 输出信息:")
                print("-" * 30)
                print(stdout)
                print("-" * 30)
        
        return status == "success"
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生异常: {str(e)}")
        print(f"异常类型: {type(e).__name__}")
        import traceback
        print("\n🔍 详细错误信息:")
        traceback.print_exc()
        return False


def main():
    """
    主函数 - 执行测试
    """
    print("SingleR 细胞类型注释测试脚本")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"脚本位置: {__file__}")
    
    # 执行测试
    success = test_singler_annotation()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 测试完成 - 成功!")
        sys.exit(0)
    else:
        print("💥 测试完成 - 失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()