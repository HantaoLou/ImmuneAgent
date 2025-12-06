#!/usr/bin/env python3
"""
Cluster Markers Detection Test Script

测试 detect_cluster_markers 方法的功能
使用真实的单细胞数据进行聚类标记基因检测测试
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent.parent  # 获取mcp_r_annotation目录
sys.path.insert(0, str(current_dir))

# 导入annotation_mcp_server模块
from annotation_mcp_server import detect_cluster_markers


def test_detect_cluster_markers():
    """
    测试 detect_cluster_markers 方法
    
    使用指定的输入文件进行聚类标记基因检测
    """
    print("=" * 60)
    print("Cluster Markers Detection 测试脚本")
    print("=" * 60)
    
    # 输入文件路径
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    # 检查输入文件是否存在
    print(f"检查输入文件: {input_rds}")
    if not os.path.exists(input_rds):
        print(f"❌ 输入文件不存在: {input_rds}")
        return False
    
    file_size = os.path.getsize(input_rds) / (1024 * 1024)  # MB
    print(f"✅ 输入文件存在")
    print(f"   文件大小: {file_size:.2f} MB")
    print()
    
    # 测试参数
    test_params = {
        "test_use": "wilcox",           # Wilcoxon rank sum test
        "only_pos": True,               # 只返回正向标记基因
        "min_pct": 0.25,               # 最小表达百分比
        "logfc_threshold": 0.5,        # 最小log2倍数变化阈值
        "top_n": 10                    # 每个聚类返回的顶级标记基因数量
    }
    
    print("-" * 40)
    print("测试参数:")
    print(f"  输入文件: {input_rds}")
    print(f"  统计检验方法: {test_params['test_use']}")
    print(f"  只返回正向标记: {test_params['only_pos']}")
    print(f"  最小表达百分比: {test_params['min_pct']}")
    print(f"  Log2FC阈值: {test_params['logfc_threshold']}")
    print(f"  每聚类顶级基因数: {test_params['top_n']}")
    print("-" * 40)
    print()
    
    try:
        print("🚀 开始执行 Cluster Markers Detection...")
        print("注意: 这可能需要几分钟时间，请耐心等待...")
        print()
        
        # 调用 detect_cluster_markers 方法
        result = detect_cluster_markers(
            input_rds=input_rds,
            test_use=test_params["test_use"],
            only_pos=test_params["only_pos"],
            min_pct=test_params["min_pct"],
            logfc_threshold=test_params["logfc_threshold"],
            top_n=test_params["top_n"]
        )
        
        print("=" * 60)
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
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path) / 1024  # KB
                        print(f"  {i}. {file_path} ({file_size:.1f} KB)")
                    else:
                        print(f"  {i}. {file_path} (文件不存在)")
            
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
            
        else:
            print("❌ 执行失败!")
            
            # 打印错误消息
            message = result.get("message", "未知错误")
            print(f"错误消息: {message}")
            
            # 打印错误详情
            stderr = result.get("stderr", "")
            if stderr:
                print(f"\n🔍 错误详情:")
                print("-" * 30)
                print(stderr)
                print("-" * 30)
            
            # 打印标准输出（可能包含有用信息）
            stdout = result.get("stdout", "")
            if stdout:
                print(f"\n📋 输出信息:")
                print("-" * 30)
                print(stdout)
                print("-" * 30)
        
        print()
        return status == "success"
        
    except Exception as e:
        print(f"❌ 测试过程中发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("Cluster Markers Detection 测试脚本")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"脚本位置: {__file__}")
    print()
    
    # 执行测试
    success = test_detect_cluster_markers()
    
    print("=" * 60)
    if success:
        print("🎉 测试完成 - 成功!")
        return 0
    else:
        print("💥 测试完成 - 失败!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)