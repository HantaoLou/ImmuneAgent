#!/usr/bin/env python3
"""
测试 export_annotations 方法
使用真实的单细胞RNA测序数据进行注释导出测试

作者: 生信分析专家
日期: 2025年1月
"""

import os
import sys
import json
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent.parent  # 获取mcp_r_annotation目录
sys.path.insert(0, str(current_dir))

# 导入目标模块
from annotation_mcp_server import export_annotations, load_config

def print_separator(title="", char="=", width=80):
    """打印分隔线"""
    if title:
        title_line = f" {title} "
        padding = (width - len(title_line)) // 2
        line = char * padding + title_line + char * padding
        if len(line) < width:
            line += char
    else:
        line = char * width
    print(line)

def print_result_summary(result):
    """打印结果摘要"""
    print_separator("执行结果摘要")
    print(f"状态: {result.get('status', 'unknown')}")
    print(f"消息: {result.get('message', 'N/A')}")
    
    if 'generated_files' in result and result['generated_files']:
        print(f"生成文件数量: {len(result['generated_files'])}")
        print("生成的文件:")
        for file_path in result['generated_files']:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"  - {file_path} ({file_size:.2f} KB)")
            else:
                print(f"  - {file_path} (文件不存在)")
    
    if 'export_summary' in result and result['export_summary']:
        print("导出摘要:")
        summary = result['export_summary']
        if isinstance(summary, dict):
            for key, value in summary.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {summary}")

def validate_input_file(file_path):
    """验证输入文件"""
    print_separator("输入文件验证")
    
    if not os.path.exists(file_path):
        print(f"❌ 错误: 输入文件不存在: {file_path}")
        return False
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"✅ 输入文件存在: {file_path}")
    print(f"📊 文件大小: {file_size_mb:.2f} MB")
    
    if file_size_mb < 1:
        print("⚠️  警告: 文件大小较小，可能不是完整的单细胞数据")
    
    return True

def test_multiple_formats(input_rds, annotation_columns):
    """测试多种导出格式"""
    formats_to_test = ["csv", "tsv"]  # 先测试基础格式，避免依赖问题
    
    results = {}
    
    for export_format in formats_to_test:
        print_separator(f"测试 {export_format.upper()} 格式导出")
        
        try:
            result = export_annotations(
                input_rds=input_rds,
                annotation_columns=annotation_columns,
                export_format=export_format,
                include_umap=True
            )
            
            results[export_format] = result
            print(f"✅ {export_format.upper()} 格式导出: {result.get('status', 'unknown')}")
            
        except Exception as e:
            print(f"❌ {export_format.upper()} 格式导出失败: {str(e)}")
            results[export_format] = {"status": "error", "message": str(e)}
    
    return results

if __name__ == "__main__":
    print_separator("Export Annotations 测试", "=", 80)
    print("🧬 单细胞RNA测序数据注释导出测试")
    print(f"🐍 Python版本: {sys.version}")
    print(f"📁 工作目录: {os.getcwd()}")
    print()
    
    # 测试参数配置
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    # 科学严谨的注释列选择 (B细胞相关)
    annotation_columns = [
        "seurat_clusters",      # 聚类结果
        "orig.ident"           # 样本标识
    ]
    
    try:
        # 验证输入文件
        if not validate_input_file(input_rds):
            sys.exit(1)
        
        # 加载配置
        print_separator("配置加载")
        try:
            config = load_config()
            print("✅ 配置加载成功")
        except Exception as e:
            print(f"⚠️  配置加载警告: {e}")
            print("继续使用默认配置...")
        
        # 打印测试参数
        print_separator("测试参数")
        print(f"输入文件: {input_rds}")
        print(f"注释列: {', '.join(annotation_columns)}")
        print(f"包含UMAP坐标: True")
        print()
        
        # 测试1: 基础CSV导出
        print_separator("测试1: CSV格式导出")
        print("🔬 正在导出注释到CSV格式...")
        print("📊 导出内容:")
        print("  - 细胞条形码")
        print("  - 聚类注释")
        print("  - 样本标识")
        print("  - UMAP坐标")
        print()
        
        result_csv = export_annotations(
            input_rds=input_rds,
            annotation_columns=annotation_columns,
            export_format="csv",
            include_umap=True
        )
        
        print_result_summary(result_csv)
        
        # 测试2: TSV格式导出
        print_separator("测试2: TSV格式导出")
        print("🔬 正在导出注释到TSV格式...")
        
        result_tsv = export_annotations(
            input_rds=input_rds,
            annotation_columns=annotation_columns,
            export_format="tsv",
            include_umap=True
        )
        
        print_result_summary(result_tsv)
        
        # 测试3: 自动检测注释列
        print_separator("测试3: 自动检测注释列")
        print("🔬 让系统自动检测注释相关列...")
        
        result_auto = export_annotations(
            input_rds=input_rds,
            annotation_columns=None,  # 自动检测
            export_format="csv",
            include_umap=False  # 不包含UMAP
        )
        
        print_result_summary(result_auto)
        
        # 测试4: 不包含UMAP坐标
        print_separator("测试4: 不包含UMAP坐标")
        print("🔬 导出注释但不包含UMAP坐标...")
        
        result_no_umap = export_annotations(
            input_rds=input_rds,
            annotation_columns=annotation_columns,
            export_format="csv",
            include_umap=False
        )
        
        print_result_summary(result_no_umap)
        
        # 综合结果验证
        print_separator("综合结果验证")
        
        all_tests = [
            ("CSV导出", result_csv),
            ("TSV导出", result_tsv),
            ("自动检测", result_auto),
            ("无UMAP", result_no_umap)
        ]
        
        success_count = 0
        for test_name, result in all_tests:
            if result.get('status') == 'success':
                print(f"✅ {test_name}: 成功")
                success_count += 1
            else:
                print(f"❌ {test_name}: 失败 - {result.get('message', 'Unknown error')}")
        
        print(f"\n📊 测试总结: {success_count}/{len(all_tests)} 个测试通过")
        
        if success_count == len(all_tests):
            print("🎉 所有测试均通过!")
        elif success_count > 0:
            print("⚠️  部分测试通过，请检查失败的测试")
        else:
            print("❌ 所有测试均失败，请检查配置和数据")
        
        print_separator("测试完成")
        
    except Exception as e:
        print_separator("测试异常")
        print(f"❌ 测试过程中发生异常: {str(e)}")
        print(f"异常类型: {type(e).__name__}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        sys.exit(1)