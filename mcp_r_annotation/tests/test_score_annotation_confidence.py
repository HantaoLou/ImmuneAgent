#!/usr/bin/env python3
"""
测试 score_annotation_confidence 方法
使用真实的单细胞RNA测序数据进行注释置信度评分测试

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
from annotation_mcp_server import score_annotation_confidence, load_config

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
    
    if 'confidence_summary' in result and result['confidence_summary']:
        print("置信度摘要:")
        summary = result['confidence_summary']
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

if __name__ == "__main__":
    print_separator("Score Annotation Confidence 测试", "=", 80)
    print("🧬 单细胞RNA测序数据注释置信度评分测试")
    print(f"🐍 Python版本: {sys.version}")
    print(f"📁 工作目录: {os.getcwd()}")
    print()
    
    # 测试参数配置
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    annotation_column = "seurat_clusters"  # 使用聚类结果作为注释列
    
    # B细胞相关的标记基因 (科学严谨的基因选择)
    marker_genes = {
        "B cells": ["CD79A", "CD79B", "MS4A1", "CD19"],  # 经典B细胞标记
        "Plasma cells": ["IGHG1", "IGHG3", "JCHAIN", "XBP1"],  # 浆细胞标记
        "Memory B cells": ["CD27", "IGHD", "IGHM"],  # 记忆B细胞标记
        "Naive B cells": ["IGHD", "IGHM", "TCL1A"],  # 初始B细胞标记
        "Germinal center B cells": ["BCL6", "AICDA", "MEF2B"]  # 生发中心B细胞标记
    }
    
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
        print(f"注释列: {annotation_column}")
        print(f"标记基因组数: {len(marker_genes)}")
        print("标记基因:")
        for cell_type, genes in marker_genes.items():
            print(f"  {cell_type}: {', '.join(genes)}")
        print()
        
        # 执行测试
        print_separator("开始执行 Score Annotation Confidence")
        print("🔬 正在计算注释置信度评分...")
        print("📊 分析内容:")
        print("  - SingleR置信度评分")
        print("  - 标记基因表达一致性")
        print("  - 聚类同质性分析")
        print("  - 轮廓系数计算")
        print()
        
        # 调用目标方法
        result = score_annotation_confidence(
            input_rds=input_rds,
            annotation_column=annotation_column,
            marker_genes=marker_genes
        )
        
        # 打印详细结果
        print_result_summary(result)
        
        # 验证结果
        print_separator("结果验证")
        if result.get('status') == 'success':
            print("✅ 测试执行成功!")
            
            # 检查生成的文件
            if 'generated_files' in result and result['generated_files']:
                print("📁 生成文件验证:")
                for file_path in result['generated_files']:
                    if os.path.exists(file_path):
                        print(f"  ✅ {os.path.basename(file_path)}")
                    else:
                        print(f"  ❌ {os.path.basename(file_path)} (缺失)")
            
            # 检查置信度摘要
            if 'confidence_summary' in result:
                print("📊 置信度分析完成")
            
        else:
            print("❌ 测试执行失败!")
            if 'message' in result:
                print(f"错误信息: {result['message']}")
        
        print_separator("测试完成")
        
    except Exception as e:
        print_separator("测试异常")
        print(f"❌ 测试过程中发生异常: {str(e)}")
        print(f"异常类型: {type(e).__name__}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        sys.exit(1)