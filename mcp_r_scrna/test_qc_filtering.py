#!/usr/bin/env python3
"""
测试脚本：单细胞RNA-seq质量控制和细胞过滤功能测试

该脚本测试 run_qc_filtering 方法，用于对单细胞RNA-seq数据进行质量控制和细胞过滤。
从生信角度，质量控制是单细胞分析的关键第一步，需要过滤低质量细胞和基因。

测试数据：D:\data\test_data_20251001\Age_Bcells.rds
这是一个包含B细胞的Seurat对象，适合进行质量控制分析。
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

# 添加当前目录到Python路径，以便导入scrna_mcp_server模块
sys.path.insert(0, str(Path(__file__).parent))

# 导入需要测试的函数和依赖
from scrna_mcp_server import run_qc_filtering, load_config


def print_separator(title: str):
    """打印分隔符和标题"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def validate_input_file(file_path: str) -> bool:
    """验证输入文件是否存在且为RDS格式"""
    if not os.path.exists(file_path):
        print(f"❌ 错误：输入文件不存在 - {file_path}")
        return False
    
    if not file_path.lower().endswith('.rds'):
        print(f"⚠️  警告：输入文件不是RDS格式 - {file_path}")
    
    file_size = os.path.getsize(file_path)
    print(f"✅ 输入文件验证通过:")
    print(f"   文件路径: {file_path}")
    print(f"   文件大小: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    return True


def print_qc_parameters(params: Dict[str, Any]):
    """打印质量控制参数"""
    print_separator("质量控制参数设置")
    print("从生信角度，以下参数用于过滤低质量细胞：")
    print(f"• 最小基因数 (min_genes): {params['min_genes']}")
    print(f"  - 过滤表达基因数过少的细胞（可能是空液滴或破损细胞）")
    print(f"• 最大基因数 (max_genes): {params['max_genes']}")
    print(f"  - 过滤表达基因数过多的细胞（可能是双细胞）")
    print(f"• 最小UMI计数 (min_counts): {params['min_counts']}")
    print(f"  - 过滤总UMI计数过低的细胞（测序深度不足）")
    print(f"• 最大线粒体基因比例 (mt_percent): {params['mt_percent']}%")
    print(f"  - 过滤线粒体基因比例过高的细胞（可能是凋亡细胞）")


def analyze_results(result: Dict[str, Any]):
    """分析和展示结果"""
    print_separator("分析结果")
    
    status = result.get("status", "unknown")
    print(f"执行状态: {status}")
    
    if status == "success":
        print("✅ 质量控制分析成功完成！")
        
        # 输出目录信息
        output_dir = result.get("output_directory", "")
        if output_dir:
            print(f"\n📁 输出目录: {output_dir}")
        
        # 生成文件信息
        generated_files = result.get("generated_files", [])
        file_count = result.get("file_count", 0)
        print(f"\n📊 生成文件数量: {file_count}")
        
        if generated_files:
            print("\n生成的文件列表:")
            for i, file_path in enumerate(generated_files, 1):
                file_name = os.path.basename(file_path)
                file_ext = os.path.splitext(file_name)[1].lower()
                
                # 根据文件类型添加说明
                if file_ext == '.rds':
                    print(f"  {i}. {file_name} (过滤后的Seurat对象)")
                elif file_ext == '.csv':
                    print(f"  {i}. {file_name} (质量控制统计表)")
                elif file_ext in ['.pdf', '.png']:
                    print(f"  {i}. {file_name} (质量控制可视化图)")
                elif file_ext == '.txt':
                    print(f"  {i}. {file_name} (分析日志)")
                else:
                    print(f"  {i}. {file_name}")
        
        # R脚本输出信息
        stdout = result.get("stdout", "")
        if stdout:
            print(f"\n📝 R脚本输出信息:")
            print(stdout)
            
    elif status == "error":
        print("❌ 质量控制分析失败！")
        error_msg = result.get("message", "未知错误")
        print(f"错误信息: {error_msg}")
        
        # 显示详细错误信息
        stderr = result.get("stderr", "")
        if stderr:
            print(f"\n标准错误输出:")
            print(stderr)
            
        stdout = result.get("stdout", "")
        if stdout:
            print(f"\n标准输出:")
            print(stdout)
    
    else:
        print(f"⚠️  未知状态: {status}")


def test_qc_filtering_with_default_params(input_rds: str):
    """使用默认参数测试质量控制"""
    print_separator("测试1: 使用默认参数进行质量控制")
    
    # 默认参数（适合大多数单细胞数据）
    params = {
        "min_genes": 200,
        "max_genes": 6000, 
        "min_counts": 1000,
        "mt_percent": 20.0
    }
    
    print_qc_parameters(params)
    
    print("\n🔄 开始执行质量控制分析...")
    result = run_qc_filtering(
        input_rds=input_rds,
        min_genes=params["min_genes"],
        max_genes=params["max_genes"],
        min_counts=params["min_counts"],
        mt_percent=params["mt_percent"]
    )
    
    analyze_results(result)
    return result


def test_qc_filtering_with_strict_params(input_rds: str):
    """使用严格参数测试质量控制"""
    print_separator("测试2: 使用严格参数进行质量控制")
    
    # 严格参数（用于高质量数据或需要更严格过滤）
    params = {
        "min_genes": 500,
        "max_genes": 4000,
        "min_counts": 2000,
        "mt_percent": 15.0
    }
    
    print_qc_parameters(params)
    print("\n注意：严格参数会过滤更多细胞，适用于高质量数据集")
    
    print("\n🔄 开始执行严格质量控制分析...")
    result = run_qc_filtering(
        input_rds=input_rds,
        min_genes=params["min_genes"],
        max_genes=params["max_genes"],
        min_counts=params["min_counts"],
        mt_percent=params["mt_percent"]
    )
    
    analyze_results(result)
    return result


def test_qc_filtering_with_lenient_params(input_rds: str):
    """使用宽松参数测试质量控制"""
    print_separator("测试3: 使用宽松参数进行质量控制")
    
    # 宽松参数（保留更多细胞，适用于珍贵样本）
    params = {
        "min_genes": 100,
        "max_genes": 8000,
        "min_counts": 500,
        "mt_percent": 25.0
    }
    
    print_qc_parameters(params)
    print("\n注意：宽松参数会保留更多细胞，适用于细胞数量有限的珍贵样本")
    
    print("\n🔄 开始执行宽松质量控制分析...")
    result = run_qc_filtering(
        input_rds=input_rds,
        min_genes=params["min_genes"],
        max_genes=params["max_genes"],
        min_counts=params["min_counts"],
        mt_percent=params["mt_percent"]
    )
    
    analyze_results(result)
    return result


def main():
    """主测试函数"""
    print_separator("单细胞RNA-seq质量控制功能测试")
    print("测试目标：run_qc_filtering 方法")
    print("数据类型：B细胞单细胞RNA-seq数据")
    print("生信意义：质量控制是单细胞分析流程的关键第一步")
    
    # 输入文件路径
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    # 验证输入文件
    if not validate_input_file(input_rds):
        print("\n❌ 测试终止：输入文件验证失败")
        return
    
    # 显示配置信息
    print_separator("系统配置信息")
    try:
        config = load_config()
        print("当前配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"⚠️  无法加载配置: {e}")
    
    # 执行测试
    results = []
    
    try:
        # 测试1：默认参数
        result1 = test_qc_filtering_with_default_params(input_rds)
        results.append(("默认参数", result1))
        
        # 测试2：严格参数
        result2 = test_qc_filtering_with_strict_params(input_rds)
        results.append(("严格参数", result2))
        
        # 测试3：宽松参数
        result3 = test_qc_filtering_with_lenient_params(input_rds)
        results.append(("宽松参数", result3))
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试总结
    print_separator("测试总结")
    success_count = 0
    for test_name, result in results:
        status = result.get("status", "unknown")
        if status == "success":
            success_count += 1
            print(f"✅ {test_name}: 成功")
        else:
            print(f"❌ {test_name}: 失败 - {result.get('message', '未知错误')}")
    
    print(f"\n📊 测试结果统计:")
    print(f"   总测试数: {len(results)}")
    print(f"   成功数: {success_count}")
    print(f"   失败数: {len(results) - success_count}")
    print(f"   成功率: {success_count/len(results)*100:.1f}%")
    
    if success_count == len(results):
        print("\n🎉 所有测试通过！质量控制功能正常工作。")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息和系统配置。")
    
    print("\n💡 生信建议:")
    print("   1. 根据数据特点选择合适的质量控制参数")
    print("   2. 检查生成的QC图表，评估过滤效果")
    print("   3. 记录过滤前后的细胞数量变化")
    print("   4. 考虑样本类型和实验目的调整参数")


if __name__ == "__main__":
    main()