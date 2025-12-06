#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCTransform标准化测试文件

测试run_normalization_sct方法的功能，包括：
- SCTransform方差稳定化和标准化
- 高变基因选择
- 技术协变量回归
- PCA降维分析
- 诊断图表生成

生信背景：
SCTransform是Seurat包中的一种先进标准化方法，相比传统的LogNormalize方法：
1. 更好地处理技术噪声和批次效应
2. 保留生物学变异的同时去除技术变异
3. 自动选择高变基因，无需手动设置阈值
4. 支持回归技术协变量（如线粒体基因比例、UMI数量等）

测试策略：
- 使用真实的单细胞数据进行测试
- 测试不同的参数组合
- 验证输出文件的完整性和科学性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrna_mcp_server import run_normalization_sct

def print_analysis_info(test_name, input_rds, vars_to_regress, n_variable_features):
    """打印分析信息"""
    print(f"\n{'='*60}")
    print(f" {test_name}")
    print(f"{'='*60}")
    print(f"📁 输入文件: {input_rds}")
    print(f"🧬 高变基因数量: {n_variable_features}")
    if vars_to_regress:
        print(f"🔧 回归变量: {', '.join(vars_to_regress)}")
    else:
        print(f"🔧 回归变量: 无")
    print(f"📊 标准化方法: SCTransform")
    print()

def analyze_results(result, test_name):
    """分析和打印结果"""
    print(f"============================================================")
    print(f" {test_name} - 分析结果")
    print(f"============================================================")
    
    status = result.get("status", "unknown")
    print(f"执行状态: {status}")
    
    if status == "success":
        print("✅ SCTransform标准化成功完成！")
        
        # 输出目录信息
        output_dir = result.get("output_dir", "")
        if output_dir:
            print(f"\n📁 输出目录: {output_dir}")
        
        # 生成文件信息
        generated_files = result.get("generated_files", [])
        print(f"\n📊 生成文件数量: {len(generated_files)}")
        
        if generated_files:
            print("\n生成的文件列表:")
            for i, file_path in enumerate(generated_files, 1):
                filename = os.path.basename(file_path)
                if filename.endswith('.rds'):
                    print(f"  {i}. {filename} (标准化后的Seurat对象)")
                elif filename.endswith('.csv'):
                    if 'variable_features' in filename:
                        print(f"  {i}. {filename} (高变基因列表)")
                    elif 'pca_embeddings' in filename:
                        print(f"  {i}. {filename} (PCA坐标)")
                    elif 'statistics' in filename:
                        print(f"  {i}. {filename} (标准化统计信息)")
                elif filename.endswith('.pdf'):
                    if 'elbow' in filename:
                        print(f"  {i}. {filename} (PCA肘部图)")
                    elif 'variable_features' in filename:
                        print(f"  {i}. {filename} (高变基因图)")
                    elif 'loadings' in filename:
                        print(f"  {i}. {filename} (PCA载荷图)")
                    elif 'heatmap' in filename:
                        print(f"  {i}. {filename} (PCA热图)")
                else:
                    print(f"  {i}. {filename}")
        
        # R脚本输出信息
        stdout = result.get("stdout", "")
        if stdout:
            print(f"\n📝 R脚本输出信息:")
            for line in stdout.strip().split('\n'):
                if line.strip():
                    print(f"✓ {line.strip()}")
    
    else:
        print("❌ SCTransform标准化失败！")
        error_msg = result.get("message", "未知错误")
        print(f"错误信息: {error_msg}")
        
        stderr = result.get("stderr", "")
        if stderr:
            print(f"错误详情: {stderr}")
    
    print()

def validate_input_file(input_rds):
    """验证输入文件"""
    print(f"🔍 验证输入文件: {input_rds}")
    
    if not os.path.exists(input_rds):
        print(f"❌ 错误: 输入文件不存在")
        return False
    
    if not input_rds.endswith('.rds'):
        print(f"❌ 错误: 输入文件必须是RDS格式")
        return False
    
    file_size = os.path.getsize(input_rds) / (1024 * 1024)  # MB
    print(f"✅ 文件验证通过，大小: {file_size:.1f} MB")
    return True

if __name__ == "__main__":
    print("🧬 SCTransform标准化测试开始")
    print("=" * 60)
    
    # 输入文件路径
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    # 验证输入文件
    if not validate_input_file(input_rds):
        print("❌ 输入文件验证失败，测试终止")
        sys.exit(1)
    
    # 测试计数器
    test_results = []
    
    # 测试1: 默认参数 - 标准SCTransform标准化
    print_analysis_info(
        "测试1: 默认参数标准化", 
        input_rds, 
        None, 
        3000
    )
    
    try:
        result1 = run_normalization_sct(
            input_rds=input_rds,
            vars_to_regress=None,
            n_variable_features=3000
        )
        analyze_results(result1, "默认参数")
        test_results.append(("默认参数", result1["status"] == "success"))
    except Exception as e:
        print(f"❌ 测试1失败: {str(e)}")
        test_results.append(("默认参数", False))
    
    # 测试2: 回归UMI计数 - 去除技术噪声
    print_analysis_info(
        "测试2: 回归UMI计数", 
        input_rds, 
        ["nCount_RNA"], 
        3000
    )
    
    try:
        result2 = run_normalization_sct(
            input_rds=input_rds,
            vars_to_regress=["nCount_RNA"],
            n_variable_features=3000
        )
        analyze_results(result2, "回归UMI计数")
        test_results.append(("回归UMI计数", result2["status"] == "success"))
    except Exception as e:
        print(f"❌ 测试2失败: {str(e)}")
        test_results.append(("回归UMI计数", False))
    
    # 测试3: 回归基因数量 - 去除检测基因数的影响
    print_analysis_info(
        "测试3: 回归基因数量", 
        input_rds, 
        ["nFeature_RNA"], 
        2000
    )
    
    try:
        result3 = run_normalization_sct(
            input_rds=input_rds,
            vars_to_regress=["nFeature_RNA"],
            n_variable_features=2000
        )
        analyze_results(result3, "回归基因数量")
        test_results.append(("回归基因数量", result3["status"] == "success"))
    except Exception as e:
        print(f"❌ 测试3失败: {str(e)}")
        test_results.append(("回归基因数量", False))
    
    # 测试4: 高变基因数量调整 - 更多特征用于下游分析
    print_analysis_info(
        "测试4: 高变基因数量调整", 
        input_rds, 
        None, 
        5000
    )
    
    try:
        result4 = run_normalization_sct(
            input_rds=input_rds,
            vars_to_regress=None,
            n_variable_features=5000
        )
        analyze_results(result4, "高变基因数量调整")
        test_results.append(("高变基因数量调整", result4["status"] == "success"))
    except Exception as e:
        print(f"❌ 测试4失败: {str(e)}")
        test_results.append(("高变基因数量调整", False))
    
    # 测试总结
    print("============================================================")
    print(" 测试总结")
    print("============================================================")
    
    success_count = sum(1 for _, success in test_results if success)
    total_count = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n📊 测试结果统计:")
    print(f"   成功数: {success_count}")
    print(f"   失败数: {total_count - success_count}")
    print(f"   成功率: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print(f"\n🎉 所有测试通过！SCTransform标准化功能正常工作。")
    else:
        print(f"\n⚠️  部分测试失败，请检查错误信息。")
    
    print(f"\n💡 生信建议:")
    print(f"   1. SCTransform相比LogNormalize能更好地处理技术噪声")
    print(f"   2. 根据数据特点选择合适的回归变量")
    print(f"   3. 高变基因数量影响下游分析的分辨率")
    print(f"   4. 检查生成的诊断图表评估标准化效果")
    print(f"   5. PCA结果用于后续的聚类和降维分析")