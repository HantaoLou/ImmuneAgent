#!/usr/bin/env python3
"""
tSNE重复数据点修复测试
专门测试修复后的tSNE功能
"""

import os
import sys
import json
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrna_mcp_server import load_config, run_r_script, run_dim_reduction

def test_tsne_duplicate_fix():
    """测试tSNE重复数据点修复功能"""
    
    print("tSNE重复数据点修复测试")
    print("=" * 60)
    
    # 输入文件
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    # 验证输入文件
    if not os.path.exists(input_rds):
        print(f"❌ 输入文件不存在: {input_rds}")
        return False
    
    file_size = os.path.getsize(input_rds) / (1024 * 1024)
    print(f"✓ 输入文件验证通过: {input_rds}")
    print(f"  文件大小: {file_size:.1f} MB")
    print()
    
    # 测试仅tSNE（基于PCA）
    print("测试: 仅tSNE降维分析（基于PCA）")
    print("-" * 40)
    
    try:
        # 首先运行PCA
        print("步骤1: 运行PCA...")
        pca_result = run_dim_reduction(
            input_rds=input_rds,
            methods=["PCA"],
            dims=30
        )
        
        if not pca_result.get("success", False):
            print("❌ PCA分析失败")
            return False
        
        print("✓ PCA分析成功")
        
        # 使用PCA结果运行tSNE
        print("步骤2: 运行tSNE（基于PCA结果）...")
        pca_output_rds = os.path.join(pca_result["output_dir"], "seurat_with_reductions.rds")
        
        tsne_result = run_dim_reduction(
            input_rds=pca_output_rds,
            methods=["tSNE"],
            dims=30
        )
        
        if tsne_result.get("success", False):
            print("✓ tSNE分析成功！重复数据点问题已解决")
            print(f"✓ 输出目录: {tsne_result['output_dir']}")
            print(f"✓ 生成文件数量: {tsne_result['file_count']}")
            
            # 显示生成的文件
            if "files" in tsne_result:
                print("\n生成的文件:")
                for file_type, files in tsne_result["files"].items():
                    if files:
                        print(f"  - {file_type}文件: {len(files)}")
                        for file in files:
                            print(f"    * {file}")
            
            # 显示R脚本输出的关键信息
            if "stdout" in tsne_result:
                stdout_lines = tsne_result["stdout"].split('\n')
                print("\nR脚本关键输出:")
                for line in stdout_lines:
                    if any(keyword in line.lower() for keyword in 
                          ['duplicate', 'tsne', 'completed', 'error']):
                        print(f"  {line}")
            
            return True
        else:
            print("❌ tSNE分析仍然失败")
            if "message" in tsne_result:
                print(f"错误信息: {tsne_result['message']}")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出现异常: {str(e)}")
        return False

def print_solution_summary():
    """打印解决方案总结"""
    print("\n" + "=" * 60)
    print("tSNE重复数据点问题解决方案总结")
    print("=" * 60)
    print("1. 问题诊断:")
    print("   - tSNE算法对重复数据点敏感")
    print("   - 单细胞数据中常见重复的PCA坐标")
    print("   - 原脚本缺少重复数据点检测")
    print()
    print("2. 科学解决方案:")
    print("   - 在运行tSNE前检测重复的PCA坐标")
    print("   - 仅对唯一数据点运行tSNE")
    print("   - 将重复点分配到相同的tSNE坐标")
    print("   - 保持原始数据集的完整性")
    print()
    print("3. 生物学意义:")
    print("   - 保留了所有细胞的信息")
    print("   - 重复点反映了真实的生物学相似性")
    print("   - 不会丢失任何细胞数据")
    print()
    print("4. 技术优势:")
    print("   - 自动检测和处理重复点")
    print("   - 向后兼容（无重复时正常运行）")
    print("   - 保持Seurat对象结构完整")

if __name__ == "__main__":
    success = test_tsne_duplicate_fix()
    print_solution_summary()
    
    if success:
        print("\n🎉 tSNE重复数据点问题已成功解决！")
    else:
        print("\n❌ 测试失败，需要进一步调试")