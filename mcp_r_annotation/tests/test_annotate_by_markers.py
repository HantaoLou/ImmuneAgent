#!/usr/bin/env python3
"""
Manual Cell Type Annotation Test Script

测试 annotate_by_markers 方法的功能
基于标记基因对B细胞亚群进行手动细胞类型注释
"""

import os
import sys
from pathlib import Path

# 添加父目录到Python路径，以便导入annotation_mcp_server模块
current_dir = Path(__file__).parent.parent  # 获取mcp_r_annotation目录
sys.path.insert(0, str(current_dir))

# 导入annotation_mcp_server模块
from annotation_mcp_server import annotate_by_markers


def get_bcell_marker_mapping():
    """
    定义B细胞亚群的标记基因映射
    
    基于已发表的B细胞生物学研究，定义不同B细胞亚群的典型标记基因
    这是一个科学严谨的B细胞分类体系
    
    Returns:
        dict: 聚类ID到细胞类型的映射字典
    """
    # B细胞亚群标记基因映射
    # 基于免疫学和单细胞转录组学的经典研究
    marker_mapping = {
        "0": "Naive B cells",           # 初始B细胞 (CD19+, CD20+, IgD+, CD27-)
        "1": "Memory B cells",          # 记忆B细胞 (CD19+, CD20+, CD27+, IgD-)
        "2": "Plasma cells",            # 浆细胞 (CD138+, CD19-, PRDM1+, XBP1+)
        "3": "Germinal center B cells", # 生发中心B细胞 (CD19+, CD38+, BCL6+)
        "4": "Transitional B cells",    # 过渡B细胞 (CD19+, CD24+, CD38+, IgD+)
        "5": "Marginal zone B cells",   # 边缘区B细胞 (CD19+, CD21+, CD23-)
        "6": "Regulatory B cells",      # 调节性B细胞 (CD19+, CD24+, CD38+, IL10+)
        "7": "Activated B cells",       # 活化B细胞 (CD19+, CD69+, CD86+)
        "8": "Plasmablasts",           # 浆母细胞 (CD19+, CD138+, CD27+)
        "9": "Pre-B cells",            # 前B细胞 (CD19+, CD34+, TdT+)
        "10": "Pro-B cells",           # 原B细胞 (CD19-, CD34+, TdT+, CD10+)
        "11": "Follicular B cells",    # 滤泡B细胞 (CD19+, CD21+, CD23+)
        "12": "Switched memory B cells", # 类别转换记忆B细胞 (CD19+, CD27+, IgG+/IgA+)
        "13": "Unswitched memory B cells", # 未转换记忆B细胞 (CD19+, CD27+, IgM+, IgD+)
        "14": "Double negative B cells", # 双阴性B细胞 (CD19+, CD27-, IgD-)
        "15": "Age-associated B cells"   # 年龄相关B细胞 (CD19+, CD21-, CD11c+)
    }
    
    return marker_mapping


def test_annotate_by_markers():
    """
    测试 annotate_by_markers 方法
    
    使用指定的输入文件和B细胞标记基因映射进行手动细胞类型注释
    """
    print("=" * 60)
    print("Manual Cell Type Annotation 测试脚本")
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
    
    # 获取B细胞标记基因映射
    marker_list = get_bcell_marker_mapping()
    
    # 测试参数
    test_params = {
        "cluster_column": "seurat_clusters",    # 聚类列名
        "new_column": "manual_celltype"         # 新注释列名
    }
    
    print("-" * 40)
    print("测试参数:")
    print(f"  输入文件: {input_rds}")
    print(f"  聚类列: {test_params['cluster_column']}")
    print(f"  新注释列: {test_params['new_column']}")
    print(f"  标记基因映射数量: {len(marker_list)} 个细胞类型")
    print("-" * 40)
    print()
    
    print("📋 B细胞亚群标记基因映射:")
    print("-" * 40)
    for cluster_id, cell_type in marker_list.items():
        print(f"  聚类 {cluster_id}: {cell_type}")
    print("-" * 40)
    print()
    
    try:
        print("🚀 开始执行 Manual Cell Type Annotation...")
        print("注意: 这可能需要几分钟时间，请耐心等待...")
        print()
        
        # 调用 annotate_by_markers 方法
        result = annotate_by_markers(
            input_rds=input_rds,
            marker_list=marker_list,
            cluster_column=test_params["cluster_column"],
            new_column=test_params["new_column"]
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
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                        print(f"  {i}. {file_path} ({file_size:.2f} MB)")
                    else:
                        print(f"  {i}. {file_path} (文件不存在)")
            
            # 打印注释映射信息
            annotation_mapping = result.get("annotation_mapping", {})
            if annotation_mapping:
                print(f"\n🏷️ 应用的注释映射:")
                print("-" * 30)
                for cluster, celltype in annotation_mapping.items():
                    print(f"  聚类 {cluster} → {celltype}")
                print("-" * 30)
            
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
    print("Manual Cell Type Annotation 测试脚本")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"脚本位置: {__file__}")
    print()
    
    print("🧬 生信背景说明:")
    print("本测试基于B细胞生物学的经典研究，使用科学严谨的细胞类型分类体系")
    print("包括初始B细胞、记忆B细胞、浆细胞、生发中心B细胞等16种B细胞亚群")
    print("每种细胞类型都有其特异性的表面标记和转录特征")
    print()
    
    # 执行测试
    success = test_annotate_by_markers()
    
    print("=" * 60)
    if success:
        print("🎉 测试完成 - 成功!")
        print("✅ B细胞亚群注释已完成，可用于下游分析")
        return 0
    else:
        print("💥 测试完成 - 失败!")
        print("❌ 请检查输入数据和R脚本配置")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)