#!/usr/bin/env python3
"""
Marker基因检测测试 - 独立版本

本测试文件专门用于测试marker基因检测功能，避免MCP依赖问题。
直接调用R脚本进行marker基因检测，验证生物学意义和技术实现。

生物信息学背景:
- Marker基因是特定细胞类型或状态的特征性表达基因
- 通过FindAllMarkers识别每个聚类的特异性表达基因
- 评估基因的表达特异性(pct.1 vs pct.2)和差异倍数(log2FC)
- 生成可视化图表帮助解释生物学意义

测试策略:
1. 使用真实的单细胞数据(Age_Bcells.rds)
2. 测试不同参数组合的科学合理性
3. 验证输出文件的完整性和生物学意义
4. 分析marker基因的表达模式和功能注释
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

def validate_input_file(file_path: str) -> bool:
    """验证输入RDS文件"""
    if not os.path.exists(file_path):
        print(f"❌ 输入文件不存在: {file_path}")
        return False
    
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    print(f"✓ 输入文件验证通过: {file_path}")
    print(f"  文件大小: {file_size:.2f} MB")
    return True

def clean_output_directory(output_dir: str):
    """清理输出目录"""
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            print(f"✓ 清理输出目录: {output_dir}")
        except PermissionError as e:
            print(f"⚠️ 清理目录时遇到权限问题: {e}")
            print("  继续执行测试...")
    os.makedirs(output_dir, exist_ok=True)

def run_r_script(script_name: str, input_rds: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """直接运行R脚本，避免MCP依赖"""
    config = {
        "base_dir": str(Path(__file__).parent),
        "output_dir": "output",
        "default_timeout": 3600
    }
    
    script_path = Path(config["base_dir"]) / "scripts" / f"{script_name}.R"
    output_dir = Path(config["base_dir"]) / config["output_dir"] / script_name
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备R脚本参数
    r_params = {
        "input_rds": input_rds,
        "output_dir": str(output_dir),
        **params
    }
    
    # 创建临时参数文件
    params_file = output_dir / "params.json"
    with open(params_file, 'w') as f:
        json.dump(r_params, f, indent=2)
    
    # 运行R脚本 - 传递两个参数：input_rds 和 params_file
    cmd = ["Rscript", str(script_path), input_rds, str(params_file)]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config["default_timeout"],
            cwd=config["base_dir"],
            encoding='utf-8',  # 明确指定UTF-8编码
            errors='replace'   # 遇到编码错误时替换为占位符
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "output_dir": str(output_dir)
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Script timeout after {config['default_timeout']} seconds",
            "return_code": -1
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "return_code": -1
        }

def analyze_marker_results(output_dir: str) -> Dict[str, Any]:
    """分析marker检测结果"""
    results = {
        "files_generated": [],
        "marker_stats": {},
        "biological_insights": []
    }
    
    output_path = Path(output_dir)
    
    # 检查生成的文件
    expected_files = [
        "all_markers.csv",
        "topN_markers.csv", 
        "marker_summary.csv",
        "dotplot_top_markers.pdf",
        "heatmap_top_markers.pdf",
        "violin_top_markers.pdf",
        "overall_statistics.csv"
    ]
    
    for file_name in expected_files:
        file_path = output_path / file_name
        if file_path.exists():
            results["files_generated"].append(file_name)
            
    # 分析marker统计信息
    all_markers_file = output_path / "all_markers.csv"
    if all_markers_file.exists():
        try:
            import pandas as pd
            markers_df = pd.read_csv(all_markers_file)
            
            results["marker_stats"] = {
                "total_markers": len(markers_df),
                "significant_markers": len(markers_df[markers_df['p_val_adj'] < 0.05]),
                "clusters_analyzed": markers_df['cluster'].nunique(),
                "avg_logfc_range": [markers_df['avg_log2FC'].min(), markers_df['avg_log2FC'].max()],
                "top_genes": markers_df.nlargest(5, 'avg_log2FC')['gene'].tolist()
            }
            
            # 生物学洞察
            high_fc_genes = markers_df[markers_df['avg_log2FC'] > 1.0]
            if len(high_fc_genes) > 0:
                results["biological_insights"].append(
                    f"发现 {len(high_fc_genes)} 个高表达差异基因 (log2FC > 1.0)"
                )
                
            specific_genes = markers_df[(markers_df['pct.1'] > 0.5) & (markers_df['pct.2'] < 0.2)]
            if len(specific_genes) > 0:
                results["biological_insights"].append(
                    f"发现 {len(specific_genes)} 个高特异性marker基因"
                )
                
        except Exception as e:
            results["marker_stats"]["error"] = f"分析CSV文件时出错: {e}"
    
    return results

def run_marker_detection_tests():
    """运行marker检测测试套件"""
    print("Marker基因检测测试")
    print("=" * 60)
    
    # 输入文件路径
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    # 验证输入文件
    if not validate_input_file(input_rds):
        return
    
    # 清理输出目录
    clean_output_directory("output/marker_detection")
    
    # 测试用例
    test_cases = [
        {
            "name": "默认参数聚类marker检测",
            "description": "使用标准参数检测每个聚类的marker基因",
            "params": {
                "group_by": "seurat_clusters",
                "only_pos": True,
                "min_pct": 0.25,
                "logfc_threshold": 0.5,
                "top_n": 10
            }
        },
        {
            "name": "高阈值严格筛选",
            "description": "使用更严格的阈值筛选高质量marker基因",
            "params": {
                "group_by": "seurat_clusters", 
                "only_pos": True,
                "min_pct": 0.4,
                "logfc_threshold": 1.0,
                "top_n": 5
            }
        },
        {
            "name": "细胞类型分组marker检测",
            "description": "基于细胞类型注释检测marker基因",
            "params": {
                "group_by": "CellType",
                "only_pos": True,
                "min_pct": 0.25,
                "logfc_threshold": 0.5,
                "top_n": 15
            }
        },
        {
            "name": "包含负向marker检测",
            "description": "同时检测正向和负向marker基因",
            "params": {
                "group_by": "seurat_clusters",
                "only_pos": False,
                "min_pct": 0.2,
                "logfc_threshold": 0.3,
                "top_n": 8
            }
        }
    ]
    
    # 运行测试
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 20} 测试{i}: {test_case['name']} {'=' * 20}")
        print("参数设置:")
        for key, value in test_case['params'].items():
            print(f"  {key}: {value}")
        print()
        
        # 运行marker检测
        result = run_r_script("marker_detection", input_rds, test_case['params'])
        
        print(f"=== 测试{i}: {test_case['name']} 结果分析 ===")
        if result["success"]:
            print("✓ Marker检测成功!")
            
            # 分析结果
            analysis = analyze_marker_results(result["output_dir"])
            
            print(f"\n📊 统计信息:")
            print(f"  生成文件数: {len(analysis['files_generated'])}")
            if analysis["marker_stats"]:
                stats = analysis["marker_stats"]
                if "total_markers" in stats:
                    print(f"  总marker数: {stats['total_markers']}")
                    print(f"  显著marker数: {stats['significant_markers']}")
                    print(f"  分析聚类数: {stats['clusters_analyzed']}")
                    print(f"  Log2FC范围: {stats['avg_logfc_range'][0]:.2f} ~ {stats['avg_logfc_range'][1]:.2f}")
                    print(f"  顶级基因: {', '.join(stats['top_genes'][:3])}")
            
            print(f"\n🔬 生物学洞察:")
            for insight in analysis["biological_insights"]:
                print(f"  • {insight}")
                
            print(f"\n📁 生成文件:")
            for file_name in analysis["files_generated"]:
                print(f"  • {file_name}")
                
        else:
            print("❌ Marker检测失败!")
            print(f"错误消息: {result.get('error', 'R script execution failed')}")
            if result.get('return_code'):
                print(f"返回码: {result['return_code']}")
            
            if result.get("stderr"):
                print(f"\n🔍 错误详情:")
                print("-" * 30)
                print(result["stderr"])
                print("-" * 30)
            
            if result.get("stdout"):
                print(f"\n📋 输出信息:")
                print("-" * 30)
                print(result["stdout"])
                print("-" * 30)
        
        print()

if __name__ == "__main__":
    run_marker_detection_tests()