#!/usr/bin/env python3
"""
测试降维分析功能 (Dimensionality Reduction)

测试 run_dim_reduction 方法，验证PCA、UMAP、tSNE等降维算法的执行。
从生信角度确保降维分析的科学严谨性和结果可靠性。
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

# 添加当前目录到Python路径，以便导入scrna_mcp_server模块
sys.path.insert(0, str(Path(__file__).parent))

def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {
        "base_dir": str(Path(__file__).parent),
        "output_dir": "output",
        "default_timeout": 3600
    }

def run_r_script(
    script_name: str,
    input_rds: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 3600
) -> Dict[str, Any]:
    """
    直接运行R脚本，避免MCP依赖
    
    Args:
        script_name: R脚本名称（不含.R扩展名）
        input_rds: 输入Seurat RDS文件路径
        params: 传递给R脚本的参数（JSON格式）
        timeout: 执行超时时间（秒）
    
    Returns:
        包含状态、生成文件和消息的字典
    """
    # 验证输入文件
    if not os.path.exists(input_rds):
        return {
            "status": "error",
            "message": f"输入文件不存在: {input_rds}",
            "generated_files": []
        }

    # 加载配置
    config = load_config()
    working_dir = Path(__file__).parent
    base_dir = Path(config["base_dir"])

    # R脚本路径
    r_script_path = working_dir / "scripts" / f"{script_name}.R"

    # 检查R脚本是否存在
    if not r_script_path.exists():
        return {
            "status": "error",
            "message": f"R脚本不存在: {r_script_path}",
            "generated_files": []
        }

    # 准备参数
    params_json = json.dumps(params) if params else "{}"

    try:
        # 执行R脚本
        result = subprocess.run(
            ["Rscript", str(r_script_path), input_rds, params_json],
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )

        # 检查执行结果
        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"R脚本执行失败 (返回码: {result.returncode})",
                "stderr": result.stderr,
                "stdout": result.stdout,
                "generated_files": []
            }

        # 收集生成的文件
        output_dir = base_dir / config["output_dir"] / script_name
        generated_files = []

        if output_dir.exists():
            # RDS文件（处理后的Seurat对象）
            rds_files = list(output_dir.glob("*.rds"))
            generated_files.extend([str(f) for f in rds_files])

            # CSV文件（表格、统计数据）
            csv_files = list(output_dir.glob("*.csv"))
            generated_files.extend([str(f) for f in csv_files])

            # PDF/PNG文件（图表）
            plot_files = list(output_dir.glob("*.pdf")) + list(output_dir.glob("*.png"))
            generated_files.extend([str(f) for f in plot_files])

            # TXT文件（日志、摘要）
            txt_files = list(output_dir.glob("*.txt"))
            generated_files.extend([str(f) for f in txt_files])

        return {
            "status": "success",
            "message": f"{script_name} 分析成功完成",
            "output_directory": str(output_dir),
            "generated_files": generated_files,
            "file_count": len(generated_files),
            "stdout": result.stdout
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": f"R脚本执行超时 (超过 {timeout} 秒)",
            "generated_files": []
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"执行R脚本时发生错误: {str(e)}",
            "generated_files": []
        }

def run_dim_reduction(
    input_rds: str,
    methods: Optional[List[str]] = None,
    dims: int = 30,
    n_neighbors: int = 30,
    min_dist: float = 0.3
) -> Dict[str, Any]:
    """
    降维分析和可视化
    
    生成多种嵌入方法：
    - PCA (主成分分析)
    - UMAP (均匀流形逼近和投影)
    - tSNE (t-分布随机邻域嵌入)
    
    Args:
        input_rds: 输入Seurat RDS文件路径（标准化后）
        methods: 要运行的方法列表（默认: ["PCA", "UMAP", "tSNE"]）
        dims: 计算的维度数量（默认: 30）
        n_neighbors: UMAP参数，局部邻域大小（默认: 30）
        min_dist: UMAP参数，最小距离（默认: 0.3）
    
    Returns:
        包含状态、Seurat对象（含嵌入）和可视化图表的字典
    """
    params = {
        "methods": methods or ["PCA", "UMAP", "tSNE"],
        "dims": dims,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist
    }
    return run_r_script("dim_reduction", input_rds, params)

def validate_input_file(input_rds: str) -> bool:
    """
    验证输入文件的有效性
    
    Args:
        input_rds: 输入RDS文件路径
    
    Returns:
        文件是否有效
    """
    if not os.path.exists(input_rds):
        print(f"❌ 输入文件不存在: {input_rds}")
        return False
    
    file_size = os.path.getsize(input_rds) / (1024 * 1024)  # MB
    print(f"✓ 输入文件验证通过: {input_rds}")
    print(f"  文件大小: {file_size:.1f} MB")
    
    return True

def analyze_results(result: Dict[str, Any]) -> None:
    """
    分析降维结果
    
    Args:
        result: run_dim_reduction的返回结果
    """
    print("\n" + "="*60)
    print("降维分析结果分析")
    print("="*60)
    
    if result["status"] == "error":
        print(f"❌ 降维分析失败: {result['message']}")
        if "stderr" in result:
            print(f"错误信息: {result['stderr']}")
        return
    
    print(f"✓ 降维分析执行成功")
    print(f"✓ 输出目录: {result.get('output_directory', 'N/A')}")
    print(f"✓ 生成文件数量: {result.get('file_count', 0)}")
    
    # 分析生成的文件
    generated_files = result.get("generated_files", [])
    if generated_files:
        print("\n生成的文件:")
        
        # 分类文件
        rds_files = [f for f in generated_files if f.endswith('.rds')]
        csv_files = [f for f in generated_files if f.endswith('.csv')]
        plot_files = [f for f in generated_files if f.endswith(('.pdf', '.png'))]
        
        print(f"  - RDS文件: {len(rds_files)}")
        for f in rds_files:
            print(f"    * {os.path.basename(f)}")
        
        print(f"  - CSV文件: {len(csv_files)}")
        for f in csv_files:
            print(f"    * {os.path.basename(f)}")
        
        print(f"  - 图形文件: {len(plot_files)}")
        for f in plot_files:
            print(f"    * {os.path.basename(f)}")
    
    # 显示R脚本输出
    if result.get("stdout"):
        print(f"\nR脚本输出:")
        print(result["stdout"])

def test_dim_reduction_basic():
    """测试基本降维分析功能"""
    print("="*60)
    print("测试1: 基本降维分析 (PCA, UMAP, tSNE)")
    print("="*60)
    
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    # 验证输入文件
    if not validate_input_file(input_rds):
        return
    
    # 运行降维分析
    print("\n开始降维分析...")
    result = run_dim_reduction(
        input_rds=input_rds,
        methods=["PCA", "UMAP", "tSNE"],
        dims=30,
        n_neighbors=30,
        min_dist=0.3
    )
    
    # 分析结果
    analyze_results(result)

def test_dim_reduction_pca_only():
    """测试仅PCA降维"""
    print("\n" + "="*60)
    print("测试2: 仅PCA降维分析")
    print("="*60)
    
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    # 验证输入文件
    if not validate_input_file(input_rds):
        return
    
    # 运行PCA分析
    print("\n开始PCA分析...")
    result = run_dim_reduction(
        input_rds=input_rds,
        methods=["PCA"],
        dims=50  # 增加PCA维度
    )
    
    # 分析结果
    analyze_results(result)

def test_dim_reduction_custom_params():
    """测试自定义参数的降维分析"""
    print("\n" + "="*60)
    print("测试3: 自定义参数降维分析")
    print("="*60)
    
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    # 验证输入文件
    if not validate_input_file(input_rds):
        return
    
    # 运行自定义参数分析
    print("\n开始自定义参数降维分析...")
    print("参数设置:")
    print("  - 方法: UMAP, tSNE")
    print("  - 维度: 20")
    print("  - UMAP邻居数: 15")
    print("  - UMAP最小距离: 0.1")
    
    result = run_dim_reduction(
        input_rds=input_rds,
        methods=["UMAP", "tSNE"],
        dims=20,
        n_neighbors=15,
        min_dist=0.1
    )
    
    # 分析结果
    analyze_results(result)

def print_bioinformatics_guidance():
    """打印生信分析指导"""
    print("\n" + "="*60)
    print("生信分析指导建议")
    print("="*60)
    print("1. 降维方法选择:")
    print("   - PCA: 线性降维，保留主要变异，适合初步探索")
    print("   - UMAP: 非线性降维，保留局部和全局结构，推荐用于可视化")
    print("   - tSNE: 非线性降维，强调局部结构，适合聚类可视化")
    
    print("\n2. 参数优化建议:")
    print("   - dims: 通常选择10-50，根据数据复杂度调整")
    print("   - n_neighbors: UMAP参数，10-100，影响局部vs全局平衡")
    print("   - min_dist: UMAP参数，0.01-0.5，控制点的紧密程度")
    
    print("\n3. 质量评估:")
    print("   - 检查降维图中的聚类分离度")
    print("   - 验证生物学相关的细胞类型分群")
    print("   - 评估批次效应的去除效果")
    
    print("\n4. 下游分析:")
    print("   - 基于降维结果进行聚类分析")
    print("   - 识别细胞类型特异性标记基因")
    print("   - 进行轨迹分析和伪时间分析")

if __name__ == "__main__":
    print("降维分析(Dimensionality Reduction)测试")
    print("="*60)
    
    try:
        # 测试1: 基本降维分析
        test_dim_reduction_basic()
        
        # 测试2: 仅PCA分析
        test_dim_reduction_pca_only()
        
        # 测试3: 自定义参数分析
        test_dim_reduction_custom_params()
        
        # 打印生信指导
        print_bioinformatics_guidance()
        
        print("\n" + "="*60)
        print("所有降维分析测试完成!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()