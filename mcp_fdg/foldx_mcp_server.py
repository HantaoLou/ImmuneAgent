"""
FoldX MCP Server

This server provides FoldX functionality for protein stability and interaction energy calculations.
FoldX is a force field-based tool for calculating the effect of mutations on protein stability.
"""

import os
import sys
import shutil
import subprocess
import pandas as pd
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Context
from typing import List, Dict, Optional

# 创建MCP服务器
mcp = FastMCP("FoldX Server")

# 全局变量
current_dir = Path(__file__).parent
FOLDX_BIN_DIR = current_dir / "foldx_bin"
TEMP_DIR = current_dir / "temp"

def check_foldx_installation():
    """检查FoldX安装"""
    foldx_executable = FOLDX_BIN_DIR / "foldx"
    rotabase_file = FOLDX_BIN_DIR / "rotabase.txt"
    
    if not foldx_executable.exists():
        raise FileNotFoundError(f"FoldX可执行文件不存在: {foldx_executable}")
    if not rotabase_file.exists():
        raise FileNotFoundError(f"FoldX rotabase文件不存在: {rotabase_file}")
    
    return True

def setup_foldx_workspace(workspace_name: str) -> Path:
    """设置FoldX工作空间"""
    workspace_path = TEMP_DIR / workspace_name
    
    # 清理并创建工作目录
    if workspace_path.exists():
        shutil.rmtree(workspace_path)
    workspace_path.mkdir(parents=True)
    
    # 复制FoldX二进制文件和rotabase
    shutil.copy(FOLDX_BIN_DIR / "foldx", workspace_path / "foldx")
    shutil.copy(FOLDX_BIN_DIR / "rotabase.txt", workspace_path / "rotabase.txt")
    
    # 设置可执行权限（在Windows上可能不需要）
    foldx_exe = workspace_path / "foldx"
    if os.name != 'nt':  # 非Windows系统
        os.chmod(foldx_exe, 0o755)
    
    return workspace_path

def run_foldx_command(workspace_path: Path, command: str, pdb_file: str = None, 
                     mutant_file: str = None, additional_args: str = "") -> subprocess.CompletedProcess:
    """运行FoldX命令"""
    cmd_parts = [str(workspace_path / "foldx"), f"--command={command}"]
    
    if pdb_file:
        cmd_parts.append(f"--pdb={pdb_file}")
    if mutant_file:
        cmd_parts.append(f"--mutant-file={mutant_file}")
    if additional_args:
        cmd_parts.extend(additional_args.split())
    
    # 在工作目录中执行命令
    result = subprocess.run(
        cmd_parts,
        cwd=workspace_path,
        capture_output=True,
        text=True,
        timeout=300  # 5分钟超时
    )
    
    return result

def parse_foldx_output(workspace_path: Path, output_file: str) -> pd.DataFrame:
    """解析FoldX输出文件"""
    output_path = workspace_path / output_file
    
    if not output_path.exists():
        raise FileNotFoundError(f"FoldX输出文件不存在: {output_path}")
    
    try:
        # FoldX输出通常是制表符分隔的文件，跳过前8行注释
        df = pd.read_csv(output_path, sep="\t", skiprows=8, engine='python')
        return df
    except Exception as e:
        raise ValueError(f"解析FoldX输出文件失败: {str(e)}")

@mcp.tool()
def repair_pdb(
    pdb_path: str,
    context: Context
) -> dict:
    """
    使用FoldX修复PDB结构
    
    Args:
        pdb_path: PDB文件路径
        context: MCP上下文
        
    Returns:
        dict: 修复结果信息
    """
    try:
        # 验证输入
        pdb_file = Path(pdb_path)
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB文件不存在: {pdb_path}")
        
        # 检查FoldX安装
        check_foldx_installation()
        
        # 设置工作空间
        workspace = setup_foldx_workspace("repair")
        
        # 复制PDB文件到工作空间
        pdb_name = pdb_file.name
        shutil.copy(pdb_file, workspace / pdb_name)
        
        # 运行RepairPDB命令
        print(f"正在修复PDB文件: {pdb_name}")
        result = run_foldx_command(workspace, "RepairPDB", pdb_name)
        
        if result.returncode != 0:
            raise RuntimeError(f"FoldX RepairPDB失败: {result.stderr}")
        
        # 检查修复后的文件
        repaired_name = pdb_name.replace('.pdb', '_Repair.pdb')
        repaired_path = workspace / repaired_name
        
        if not repaired_path.exists():
            raise FileNotFoundError(f"修复后的PDB文件未生成: {repaired_name}")
        
        return {
            "status": "success",
            "original_pdb": str(pdb_file),
            "repaired_pdb": str(repaired_path),
            "workspace": str(workspace),
            "foldx_output": result.stdout,
            "message": f"PDB文件修复成功: {repaired_name}"
        }
        
    except Exception as e:
        error_msg = f"PDB修复失败: {str(e)}"
        print(error_msg)
        return {
            "status": "error",
            "error_message": error_msg,
            "repaired_pdb": None
        }

@mcp.tool()
def calculate_mutation_energy(
    pdb_path: str,
    mutations: List[str],
    repair_first: bool = True,
    context: Context = None
) -> dict:
    """
    计算突变对蛋白质稳定性的影响
    
    Args:
        pdb_path: PDB文件路径
        mutations: 突变列表，格式如["A123T", "K456E"]
        repair_first: 是否先修复PDB结构
        context: MCP上下文
        
    Returns:
        dict: 突变能量计算结果
    """
    try:
        # 验证输入
        pdb_file = Path(pdb_path)
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB文件不存在: {pdb_path}")
        
        if not mutations:
            raise ValueError("突变列表不能为空")
        
        # 检查FoldX安装
        check_foldx_installation()
        
        # 设置工作空间
        workspace = setup_foldx_workspace("mutation")
        
        # 复制PDB文件
        pdb_name = "origin.pdb"
        shutil.copy(pdb_file, workspace / pdb_name)
        
        # 创建突变列表文件
        mutant_file = "individual_list.txt"
        with open(workspace / mutant_file, 'w', encoding='utf-8') as f:
            for mutation in mutations:
                f.write(f"{mutation};\n")
        
        # 如果需要，先修复PDB
        if repair_first:
            print("正在修复PDB结构...")
            repair_result = run_foldx_command(workspace, "RepairPDB", pdb_name)
            if repair_result.returncode != 0:
                print(f"警告: PDB修复失败，继续使用原始文件: {repair_result.stderr}")
            else:
                # 重命名修复后的文件
                repaired_name = pdb_name.replace('.pdb', '_Repair.pdb')
                if (workspace / repaired_name).exists():
                    os.rename(workspace / repaired_name, workspace / pdb_name)
        
        # 运行BuildModel命令计算突变能量
        print(f"正在计算 {len(mutations)} 个突变的能量变化...")
        build_result = run_foldx_command(
            workspace, 
            "BuildModel", 
            pdb_name, 
            mutant_file,
            "--numberOfRuns=1"
        )
        
        if build_result.returncode != 0:
            raise RuntimeError(f"FoldX BuildModel失败: {build_result.stderr}")
        
        # 解析输出结果
        output_file = f"Dif_{pdb_name.replace('.pdb', '.fxout')}"
        try:
            results_df = parse_foldx_output(workspace, output_file)
            
            # 处理结果数据
            if "total energy" in results_df.columns:
                results_df = results_df.rename(columns={"total energy": "build_model_energy"})
            
            # 添加突变信息
            results_df["mutant"] = mutations
            
            # 转换为字典格式
            results_list = []
            for _, row in results_df.iterrows():
                result_dict = row.to_dict()
                results_list.append(result_dict)
            
            return {
                "status": "success",
                "pdb_file": str(pdb_file),
                "mutations_count": len(mutations),
                "mutations": mutations,
                "results": results_list,
                "workspace": str(workspace),
                "foldx_output": build_result.stdout,
                "message": f"成功计算 {len(mutations)} 个突变的能量变化"
            }
            
        except Exception as e:
            raise ValueError(f"解析FoldX结果失败: {str(e)}")
        
    except Exception as e:
        error_msg = f"突变能量计算失败: {str(e)}"
        print(error_msg)
        return {
            "status": "error",
            "error_message": error_msg,
            "results": None
        }

@mcp.tool()
def analyze_interaction_energy(
    pdb_path: str,
    context: Context
) -> dict:
    """
    分析蛋白质相互作用能量
    
    Args:
        pdb_path: PDB文件路径
        context: MCP上下文
        
    Returns:
        dict: 相互作用能量分析结果
    """
    try:
        # 验证输入
        pdb_file = Path(pdb_path)
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB文件不存在: {pdb_path}")
        
        # 检查FoldX安装
        check_foldx_installation()
        
        # 设置工作空间
        workspace = setup_foldx_workspace("interaction")
        
        # 复制PDB文件
        pdb_name = pdb_file.name
        shutil.copy(pdb_file, workspace / pdb_name)
        
        # 运行AnalyseComplex命令
        print(f"正在分析蛋白质复合物相互作用: {pdb_name}")
        result = run_foldx_command(workspace, "AnalyseComplex", pdb_name)
        
        if result.returncode != 0:
            raise RuntimeError(f"FoldX AnalyseComplex失败: {result.stderr}")
        
        # 解析输出结果
        output_file = f"Interaction_{pdb_name.replace('.pdb', '_AC.fxout')}"
        try:
            results_df = parse_foldx_output(workspace, output_file)
            
            # 转换为字典格式
            results_list = results_df.to_dict('records')
            
            return {
                "status": "success",
                "pdb_file": str(pdb_file),
                "interaction_analysis": results_list,
                "workspace": str(workspace),
                "foldx_output": result.stdout,
                "message": f"成功分析蛋白质相互作用能量"
            }
            
        except Exception as e:
            # 如果解析失败，至少返回FoldX的原始输出
            return {
                "status": "partial_success",
                "pdb_file": str(pdb_file),
                "foldx_output": result.stdout,
                "parse_error": str(e),
                "message": "FoldX分析完成，但结果解析失败"
            }
        
    except Exception as e:
        error_msg = f"相互作用能量分析失败: {str(e)}"
        print(error_msg)
        return {
            "status": "error",
            "error_message": error_msg,
            "interaction_analysis": None
        }

@mcp.tool()
def get_foldx_info(context: Context) -> dict:
    """
    获取FoldX安装和配置信息
    
    Returns:
        dict: FoldX状态信息
    """
    try:
        check_foldx_installation()
        
        foldx_exe = FOLDX_BIN_DIR / "foldx"
        rotabase = FOLDX_BIN_DIR / "rotabase.txt"
        
        return {
            "foldx_installed": True,
            "foldx_executable": str(foldx_exe),
            "rotabase_file": str(rotabase),
            "temp_directory": str(TEMP_DIR),
            "executable_exists": foldx_exe.exists(),
            "rotabase_exists": rotabase.exists(),
            "status": "ready"
        }
    except Exception as e:
        return {
            "foldx_installed": False,
            "error": str(e),
            "status": "error"
        }

if __name__ == "__main__":
    # 启动时检查FoldX安装
    try:
        check_foldx_installation()
        print("FoldX MCP服务器启动成功")
    except Exception as e:
        print(f"服务器启动失败: {e}")
    
    mcp.run()