"""
集成MCP服务器 - GearBind, FoldX, DDG Calculator

此服务器将三个工具集成在一个MCP服务中，提供完整的蛋白质突变分析流程。
支持从单个PDB文件开始，进行完整的稳定性和结合亲和力分析。
"""

import os
import sys
import asyncio
import torch
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from mcp.server.fastmcp import FastMCP, Context

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入各个工具的核心功能
from ddg_models.predictor import DDGPredictor
from ddg_utils.misc import *
from ddg_utils.data import *
from ddg_utils.protein import *

# 创建集成MCP服务器
mcp = FastMCP("Integrated Protein Analysis Server")

# 全局变量
DDG_MODEL = None
GEARBIND_SOLVERS = []
FOLDX_BIN_DIR = current_dir / "foldx_bin"
TEMP_DIR = current_dir / "temp"

class ProteinAnalysisWorkflow:
    """蛋白质分析工作流类"""
    
    def __init__(self, workspace_name: str):
        self.workspace_name = workspace_name
        self.workspace_path = TEMP_DIR / workspace_name
        self.setup_workspace()
    
    def setup_workspace(self):
        """设置工作空间"""
        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)
        self.workspace_path.mkdir(parents=True)
        
        # 创建子目录
        (self.workspace_path / "foldx").mkdir()
        (self.workspace_path / "gearbind").mkdir()
        (self.workspace_path / "gearbind" / "data").mkdir()
        (self.workspace_path / "ddg").mkdir()
    
    def cleanup(self):
        """清理工作空间"""
        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)

def initialize_all_models():
    """初始化所有模型"""
    global DDG_MODEL, GEARBIND_SOLVERS
    
    # 初始化DDG模型
    if DDG_MODEL is None:
        try:
            model_path = current_dir / "ddg_checkpoints" / "model.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"DDG模型文件不存在: {model_path}")
            
            ckpt = torch.load(str(model_path), map_location='cpu')
            config = ckpt['config']
            weight = ckpt['model']
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            DDG_MODEL = DDGPredictor(config.model).to(device)
            DDG_MODEL.load_state_dict(weight)
            DDG_MODEL.eval()
            print("✓ DDG模型初始化成功")
        except Exception as e:
            print(f"✗ DDG模型初始化失败: {e}")
            raise
    
    # 初始化GearBind模型
    if not GEARBIND_SOLVERS:
        try:
            # 导入必要的模块
            from torchdrug import core, models, data, utils
            from easydict import EasyDict as edict
            from gearbind import dataset, layer, model, task, util
            from FDG_function import test_data
            
            # 检查检查点文件
            checkpoint_dir = current_dir / "gearbind_checkpoints"
            checkpoint_files = [checkpoint_dir / f"cl_gearbind{i}.pth" for i in range(5)]
            
            missing_files = [f for f in checkpoint_files if not f.exists()]
            if missing_files:
                raise FileNotFoundError(f"GearBind检查点文件缺失: {missing_files}")
            
            # 配置GearBind
            cfg = {
                'task': {
                    'class': 'BindingAffinityChange',
                    'model': {
                        'class': 'BindModel',
                        'num_mlp_layer': 2,
                        'model': {
                            'class': 'GearBind',
                            'input_dim': 58,
                            'hidden_dims': [128, 128, 128, 128],
                            'batch_norm': True,
                            'short_cut': True,
                            'concat_hidden': True,
                            'num_relation': 7,
                            'edge_input_dim': 59,
                            'num_angle_bin': 8
                        }
                    },
                    'graph_construction_model': {
                        'class': 'GraphConstruction',
                        'node_layers': [{'class': 'InterfaceGraph', 'cutoff': 6.0}],
                        'edge_layers': [
                            {'class': 'SequentialEdge', 'max_distance': 2},
                            {'class': 'SpatialEdge', 'radius': 10.0, 'max_distance': 5},
                            {'class': 'KNNEdge', 'k': 10, 'max_distance': 5}
                        ],
                        'edge_feature': 'gearnet'
                    },
                    'normalization': False,
                    'task': ['ddG'],
                    'criterion': 'mse',
                    'metric': ['mae', 'rmse', 'spearmanr', 'pearsonr']
                },
                'optimizer': {'class': 'Adam', 'lr': 0.0001},
                'engine': {'gpus': [0], 'batch_size': 2},
                'checkpoints': [str(f) for f in checkpoint_files]
            }
            
            cfg["dataset"] = {
                'class': 'test_data',
                'path': str(TEMP_DIR),
                'node_feature': 'residue_symbol',
                'residue_feature': 'default',
                'split': {'test_set': "gearbind"}
            }
            
            cfg = edict(cfg)
            dataset = test_data(path=cfg.dataset.path, split_list=[cfg.dataset.split.test_set])
            
            # 加载所有模型
            for i, checkpoint_path in enumerate(cfg.checkpoints):
                config_copy = cfg.copy()
                config_copy.checkpoint = checkpoint_path
                solver = util.build_solver(config_copy, dataset)
                GEARBIND_SOLVERS.append(solver)
            
            print("✓ GearBind模型初始化成功")
        except Exception as e:
            print(f"✗ GearBind模型初始化失败: {e}")
            raise

def check_foldx_available():
    """检查FoldX是否可用"""
    foldx_exe = FOLDX_BIN_DIR / "foldx"
    rotabase = FOLDX_BIN_DIR / "rotabase.txt"
    
    if not foldx_exe.exists() or not rotabase.exists():
        raise FileNotFoundError("FoldX未正确安装，请运行install_requirements.py")
    
    return True

@mcp.tool()
def analyze_protein_mutation(
    wt_pdb_path: str,
    mut_pdb_path: str,
    mutations: Optional[List[str]] = None,
    include_foldx: bool = True,
    include_ddg: bool = True,
    include_gearbind: bool = True,
    context: Context = None
) -> dict:
    """
    完整的蛋白质突变分析流程
    
    Args:
        wt_pdb_path: 野生型PDB文件路径
        mut_pdb_path: 突变型PDB文件路径
        mutations: 突变列表（可选，用于FoldX分析）
        include_foldx: 是否包含FoldX分析
        include_ddg: 是否包含DDG分析
        include_gearbind: 是否包含GearBind分析
        context: MCP上下文
        
    Returns:
        dict: 综合分析结果
    """
    workflow = None
    try:
        # 验证输入文件
        wt_path = Path(wt_pdb_path)
        mut_path = Path(mut_pdb_path)
        
        if not wt_path.exists():
            raise FileNotFoundError(f"野生型PDB文件不存在: {wt_pdb_path}")
        if not mut_path.exists():
            raise FileNotFoundError(f"突变型PDB文件不存在: {mut_pdb_path}")
        
        # 初始化模型
        initialize_all_models()
        
        # 创建工作流
        workflow = ProteinAnalysisWorkflow("integrated_analysis")
        
        results = {
            "input_files": {
                "wt_pdb": str(wt_path),
                "mut_pdb": str(mut_path)
            },
            "analysis_results": {},
            "summary": {},
            "status": "success"
        }
        
        # DDG分析
        if include_ddg and DDG_MODEL is not None:
            try:
                print("正在进行DDG分析...")
                batch = load_wt_mut_pdb_pair(str(wt_path), str(mut_path))
                device = next(DDG_MODEL.parameters()).device
                batch = recursive_to(batch, device)
                
                with torch.no_grad():
                    pred = DDG_MODEL(batch['wt'], batch['mut'])
                    ddg_value = float(pred.cpu())
                
                results["analysis_results"]["ddg"] = {
                    "stability_change": ddg_value,
                    "interpretation": "正值表示稳定性降低，负值表示稳定性增加",
                    "status": "success"
                }
                print(f"✓ DDG分析完成: ΔΔG = {ddg_value:.4f}")
                
            except Exception as e:
                results["analysis_results"]["ddg"] = {
                    "status": "error",
                    "error": str(e)
                }
                print(f"✗ DDG分析失败: {e}")
        
        # GearBind分析
        if include_gearbind and GEARBIND_SOLVERS:
            try:
                print("正在进行GearBind分析...")
                
                # 准备GearBind数据
                from Bio.PDB import PDBParser
                from FDG_function import get_sequence, get_chain, test
                
                parser = PDBParser(QUIET=True)
                wt_sequences = get_sequence(parser, str(wt_path))
                mut_sequences = get_sequence(parser, str(mut_path))
                chains = get_chain(parser, str(wt_path))
                
                # 识别突变
                detected_mutations = []
                for chain_id in wt_sequences:
                    if chain_id in mut_sequences:
                        wt_seq = wt_sequences[chain_id]
                        mut_seq = mut_sequences[chain_id]
                        for j, (wt_aa, mut_aa) in enumerate(zip(wt_seq, mut_seq)):
                            if wt_aa != mut_aa:
                                detected_mutations.append(f"{wt_aa}{chain_id}{j+1}{mut_aa}")
                
                # 设置GearBind工作目录
                gearbind_dir = workflow.workspace_path / "gearbind"
                data_dir = gearbind_dir / "data"
                
                # 准备数据文件
                data_info = {
                    "pdb_id": ["test_data"],
                    "mutation": [",".join(detected_mutations)],
                    "chain_a": ["".join([i for i in chains if i in ["H", "L"]])],
                    "chain_b": ["".join([i for i in chains if i not in ["H", "L"]])],
                    "wt_protein": ["WT.pdb"],
                    "mt_protein": ["MUT.pdb"]
                }
                
                pd.DataFrame(data_info).to_csv(gearbind_dir / "data.csv", index=False)
                shutil.copy(str(wt_path), str(data_dir / "WT.pdb"))
                shutil.copy(str(mut_path), str(data_dir / "MUT.pdb"))
                
                # 执行GearBind预测
                from FDG_function import test, cfg, dataset
                predictions = test(cfg, dataset)
                
                mean_prediction = np.mean(predictions, axis=0)[0] if predictions.size > 0 else 0.0
                
                results["analysis_results"]["gearbind"] = {
                    "binding_affinity_change": float(mean_prediction),
                    "mutations_detected": detected_mutations,
                    "individual_predictions": predictions.flatten().tolist() if predictions.size > 0 else [],
                    "interpretation": "正值表示结合亲和力降低，负值表示结合亲和力增加",
                    "status": "success"
                }
                print(f"✓ GearBind分析完成: ΔΔG = {mean_prediction:.4f}")
                
            except Exception as e:
                results["analysis_results"]["gearbind"] = {
                    "status": "error",
                    "error": str(e)
                }
                print(f"✗ GearBind分析失败: {e}")
        
        # FoldX分析
        if include_foldx and mutations:
            try:
                print("正在进行FoldX分析...")
                check_foldx_available()
                
                # 设置FoldX工作目录
                foldx_dir = workflow.workspace_path / "foldx"
                
                # 复制FoldX二进制文件
                shutil.copy(FOLDX_BIN_DIR / "foldx", foldx_dir / "foldx")
                shutil.copy(FOLDX_BIN_DIR / "rotabase.txt", foldx_dir / "rotabase.txt")
                
                # 复制PDB文件
                pdb_name = "origin.pdb"
                shutil.copy(str(wt_path), foldx_dir / pdb_name)
                
                # 创建突变列表文件
                mutant_file = "individual_list.txt"
                with open(foldx_dir / mutant_file, 'w', encoding='utf-8') as f:
                    for mutation in mutations:
                        f.write(f"{mutation};\n")
                
                # 运行FoldX BuildModel
                import subprocess
                cmd = [
                    str(foldx_dir / "foldx"),
                    "--command=BuildModel",
                    f"--pdb={pdb_name}",
                    f"--mutant-file={mutant_file}",
                    "--numberOfRuns=1"
                ]
                
                result = subprocess.run(cmd, cwd=foldx_dir, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    # 解析FoldX结果
                    output_file = foldx_dir / f"Dif_{pdb_name.replace('.pdb', '.fxout')}"
                    if output_file.exists():
                        foldx_df = pd.read_csv(output_file, sep="\t", skiprows=8, engine='python')
                        foldx_results = foldx_df.to_dict('records')
                        
                        results["analysis_results"]["foldx"] = {
                            "mutation_energies": foldx_results,
                            "mutations_analyzed": mutations,
                            "interpretation": "total energy值表示突变对蛋白质稳定性的影响",
                            "status": "success"
                        }
                        print("✓ FoldX分析完成")
                    else:
                        raise FileNotFoundError("FoldX输出文件未生成")
                else:
                    raise RuntimeError(f"FoldX执行失败: {result.stderr}")
                
            except Exception as e:
                results["analysis_results"]["foldx"] = {
                    "status": "error",
                    "error": str(e)
                }
                print(f"✗ FoldX分析失败: {e}")
        
        # 生成综合总结
        summary = {}
        if "ddg" in results["analysis_results"] and results["analysis_results"]["ddg"]["status"] == "success":
            summary["stability_prediction"] = results["analysis_results"]["ddg"]["stability_change"]
        
        if "gearbind" in results["analysis_results"] and results["analysis_results"]["gearbind"]["status"] == "success":
            summary["binding_prediction"] = results["analysis_results"]["gearbind"]["binding_affinity_change"]
        
        if "foldx" in results["analysis_results"] and results["analysis_results"]["foldx"]["status"] == "success":
            summary["foldx_analysis"] = "完成"
        
        results["summary"] = summary
        
        print("🎉 蛋白质突变分析完成")
        return results
        
    except Exception as e:
        error_msg = f"蛋白质突变分析失败: {str(e)}"
        print(error_msg)
        return {
            "status": "error",
            "error_message": error_msg,
            "analysis_results": {}
        }
    finally:
        if workflow:
            workflow.cleanup()

@mcp.tool()
def get_system_status(context: Context) -> dict:
    """
    获取系统状态信息
    
    Returns:
        dict: 系统状态和配置信息
    """
    try:
        status = {
            "models": {},
            "dependencies": {},
            "system": {}
        }
        
        # 检查DDG模型
        try:
            initialize_all_models()
            status["models"]["ddg"] = {
                "loaded": DDG_MODEL is not None,
                "device": str(next(DDG_MODEL.parameters()).device) if DDG_MODEL else "未加载"
            }
        except Exception as e:
            status["models"]["ddg"] = {"loaded": False, "error": str(e)}
        
        # 检查GearBind模型
        status["models"]["gearbind"] = {
            "loaded": len(GEARBIND_SOLVERS) > 0,
            "models_count": len(GEARBIND_SOLVERS)
        }
        
        # 检查FoldX
        try:
            check_foldx_available()
            status["models"]["foldx"] = {"available": True}
        except Exception as e:
            status["models"]["foldx"] = {"available": False, "error": str(e)}
        
        # 系统信息
        status["system"] = {
            "cuda_available": torch.cuda.is_available(),
            "python_version": sys.version,
            "workspace_directory": str(TEMP_DIR)
        }
        
        return status
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    # 启动时初始化
    try:
        print("正在初始化集成蛋白质分析服务器...")
        initialize_all_models()
        check_foldx_available()
        print("🚀 集成MCP服务器启动成功")
        print("可用功能:")
        print("  - DDG Calculator: 蛋白质稳定性预测")
        print("  - GearBind: 结合亲和力预测")
        print("  - FoldX: 突变能量计算")
    except Exception as e:
        print(f"⚠️  服务器启动警告: {e}")
        print("某些功能可能不可用，请检查安装配置")
    
    mcp.run()