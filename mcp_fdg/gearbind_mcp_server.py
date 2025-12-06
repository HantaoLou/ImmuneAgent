"""
GearBind MCP Server

This server provides GearBind functionality for protein-protein binding affinity prediction.
GearBind predicts the change in binding affinity upon mutation using graph neural networks.
"""

import os
import sys
import math
import pickle
import pandas as pd
import numpy as np
import torch
import shutil
from pathlib import Path
from tqdm import tqdm
from mcp.server.fastmcp import FastMCP, Context
from Bio.PDB import PDBParser
from Bio import PDB, SeqIO
from collections import defaultdict

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入torchdrug相关模块
from torchdrug.utils import comm
try:
    from torchdrug import core, models, data, utils
except:
    from torchdrug import core, models, data, utils

from easydict import EasyDict as edict
from gearbind import dataset, layer, model, task, util
from gearbind import residue_constants
from torchdrug.core import Registry as R
from torch.utils import data as torch_data
from torch.nn import functional as F

# 创建MCP服务器
mcp = FastMCP("GearBind Server")

# 全局变量
GEARBIND_CONFIG = None
GEARBIND_SOLVERS = []
GEARBIND_DATASET = None

# 氨基酸单字母代码映射
ONE_LETTER = {'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q',
              'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',
              'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',
              'GLY':'G', 'PRO':'P', 'CYS':'C'}

def get_sequence(parser, pdb_path):
    """从PDB文件提取序列"""
    structure = parser.get_structure("protein", pdb_path)
    sequences = {}
    
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            sequence = ""
            for residue in chain:
                if residue.get_id()[0] == ' ':  # 只处理标准残基
                    res_name = residue.get_resname()
                    if res_name in ONE_LETTER:
                        sequence += ONE_LETTER[res_name]
            if sequence:
                sequences[chain_id] = sequence
    
    return sequences

def get_chain(parser, pdb_path):
    """获取PDB文件中的链ID"""
    structure = parser.get_structure("protein", pdb_path)
    chains = []
    
    for model in structure:
        for chain in model:
            chains.append(chain.get_id())
    
    return chains

def initialize_gearbind():
    """初始化GearBind模型"""
    global GEARBIND_CONFIG, GEARBIND_SOLVERS, GEARBIND_DATASET
    
    if GEARBIND_SOLVERS:
        return GEARBIND_SOLVERS
    
    try:
        # 检查检查点文件
        checkpoint_dir = current_dir / "gearbind_checkpoints"
        checkpoint_files = [
            checkpoint_dir / f"cl_gearbind{i}.pth" for i in range(5)
        ]
        
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
        
        # 数据集配置
        final_dataset = {
            'class': 'test_data',
            'path': str(current_dir / 'temp'),
            'node_feature': 'residue_symbol',
            'residue_feature': 'default',
            'split': {'test_set': "gearbind"}
        }
        
        cfg["dataset"] = final_dataset
        GEARBIND_CONFIG = edict(cfg)
        
        # 创建临时目录
        temp_dir = current_dir / 'temp'
        temp_dir.mkdir(exist_ok=True)
        
        # 初始化数据集（这里需要实现test_data类）
        from FDG_function import test_data
        GEARBIND_DATASET = test_data(
            path=GEARBIND_CONFIG.dataset.path, 
            split_list=[GEARBIND_CONFIG.dataset.split.test_set]
        )
        
        # 加载所有模型
        GEARBIND_SOLVERS = []
        for i, checkpoint_path in enumerate(GEARBIND_CONFIG.checkpoints):
            config_copy = GEARBIND_CONFIG.copy()
            config_copy.checkpoint = checkpoint_path
            solver = util.build_solver(config_copy, GEARBIND_DATASET)
            GEARBIND_SOLVERS.append(solver)
            print(f"已加载GearBind模型 {i+1}/5")
        
        print("GearBind模型初始化成功")
        return GEARBIND_SOLVERS
        
    except Exception as e:
        print(f"GearBind模型初始化失败: {str(e)}")
        raise

def gearbind_test(cfg, dataset):
    """执行GearBind预测"""
    try:
        device = torch.device(cfg.engine.gpus[0] if torch.cuda.is_available() else 'cpu')
        
        predictions = []
        for i, solver in enumerate(GEARBIND_SOLVERS):
            solver.model.eval()
            dataloader = data.DataLoader(dataset, cfg.engine.batch_size, shuffle=False, num_workers=0)
            
            batch_predictions = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(device)
                    pred = solver.model(batch)
                    batch_predictions.extend(pred['ddG'].cpu().numpy())
            
            predictions.append(batch_predictions)
        
        # 返回所有模型的预测结果
        return np.array(predictions)
        
    except Exception as e:
        print(f"GearBind预测失败: {str(e)}")
        raise

@mcp.tool()
def predict_binding_affinity(
    wt_pdb_path: str,
    mut_pdb_path: str,
    context: Context
) -> dict:
    """
    预测蛋白质突变对结合亲和力的影响
    
    Args:
        wt_pdb_path: 野生型PDB文件路径
        mut_pdb_path: 突变型PDB文件路径
        context: MCP上下文
        
    Returns:
        dict: 包含结合亲和力预测结果的字典
    """
    try:
        # 验证输入文件
        wt_path = Path(wt_pdb_path)
        mut_path = Path(mut_pdb_path)
        
        if not wt_path.exists():
            raise FileNotFoundError(f"野生型PDB文件不存在: {wt_pdb_path}")
        if not mut_path.exists():
            raise FileNotFoundError(f"突变型PDB文件不存在: {mut_pdb_path}")
        
        # 初始化模型
        solvers = initialize_gearbind()
        
        # 创建临时工作目录
        temp_dir = current_dir / 'temp' / 'gearbind'
        temp_dir.mkdir(parents=True, exist_ok=True)
        data_dir = temp_dir / 'data'
        data_dir.mkdir(exist_ok=True)
        
        # 解析PDB文件获取序列和链信息
        parser = PDBParser(QUIET=True)
        wt_sequences = get_sequence(parser, str(wt_path))
        mut_sequences = get_sequence(parser, str(mut_path))
        chains = get_chain(parser, str(wt_path))
        
        # 识别突变
        mutations = []
        for chain_id in wt_sequences:
            if chain_id in mut_sequences:
                wt_seq = wt_sequences[chain_id]
                mut_seq = mut_sequences[chain_id]
                for j, (wt_aa, mut_aa) in enumerate(zip(wt_seq, mut_seq)):
                    if wt_aa != mut_aa:
                        mutations.append(f"{wt_aa}{chain_id}{j+1}{mut_aa}")
        
        mutations_str = ",".join(mutations)
        
        # 准备数据
        data_info = {
            "pdb_id": ["test_data"],
            "mutation": [mutations_str],
            "chain_a": ["".join([i for i in chains if i in ["H", "L"]])],
            "chain_b": ["".join([i for i in chains if i not in ["H", "L"]])],
            "wt_protein": ["WT.pdb"],
            "mt_protein": ["MUT.pdb"]
        }
        
        # 保存数据文件
        pd.DataFrame(data_info).to_csv(temp_dir / "data.csv", index=False)
        shutil.copy(str(wt_path), str(data_dir / "WT.pdb"))
        shutil.copy(str(mut_path), str(data_dir / "MUT.pdb"))
        
        # 执行预测
        predictions = gearbind_test(GEARBIND_CONFIG, GEARBIND_DATASET)
        
        # 计算统计结果
        mean_prediction = np.mean(predictions, axis=0)[0] if predictions.size > 0 else 0.0
        individual_predictions = predictions.flatten().tolist() if predictions.size > 0 else []
        
        result = {
            "binding_affinity_change": float(mean_prediction),
            "individual_predictions": individual_predictions,
            "mutations_detected": mutations,
            "mutations_string": mutations_str,
            "chains_analyzed": chains,
            "wt_pdb": str(wt_path),
            "mut_pdb": str(mut_path),
            "status": "success"
        }
        
        print(f"GearBind预测完成: ΔΔG = {mean_prediction:.4f}")
        return result
        
    except Exception as e:
        error_msg = f"GearBind预测失败: {str(e)}"
        print(error_msg)
        return {
            "status": "error",
            "error_message": error_msg,
            "binding_affinity_change": None
        }

@mcp.tool()
def get_gearbind_model_info(context: Context) -> dict:
    """
    获取GearBind模型信息
    
    Returns:
        dict: 模型配置和状态信息
    """
    try:
        solvers = initialize_gearbind()
        
        return {
            "models_loaded": len(solvers),
            "expected_models": 5,
            "cuda_available": torch.cuda.is_available(),
            "temp_directory": str(current_dir / 'temp'),
            "status": "ready"
        }
    except Exception as e:
        return {
            "models_loaded": 0,
            "error": str(e),
            "status": "error"
        }

if __name__ == "__main__":
    # 启动时初始化模型
    try:
        initialize_gearbind()
        print("GearBind MCP服务器启动成功")
    except Exception as e:
        print(f"服务器启动失败: {e}")
    
    mcp.run()