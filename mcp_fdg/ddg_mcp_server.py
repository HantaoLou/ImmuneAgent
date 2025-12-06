"""
DDG Calculator MCP Server

This server provides DDG (ΔΔG) calculation functionality for protein stability prediction.
DDG Calculator predicts the change in binding free energy upon mutation.
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Context

# 添加当前目录到Python路径，以便导入本地模块
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from ddg_models.predictor import DDGPredictor
from ddg_utils.misc import *
from ddg_utils.data import *
from ddg_utils.protein import *

# 创建MCP服务器
mcp = FastMCP("DDG Calculator Server")

# 全局变量：DDG模型
DDG_MODEL = None
DDG_CONFIG = None

def initialize_ddg_model():
    """初始化DDG模型"""
    global DDG_MODEL, DDG_CONFIG
    
    if DDG_MODEL is not None:
        return DDG_MODEL
    
    try:
        # 检查模型文件是否存在
        model_path = current_dir / "ddg_checkpoints" / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"DDG模型文件不存在: {model_path}")
        
        # 加载模型
        print(f"正在加载DDG模型: {model_path}")
        ckpt = torch.load(str(model_path), map_location='cpu')
        DDG_CONFIG = ckpt['config']
        weight = ckpt['model']
        
        # 检查CUDA可用性
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        
        DDG_MODEL = DDGPredictor(DDG_CONFIG.model).to(device)
        DDG_MODEL.load_state_dict(weight)
        DDG_MODEL.eval()
        
        print("DDG模型初始化成功")
        return DDG_MODEL
        
    except Exception as e:
        print(f"DDG模型初始化失败: {str(e)}")
        raise

@mcp.tool()
def calculate_ddg(
    wt_pdb_path: str,
    mut_pdb_path: str,
    context: Context
) -> dict:
    """
    计算蛋白质突变的ΔΔG值
    
    Args:
        wt_pdb_path: 野生型PDB文件路径
        mut_pdb_path: 突变型PDB文件路径
        context: MCP上下文
        
    Returns:
        dict: 包含ΔΔG预测值的字典
        
    Raises:
        FileNotFoundError: 当PDB文件不存在时
        RuntimeError: 当模型预测失败时
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
        model = initialize_ddg_model()
        
        # 加载PDB数据
        print(f"正在处理PDB文件: {wt_path.name} -> {mut_path.name}")
        batch = load_wt_mut_pdb_pair(str(wt_path), str(mut_path))
        
        # 移动到正确的设备
        device = next(model.parameters()).device
        batch = recursive_to(batch, device)
        
        # 进行预测
        with torch.no_grad():
            pred = model(batch['wt'], batch['mut'])
            ddg_value = float(pred.cpu())
        
        result = {
            "ddg_prediction": ddg_value,
            "wt_pdb": str(wt_path),
            "mut_pdb": str(mut_path),
            "device_used": str(device),
            "status": "success"
        }
        
        print(f"DDG预测完成: ΔΔG = {ddg_value:.4f}")
        return result
        
    except Exception as e:
        error_msg = f"DDG计算失败: {str(e)}"
        print(error_msg)
        return {
            "status": "error",
            "error_message": error_msg,
            "ddg_prediction": None
        }

@mcp.tool()
def get_ddg_model_info(context: Context) -> dict:
    """
    获取DDG模型信息
    
    Returns:
        dict: 模型配置和状态信息
    """
    try:
        model = initialize_ddg_model()
        device = next(model.parameters()).device
        
        return {
            "model_loaded": True,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "model_config": str(DDG_CONFIG) if DDG_CONFIG else "未加载",
            "status": "ready"
        }
    except Exception as e:
        return {
            "model_loaded": False,
            "error": str(e),
            "status": "error"
        }

if __name__ == "__main__":
    # 启动时初始化模型
    try:
        initialize_ddg_model()
        print("DDG Calculator MCP服务器启动成功")
    except Exception as e:
        print(f"服务器启动失败: {e}")
    
    mcp.run()