"""
FDG MCP Server - Core FDG Tool Wrapper

This server exposes the core FDG (Foldx, DDG, GearBind) process via MCP protocol.
"""

from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("MetaBcr Core Server")

@mcp.tool()
def metabcr(input_file_path: str = None, output_file_path: str = None) -> str:
    """MetaBCR: A Deep Learning Framework for Antibody-Antigen Interaction Prediction
    MetaBCR is designed to predict the binding affinity between antibodies and antigens using deep learning models.
    It supports multiple model architectures, including CNN, GNN, and BERT-based models, and can be configured
    for various tasks and datasets through command-line arguments and configuration files.
    
    Args:
        input_file_path: Optional input file path. If provided, this path will be used; otherwise, the default path will be used
        output_file_path: Optional output directory path. If provided, results will be saved to this directory; otherwise, the default output path will be used
    Returns:
        Predicted binding affinities saved as an Excel file.
    """
    import os
    import sys
    import torch
    import numpy as np
    import glob
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    
    # 设置基础路径
    METABCR_ROOT = "/data/lht/meta_bcr"
    
    # 添加到Python路径以便导入模块
    if METABCR_ROOT not in sys.path:
        sys.path.append(METABCR_ROOT)
    
    # 设置与原始脚本完全相同的参数
    antigen_name = "flu"
    task_name = "bind"
    config_date = "250312"
    
    # 构建完整路径
    # default_input_path = os.path.join(METABCR_ROOT, "Data/FLU_infer/0322_ddg_datasets.csv")
    default_input_path = os.path.join(METABCR_ROOT, "Data/FLU_infer/test000012_flu_dataset.csv")
    # 如果提供了input_path参数，则使用它，否则使用默认路径
    fdir_tst = input_file_path if input_file_path else default_input_path
    
    # 处理输出路径
    if not output_file_path:
        output_file_path = os.path.join(METABCR_ROOT, "output")
    else:
        # 标准化路径分隔符（将Windows的\转换为Linux的/）
        output_file_path = output_file_path.replace('\\', '/')
        # 如果路径以文件扩展名结尾，取其父目录
        if output_file_path.endswith(('.csv', '.xlsx', '.xls')):
            output_file_path = os.path.dirname(output_file_path)
    
    # 检查并创建输出目录（仅在目录不存在时创建）
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path, exist_ok=True)
        print(f"创建输出目录: {output_file_path}")
    
    output_base_dir = os.path.join(output_file_path, "MetaBcr")
    # 检查并创建MetaBcr目录（仅在目录不存在时创建）
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir, exist_ok=True)
        print(f"创建MetaBcr目录: {output_base_dir}")
    
    output_dir = os.path.join(output_base_dir, task_name)

    print('=============' + fdir_tst + '==============')

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 导入必要的模块
    from Config.config import get_config
    
    # 构建配置文件的完整路径
    config_path = os.path.join(METABCR_ROOT, 
                              f"Config/config_five_fold_{antigen_name}_{task_name}_meta_{config_date}_semi_supervise.json")
    
    # 加载配置
    configure = get_config(config_path)
    
    # 导入predict_metabcr模块
    import predict_metabcr
    
    # 保存原始glob函数
    original_glob = glob.glob
    
    # 创建一个自定义glob函数，确保从METABCR_ROOT开始搜索
    def custom_glob(pattern):
        if not os.path.isabs(pattern) and pattern.startswith('Results/'):
            absolute_pattern = os.path.join(METABCR_ROOT, pattern)
            return original_glob(absolute_pattern)
        return original_glob(pattern)
    
    # 替换glob.glob函数为自定义版本
    glob.glob = custom_glob
    
    # 设置测试参数
    fold_set = [0]
    fdir_tsts = [fdir_tst]
    label_str = None
    
    # 执行测试，与原始代码完全相同的循环结构
    for single_fdir_tst in fdir_tsts:
        for fold in fold_set:
            predict_metabcr.test(
                _cfg_=configure,
                antigen_name=antigen_name,
                fold=fold,
                fdir_tst=single_fdir_tst,
                output_dir=output_dir,
                label_str=label_str,
                date=config_date,
                task_name=task_name
            )
    
    # 恢复原始glob函数
    glob.glob = original_glob
    
    # 根据输入文件名生成精确的输出文件名
    input_basename = os.path.basename(fdir_tst).split(".")[0]
    expected_output_filename = f"test_results_{input_basename}_{task_name}_{config_date}_fold0.xlsx"
    expected_output_path = os.path.join(output_dir, expected_output_filename)
    
    # 首先尝试精确匹配
    if os.path.exists(expected_output_path):
        print(f"找到精确匹配的文件: {expected_output_path}")
        
        # 生成对应的CSV文件
        import pandas as pd
        csv_path = expected_output_path.replace('.xlsx', '.csv')
        df = pd.read_excel(expected_output_path)
        df.to_csv(csv_path, index=False)
        print(f"生成CSV文件: {csv_path}")
        
        return expected_output_path
    
    # 如果精确匹配失败，回退到原来的逻辑
    result_files = []
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.xlsx') and 'test_results' in file:
                result_files.append(os.path.join(output_dir, file))
    
    if result_files:
        excel_path = result_files[0]
        
        # 生成对应的CSV文件
        import pandas as pd
        csv_path = excel_path.replace('.xlsx', '.csv')
        df = pd.read_excel(excel_path)
        df.to_csv(csv_path, index=False)
        print(f"生成CSV文件: {csv_path}")
        
        return excel_path
    
    return ""


# 添加生命周期管理
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

@asynccontextmanager
async def fdg_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """处理服务器启动和关闭"""
    print("MetaBcr MCP Server 正在初始化...")
    
    try:
        yield {"initialized": True}
    finally:
        print("MetaBcr MCP Server 正在关闭...")

# 设置生命周期
mcp.lifespan = fdg_lifespan

if __name__ == "__main__":
    print("启动MetaBcr MCP服务器...")
    # 设置MCP标准路径
    # mcp.settings.sse_path = "/_mcp/v1/sse"
    # mcp.settings.message_path = "/_mcp/v1/messages/"
    # 设置网络参数
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8082
    
    # 使用SSE模式启动
    mcp.run(transport="sse")
