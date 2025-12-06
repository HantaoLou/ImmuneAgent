"""
FDG MCP Server - Core FDG Tool Wrapper

This server exposes the core FDG (Foldx, DDG, GearBind) process via MCP protocol.
"""
from mcp.server.fastmcp import FastMCP
from Bio.PDB import MMCIFParser, PDBIO
from tqdm import tqdm
from typing import Optional
import os
import sys
import pandas as pd
import re
# 添加项目根目录到Python路径 - 确保这行代码在所有导入语句之前
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import csv_run_af3

# Create MCP server
mcp = FastMCP("Alphafold3 Core Server")


def validate_input_file(file_path: str) -> dict:
    """校验输入文件是否符合标准格式要求
    
    标准字段要求：
    - ID: 抗体标识符（必需）
    - Heavy_Chain: 重链序列（必需）
    - Light_Chain: 轻链序列（必需）
    - Antigen: 抗原名称（可选，如果没有则使用默认值）
    
    Args:
        file_path: 输入文件路径
        
    Returns:
        包含校验结果的字典
    """
    # 定义必需字段
    REQUIRED_FIELDS = ['clone_id', 'Heavy', 'Light']  # 适配csv_run_af3.py的原始字段名
    OPTIONAL_FIELDS = ['Antigen']
    ALL_VALID_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS
    
    try:
        # 读取文件
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        if df.empty:
            return {
                'valid': False,
                'error': '文件为空',
                'missing_fields': REQUIRED_FIELDS,
                'extra_fields': [],
                'row_count': 0
            }
        
        columns = df.columns.tolist()
        
        # 检查必需字段
        missing_fields = [field for field in REQUIRED_FIELDS if field not in columns]
        
        # 检查额外字段（不在标准字段列表中的字段）
        extra_fields = [col for col in columns if col not in ALL_VALID_FIELDS]
        
        # 检查数据完整性
        empty_required_fields = []
        for field in REQUIRED_FIELDS:
            if field in columns and df[field].isna().any():
                empty_required_fields.append(field)
        
        # 获取抗原名称
        antigen_name = "H5N1_TEXAS"  # 默认值
        if 'Antigen' in columns and not df['Antigen'].isna().all():
            # 取第一个非空的抗原名称
            first_antigen = df['Antigen'].dropna().iloc[0] if not df['Antigen'].dropna().empty else antigen_name
            antigen_name = str(first_antigen).strip()
        
        # 判断文件是否有效
        is_valid = (len(missing_fields) == 0 and len(empty_required_fields) == 0)
        
        result = {
            'valid': is_valid,
            'missing_fields': missing_fields,
            'extra_fields': extra_fields,
            'empty_required_fields': empty_required_fields,
            'row_count': len(df),
            'antigen_name': antigen_name,
            'columns': columns
        }
        
        if not is_valid:
            error_msgs = []
            if missing_fields:
                error_msgs.append(f"缺少必需字段: {', '.join(missing_fields)}")
            if empty_required_fields:
                error_msgs.append(f"以下必需字段包含空值: {', '.join(empty_required_fields)}")
            result['error'] = '; '.join(error_msgs)
        
        if extra_fields:
            print(f"警告: 发现额外字段 {extra_fields}，将被忽略")
        
        return result
        
    except Exception as e:
        return {
            'valid': False,
            'error': f'文件读取失败: {str(e)}',
            'missing_fields': REQUIRED_FIELDS,
            'extra_fields': [],
            'empty_required_fields': [],
            'row_count': 0,
            'antigen_name': "H5N1_TEXAS",
            'columns': []
        }


@mcp.tool()
def alphafold3(
    input_file_path: str,
    antigen_name: Optional[str] = "H5N1_TEXAS",
    gpu_device: Optional[str] = "3"
) -> list:
    """Uses AlphaFold3 to predict the 3D structure of antibody sequences from an input Excel file and saves the result as a PDB file.

    This function reads an Excel file containing antibody sequences (heavy and light chains), 
    uses AlphaFold3 to predict the 3D structure of each antibody, and writes the predicted structures to a PDB file.
    AlphaFold3 is a state-of-the-art deep learning model for protein structure prediction.

    Args:
        input_file_path: Path to the input Excel file (.xlsx) or CSV file (.csv) containing antibody sequences.
        antigen_name: Name of the antigen. Optional, will use file content or default value if not provided.
        gpu_device: GPU device ID to use. Default "3".

    Returns:
        .pdb file paths of the results
    """
    # 设置基础路径
    root_dir = '/data/lht/AF3'
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
    
    try:
        # 首先校验输入文件格式
        validation_result = validate_input_file(input_file_path)
        
        if not validation_result['valid']:
            error_msg = f"输入文件格式不符合要求: {validation_result['error']}"
            print(error_msg)
            return {"error": "invalid_file_format", "message": error_msg, "details": validation_result}
        
        print(f"✓ 文件校验通过，共 {validation_result['row_count']} 行数据")
        
        # 使用校验结果中的抗原名称
        if antigen_name:
            antigen_name = validation_result['antigen_name']
        
        print(f"使用抗原名称: {antigen_name}")
        
        # 获取输入文件名
        input_file_name = os.path.basename(input_file_path)
        input_name, suffix = os.path.splitext(input_file_name)
        suffix = suffix.lstrip('.')
        
        # 将输入文件复制到AF3目录以便处理
        csv_dir = os.path.join(root_dir, 'af3_inputs', 'csv_files')
        os.makedirs(csv_dir, exist_ok=True)
        target_path = os.path.join(csv_dir, input_file_name)
        
        if input_file_path != target_path and not os.path.exists(target_path):
            import shutil
            shutil.copy2(input_file_path, target_path)
            print(f"已复制输入文件到 {target_path}")
        
        # 设置输出名称
        output_name = f"{input_name}_{antigen_name}"
        
        # 将root_dir添加到系统路径以便导入
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        
        # 动态设置csv_run_af3的参数
        print(f"调用convert2afformat处理输入文件...")
        
        # 生成动态字母映射 - 使用固定的字母映射
        antigen_letter = antigen_name[:3].upper() if len(antigen_name) >= 3 else "AGN"
        heavy_letter = "H"
        light_letter = "L"
        
        # 构建csv_run_af3期望的HEADER格式
        # 构建csv_header映射，适配csv_run_af3.py中的原始字段名
        csv_header = {
            'clone_id': 'ID',  # 映射clone_id到ID
            'Heavy': heavy_letter,  # 映射Heavy到重链字母
            'Light': light_letter,  # 映射Light到轻链字母
        }
        
        # 如果文件中有Antigen列，添加到header映射
        if 'Antigen' in validation_result['columns']:
            csv_header[antigen_name] = antigen_letter
        
        # 构建传递给convert2afformat的header参数（应该是映射后的值列表）
        header = list(csv_header.values())
        
        print(f"使用的header映射: {csv_header}")
        print(f"传递的header: {header}")
        
        # 设置csv_run_af3的全局变量
        csv_run_af3.ANTIG_LETTER = antigen_letter
        csv_run_af3.HEAVY_LETTER = heavy_letter
        csv_run_af3.LIGHT_LETTER = light_letter
        csv_run_af3.ANTIGEN_NAME = antigen_name
        
        # 设置HEADER变量以匹配csv_run_af3的期望格式
        csv_run_af3.HEADER = csv_header
        
        # 设置ROOT_DIR和其他必要的变量
        csv_run_af3.ROOT_DIR = root_dir
        csv_run_af3.PDB_DIR = os.path.join(root_dir, 'af3_inputs', 'pdb_files')
        csv_run_af3.CSV_DIR = csv_dir
        csv_run_af3.JSON_DIR = os.path.join(root_dir, 'json_files')
        csv_run_af3.MODEL_DIR = os.path.join(root_dir, 'af3_model')
        csv_run_af3.OUT_DIR = os.path.join(root_dir, 'af3_outputs')
        csv_run_af3.PUBLIC_DATA_DIR = os.path.join(root_dir, 'public_databases')
        
        # 调用convert2afformat函数
        csv_run_af3.convert2afformat(
            suffix=suffix,
            input_name=input_name,
            output_name=output_name,
            header=header
        )
        
        print(f"convert2afformat执行完成")
        
        # 设置Json文件目录和输出目录
        json_dir = os.path.join(root_dir, 'json_files', output_name)
        output_dir = os.path.join(root_dir, 'af3_outputs', output_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理JSON文件并调用AlphaFold3
        try:
            # 检查JSON目录是否存在
            if not os.path.exists(json_dir):
                print(f"警告: JSON目录 {json_dir} 不存在")
            else:
                # 处理JSON文件
                json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
                print(f"找到 {len(json_files)} 个JSON文件")
                
                for json_file in json_files:
                    json_path = os.path.join(json_dir, json_file)
                    sample_name = os.path.splitext(json_file)[0]
                    output_subdir = os.path.join(output_dir, sample_name)
                    
                    # 检查是否已经处理过
                    if not os.path.exists(output_subdir):
                        print(f"处理JSON文件: {json_path}")
                        
                        # 记录执行前的目录状态
                        existing_dirs = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()
                        
                        # 使用系统命令调用AlphaFold3
                        cmd = f"CUDA_VISIBLE_DEVICES={gpu_device} /data_new/lht/.conda/envs/alphafold3_venv/bin/python alphafold3/run_alphafold.py --json_path={json_path} --model_dir={os.path.join(root_dir, 'af3_model')} --db_dir={os.path.join(root_dir, 'public_databases')} --gpu_device=0 --output_dir={output_dir}"
                        print(f"执行命令: {cmd}")
                        
                        # 执行AlphaFold3推理
                        import subprocess
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                        if result.returncode != 0:
                            print(f"AlphaFold3执行失败: {result.stderr}")
                        else:
                            print(f"AlphaFold3执行成功: {json_file}")
                            
                            # 检测新生成的目录
                            current_dirs = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()
                            new_dirs = current_dirs - existing_dirs
                            
                            # 找到与当前样本相关的新目录
                            sample_related_dirs = [d for d in new_dirs if sample_name.lower() in d.lower()]
                            if sample_related_dirs:
                                # 选择最新的目录（通常只有一个）
                                actual_output_dir = sorted(sample_related_dirs)[-1]
                                actual_output_path = os.path.join(output_dir, actual_output_dir)
                                
                                # 存储实际生成的目录路径映射
                                mapping_file = os.path.join(output_dir, '.af3_output_mapping.json')
                                mapping = {}
                                if os.path.exists(mapping_file):
                                    import json
                                    with open(mapping_file, 'r') as f:
                                        mapping = json.load(f)
                                
                                mapping[sample_name] = actual_output_path
                                
                                import json
                                with open(mapping_file, 'w') as f:
                                    json.dump(mapping, f, indent=2)
                                
                                print(f"记录目录映射: {sample_name} -> {actual_output_path}")
                            else:
                                print(f"警告: 未找到样本 {sample_name} 对应的新生成目录")
                    else:
                        print(f"跳过已处理的文件: {json_file}")
        
        except Exception as e:
            print(f"JSON处理过程中发生错误: {str(e)}")
        
        # 统一的文件收集和转换逻辑
        return collect_and_convert_files(output_dir, json_dir)
        
    except ImportError as e:
        error_msg = f"导入csv_run_af3模块失败: {str(e)}"
        print(f"错误: {error_msg}")
        return {"error": error_msg, "type": "import_error"}
    
    except FileNotFoundError as e:
        error_msg = f"文件未找到: {str(e)}"
        print(f"错误: {error_msg}")
        return {"error": error_msg, "type": "file_not_found"}
    
    except pd.errors.EmptyDataError:
        error_msg = "输入文件为空或格式不正确"
        print(f"错误: {error_msg}")
        return {"error": error_msg, "type": "empty_data"}
    
    except pd.errors.ParserError as e:
        error_msg = f"文件解析错误: {str(e)}"
        print(f"错误: {error_msg}")
        return {"error": error_msg, "type": "parser_error"}
    
    except Exception as e:
        error_msg = f"处理过程中发生错误: {str(e)}"
        print(f"错误: {error_msg}")
        return {"error": error_msg, "type": "general_error"}


def collect_and_convert_files(output_dir, json_dir):
    """
    收集并转换AlphaFold3输出文件
    使用存储的目录映射直接定位文件，避免复杂的匹配逻辑
    """
    import json
    
    try:
        # 检查映射文件是否存在
        mapping_file = os.path.join(output_dir, '.af3_output_mapping.json')
        if not os.path.exists(mapping_file):
            return {"error": "no_mapping_file", "message": "未找到目录映射文件，请先运行AlphaFold3推理"}
        
        # 读取目录映射
        with open(mapping_file, 'r') as f:
            directory_mapping = json.load(f)
        
        # 获取JSON文件列表
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        if not json_files:
            return {"error": "no_json_files", "message": f"在 {json_dir} 中未找到JSON文件"}
        
        pdb_files = []
        
        for json_file in json_files:
            sample_name = os.path.splitext(json_file)[0]
            
            # 从映射中获取实际的输出目录
            if sample_name in directory_mapping:
                actual_output_path = directory_mapping[sample_name]
                print(f"使用映射目录: {sample_name} -> {actual_output_path}")
            else:
                # 如果映射中没有，尝试查找传统的非时间戳目录
                fallback_dir = os.path.join(output_dir, sample_name)
                if os.path.exists(fallback_dir):
                    actual_output_path = fallback_dir
                    print(f"使用传统目录: {sample_name} -> {actual_output_path}")
                else:
                    print(f"警告: 未找到样本 {sample_name} 的输出目录")
                    continue
            
            # 查找CIF文件
            cif_files = []
            if os.path.exists(actual_output_path):
                for file in os.listdir(actual_output_path):
                    if file.endswith('.cif'):
                        cif_files.append(os.path.join(actual_output_path, file))
            
            if not cif_files:
                print(f"警告: 在 {actual_output_path} 中未找到CIF文件")
                continue
            
            # 转换CIF文件为PDB格式
            for cif_file in cif_files:
                try:
                    base_name = os.path.splitext(os.path.basename(cif_file))[0]
                    pdb_file_path = os.path.join(actual_output_path, f"{base_name}.pdb")
                    
                    # 如果PDB文件已存在，直接添加到列表
                    if os.path.exists(pdb_file_path):
                        pdb_files.append(pdb_file_path)
                        print(f"PDB文件已存在: {pdb_file_path}")
                        continue
                    
                    # 使用Bio.PDB转换CIF到PDB
                    parser = MMCIFParser()
                    structure = parser.get_structure("structure_id", cif_file)
                    
                    io = PDBIO()
                    io.set_structure(structure)
                    io.save(pdb_file_path)
                    
                    pdb_files.append(pdb_file_path)
                    print(f"成功转换: {cif_file} -> {pdb_file_path}")
                except Exception as e:
                    print(f"转换CIF文件时出错 {cif_file}: {str(e)}")
        
        if not pdb_files:
            return {"error": "no_pdb_files", "message": "未生成任何PDB文件"}
        
        return {"pdb_files": pdb_files, "type": "success"}
        
    except Exception as e:
        return {"error": "collection_failed", "message": f"文件收集失败: {str(e)}"}

@asynccontextmanager
async def fdg_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """处理服务器启动和关闭"""
    print("Alphafold3 MCP Server 正在初始化...")
    try:
        yield {"initialized": True}
    finally:
        print("Alphafold3 MCP Server 正在关闭...")

# 设置生命周期
mcp.lifespan = fdg_lifespan

if __name__ == "__main__":
    print("启动Alphafold3 MCP服务器...")
    # 设置网络参数
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 18084
    
    # 使用SSE模式启动
    mcp.run(transport="sse")
