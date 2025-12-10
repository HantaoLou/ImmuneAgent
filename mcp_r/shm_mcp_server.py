import logging
import os
import tempfile
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from anarci import anarci
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import urllib.request
from urllib.parse import urlparse
from urllib.error import URLError, HTTPError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建 MCP 服务器实例
mcp = FastMCP("SHM MCP Server")

def download_url_to_temp_file(url: str, default_ext: str = None) -> str:
    """
    下载 HTTP/HTTPS URL 到临时文件
    
    Args:
        url: HTTP/HTTPS URL
        default_ext: 默认文件扩展名（如果 URL 中没有扩展名）
        
    Returns:
        临时文件路径
        
    Raises:
        Exception: 如果下载失败
    """
    try:
        # 从 URL 获取文件扩展名
        parsed_url = urlparse(url)
        url_path = parsed_url.path
        # 获取文件扩展名
        ext = os.path.splitext(url_path)[1]
        if not ext and default_ext:
            ext = default_ext
        elif not ext:
            ext = '.tmp'
        
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_file_path = temp_file.name
        temp_file.close()
        
        # 下载文件
        urllib.request.urlretrieve(url, temp_file_path)
        
        return temp_file_path
    except Exception as e:
        raise Exception(f"Failed to download URL {url}: {str(e)}")


# Pydantic参数模型定义
class ProcessCsvShmArgs(BaseModel):
    """Parameters for CSV SHM processing"""
    input_csv_path: str = Field(
        ...,
        description="Input CSV file path (支持本地路径或 HTTP/HTTPS URL)",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "/path/to/input.csv or https://example.com/data.csv",
            "help_text": "Path to input CSV file containing combine_barcode, Heavy, Light fields (支持本地路径或 HTTP/HTTPS URL)"
        }
    )
    output_csv_path: Optional[str] = Field(
        default=None,
        description="Output CSV file path",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "/path/to/output.csv",
            "help_text": "Optional: output CSV file path. Auto-generated if not provided."
        }
    )

def calculate_sequence_shm_from_results(sequence_id: str, numbering_result, alignment_result) -> int:
    """
    从ANARCI结果计算单个序列的SHM数量
    
    Args:
        sequence_id: 序列标识符
        numbering_result: ANARCI numbering结果
        alignment_result: ANARCI alignment结果
        
    Returns:
        int: SHM数量
    """
    try:
        # 检查是否成功编号
        if not numbering_result or numbering_result is None:
            logger.warning(f"ANARCI无法处理序列: {sequence_id}")
            return 0
            
        # 检查是否有比对详情
        if not alignment_result:
            logger.warning(f"未找到比对详情: {sequence_id}")
            return 0
            
        # 获取第一个域的比对详情
        domain_details = alignment_result[0]  # 第一个域
        
        # 检查是否有种系基因信息
        if 'germlines' not in domain_details:
            logger.warning(f"未找到种系基因信息: {sequence_id}")
            return 0
            
        germline_info = domain_details['germlines']
        
        # 获取V基因一致性信息
        if 'v_gene' not in germline_info:
            logger.warning(f"未找到V基因信息: {sequence_id}")
            return 0
            
        # v_gene的数据结构是: [('human', 'IGHV4-4*02'), 0.9693877551020408]
        # 第二个元素是v_identity值
        v_gene_info = germline_info['v_gene']
        if len(v_gene_info) >= 2 and isinstance(v_gene_info[1], (int, float)):
            v_identity = v_gene_info[1]
        else:
            logger.warning(f"V基因一致性数据格式异常: {sequence_id}, {v_gene_info}")
            return 0
            
        # 获取比对区域的长度
        domain_numbering, start_index, end_index = numbering_result[0]
        aligned_length = end_index - start_index + 1
        
        # 计算SHM数量: (1 - 一致性) × 比对长度
        # v_identity是小数形式(如0.86)，不是百分比
        shm_count = int((1.0 - v_identity) * aligned_length)
        
        return max(0, shm_count)
        
    except Exception as e:
        logger.error(f"计算序列SHM时出错 {sequence_id}: {e}")
        return 0

def calculate_batch_shm(sequences_data: list) -> list:
    """
    批量计算序列的SHM数量
    
    Args:
        sequences_data: [(sequence_id, sequence), ...] 格式的序列列表
        
    Returns:
        list: [shm_count, ...] SHM数量列表，与输入顺序对应
    """
    if not sequences_data:
        return []
        
    # 过滤有效序列
    valid_sequences = []
    valid_indices = []
    
    for i, (seq_id, sequence) in enumerate(sequences_data):
        if sequence and len(str(sequence)) >= 10:
            valid_sequences.append((seq_id, str(sequence)))
            valid_indices.append(i)
    
    if not valid_sequences:
        return [0] * len(sequences_data)
    
    try:
        # 批量调用ANARCI
        results = anarci(valid_sequences, scheme='imgt', output=False, assign_germline=True)
        numbering, alignment_details, hit_tables = results
        
        # 初始化结果列表
        shm_results = [0] * len(sequences_data)
        
        # 处理每个有效序列的结果
        for i, valid_idx in enumerate(valid_indices):
            seq_id = sequences_data[valid_idx][0]
            
            # 获取对应的结果
            numbering_result = numbering[i] if numbering and i < len(numbering) else None
            alignment_result = alignment_details[i] if alignment_details and i < len(alignment_details) else None
            
            # 计算SHM
            shm_count = calculate_sequence_shm_from_results(seq_id, numbering_result, alignment_result)
            shm_results[valid_idx] = shm_count
            
        return shm_results
        
    except Exception as e:
        logger.error(f"批量计算SHM时出错: {e}")
        return [0] * len(sequences_data)

def calculate_shm_pair(heavy_sequence: str, light_sequence: str) -> Tuple[int, int]:
    """
    计算重链和轻链的SHM值
    
    Args:
        heavy_sequence: 重链序列
        light_sequence: 轻链序列
        
    Returns:
        (重链SHM, 轻链SHM)
    """
    # 使用批量处理计算两个序列
    sequences_data = [
        ("heavy", heavy_sequence),
        ("light", light_sequence)
    ]
    shm_results = calculate_batch_shm(sequences_data)
    return shm_results[0], shm_results[1]

@mcp.tool()
def process_csv_shm(args: ProcessCsvShmArgs):
    """
    处理CSV文件，计算SHM值
    
    Args:
        input_csv_path: 输入CSV文件路径，包含combine_barcode、Heavy、Light字段
        output_csv_path: 输出CSV文件路径，如果为None则自动生成
    """
    # 处理 URL 下载
    temp_file_path = None
    actual_csv_path = args.input_csv_path
    
    try:
        if args.input_csv_path.startswith(('http://', 'https://')):
            try:
                temp_file_path = download_url_to_temp_file(args.input_csv_path, '.csv')
                actual_csv_path = temp_file_path
                logger.info(f"已从 URL 下载文件到临时路径: {temp_file_path}")
            except Exception as e:
                logger.error(f"下载 URL 文件失败: {e}")
                return
        
        # 检查文件是否存在
        if not os.path.exists(actual_csv_path):
            logger.error(f"输入文件不存在: {actual_csv_path}")
            return
        
        # 读取输入CSV文件
        try:
            df = pd.read_csv(actual_csv_path)
            logger.info(f"成功读取CSV文件: {actual_csv_path}")
            logger.info(f"数据行数: {len(df)}")
        except Exception as e:
            logger.error(f"读取CSV文件失败: {e}")
            return
        
        # 检查必需的列
        required_columns = ['combine_barcode', 'Heavy', 'Light']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"CSV文件缺少必需的列: {missing_columns}")
            logger.error(f"当前列: {list(df.columns)}")
            return
        
        # 准备结果列表
        results = []
        successful_count = 0
        failed_count = 0
        
        # 准备批量处理的序列数据
        all_sequences = []
        sequence_map = {}  # 映射序列索引到原始行索引和类型
        
        for idx, row in df.iterrows():
            combine_barcode = row['combine_barcode']
            heavy_seq = row['Heavy']
            light_seq = row['Light']
            
            # 检查序列是否为空
            if pd.isna(heavy_seq) or pd.isna(light_seq) or heavy_seq == '' or light_seq == '':
                logger.warning(f"跳过combine_barcode {combine_barcode}: 序列为空")
                results.append({
                    'combine_barcode': combine_barcode,
                    'Heavy': heavy_seq,
                    'Light': light_seq,
                    'H_shm': 0,
                    'L_shm': 0
                })
                failed_count += 1
                continue
            
            # 添加到批量处理列表
            heavy_idx = len(all_sequences)
            all_sequences.append((f"{combine_barcode}_H", str(heavy_seq)))
            sequence_map[heavy_idx] = (idx, 'heavy')
            
            light_idx = len(all_sequences)
            all_sequences.append((f"{combine_barcode}_L", str(light_seq)))
            sequence_map[light_idx] = (idx, 'light')
        
        # 批量计算所有序列的SHM
        if all_sequences:
            logger.info(f"开始批量处理 {len(all_sequences)} 个序列...")
            batch_shm_results = calculate_batch_shm(all_sequences)
            
            # 创建结果字典，按行索引组织
            row_results = {}
            for seq_idx, shm_value in enumerate(batch_shm_results):
                if seq_idx in sequence_map:
                    row_idx, seq_type = sequence_map[seq_idx]
                    if row_idx not in row_results:
                        row_results[row_idx] = {}
                    row_results[row_idx][seq_type] = shm_value
            
            # 处理有序列的行
            for idx, row in df.iterrows():
                if idx in row_results:
                    combine_barcode = row['combine_barcode']
                    heavy_seq = row['Heavy']
                    light_seq = row['Light']
                    
                    heavy_shm = row_results[idx].get('heavy', 0)
                    light_shm = row_results[idx].get('light', 0)
                    
                    results.append({
                        'combine_barcode': combine_barcode,
                        'Heavy': heavy_seq,
                        'Light': light_seq,
                        'H_shm': heavy_shm,
                        'L_shm': light_shm
                    })
                    
                    logger.info(f"处理完成 combine_barcode {combine_barcode}: H_shm={heavy_shm}, L_shm={light_shm}")
                    successful_count += 1
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(results)
        
        # 生成输出文件路径
        output_csv_path = args.output_csv_path
        if output_csv_path is None:
            input_path = Path(args.input_csv_path)
            output_csv_path = input_path.parent / f"{input_path.stem}_shm_results.csv"
        
        # 保存结果
        try:
            result_df.to_csv(output_csv_path, index=False)
            logger.info(f"结果已保存到: {output_csv_path}")
            logger.info(f"处理完成，共处理 {len(result_df)} 条记录")
            logger.info(f"成功处理: {successful_count} 条")
            logger.info(f"处理失败: {failed_count} 条")
            
            # 显示统计信息
            if successful_count > 0:
                successful_results = result_df[result_df['H_shm'] >= 0]
                avg_h_shm = successful_results['H_shm'].mean()
                avg_l_shm = successful_results['L_shm'].mean()
                logger.info(f"平均H_shm: {avg_h_shm:.2f}")
                logger.info(f"平均L_shm: {avg_l_shm:.2f}")
            
        except Exception as e:
            logger.error(f"保存结果文件失败: {e}")
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
    finally:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"已清理临时文件: {temp_file_path}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")


@asynccontextmanager
async def shm_analysis_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """处理服务器启动和关闭"""
    print("SHM MCP Server 正在初始化...")
    
    try:
        yield {"initialized": True}
    finally:
        print("SHM MCP Server 正在关闭...")

# 设置生命周期
mcp.lifespan = shm_analysis_lifespan


if __name__ == "__main__":
    """
    启动 MCP 服务器: python shm_mcp_server.py
    """
    import sys
    # 启动 MCP 服务器
    mcp.settings.host = "127.0.0.1"
    mcp.settings.port = 8001
    mcp.run(transport="streamable-http")