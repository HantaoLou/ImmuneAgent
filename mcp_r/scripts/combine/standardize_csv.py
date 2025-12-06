"""
标准化CSV文件工具
作者: Python专家
功能: 将指定字段组合成combine_barcode字段并更新到CSV文件
"""

import pandas as pd
import os
from typing import List, Optional


def standardize_csv(bcr_file_path: str, 
                   combine_fields: List[str], 
                   output_path: Optional[str] = None, 
                   separator: str = "_") -> str:
    """
    标准化CSV文件 - 组合字段生成combine_barcode
    
    Args:
        bcr_file_path (str): BCR文件路径
        combine_fields (List[str]): 需要组合的字段名列表
        output_path (Optional[str]): 输出文件路径，默认覆盖原文件
        separator (str): 字段连接符，默认为"_"
    
    Returns:
        str: 输出文件路径
    
    Raises:
        FileNotFoundError: 当BCR文件不存在时
        ValueError: 当字段不存在或参数无效时
    """
    
    # 参数验证
    if not os.path.exists(bcr_file_path):
        raise FileNotFoundError(f"错误: BCR文件不存在: {bcr_file_path}")
    
    if not combine_fields:
        raise ValueError("错误: 必须提供至少一个字段名")
    
    # 读取BCR文件
    print(f"读取BCR文件: {bcr_file_path}")
    try:
        df = pd.read_csv(bcr_file_path)
    except Exception as e:
        raise ValueError(f"读取BCR文件失败: {e}")
    
    print(f"BCR文件包含 {len(df)} 行数据")
    
    # 获取可用字段
    available_fields = df.columns.tolist()
    print(f"当前BCR包含字段: {available_fields}")
    
    # 验证字段是否存在
    missing_fields = [field for field in combine_fields if field not in available_fields]
    if missing_fields:
        print(f"错误: 以下字段在CSV文件中不存在: {missing_fields}")
        print(f"可用字段: {available_fields}")
        raise ValueError(f"字段不存在: {missing_fields}")
    
    print(f"字段验证通过，将组合以下字段: {combine_fields}")
    
    # 生成combine_barcode字段
    print("开始生成combine_barcode字段...")
    
    # 将所有字段转换为字符串并组合
    combine_values = []
    for _, row in df.iterrows():
        field_values = [str(row[field]) for field in combine_fields]
        combined_value = separator.join(field_values)
        combine_values.append(combined_value)
    
    # 添加新字段到DataFrame
    df['combine_barcode'] = combine_values
    
    print("combine_barcode字段生成完成")
    
    # 显示示例值
    print("示例combine_barcode值:")
    sample_values = df['combine_barcode'].head().tolist()
    for i, value in enumerate(sample_values, 1):
        print(f"  {i}. {value}")
    
    # 统计信息
    total_rows = len(df)
    unique_barcodes = df['combine_barcode'].nunique()
    duplicate_rate = (total_rows - unique_barcodes) / total_rows * 100
    
    print(f"\n统计信息:")
    print(f"总行数: {total_rows}")
    print(f"唯一combine_barcode数: {unique_barcodes}")
    print(f"重复率: {duplicate_rate:.2f}%")
    
    # 保存文件
    if output_path is None:
        output_path = bcr_file_path
    else:
        # 检查output_path是否为目录
        if os.path.isdir(output_path):
            # 从原文件路径提取文件名
            filename = os.path.basename(bcr_file_path)
            output_path = os.path.join(output_path, filename)
    
    print(f"\n保存更新后的CSV文件到: {output_path}")
    try:
        df.to_csv(output_path, index=False)
        print("标准化完成!")
    except Exception as e:
        raise ValueError(f"保存文件失败: {e}")
    
    print(f"最终CSV包含 {len(df.columns)} 个字段")
    print(f"字段列表: {df.columns.tolist()}")
    
    return output_path


def preview_csv_fields(csv_file_path: str) -> None:
    """
    预览CSV文件的字段信息
    
    Args:
        csv_file_path (str): CSV文件路径
    """
    
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件不存在: {csv_file_path}")
        return
    
    try:
        df = pd.read_csv(csv_file_path)
        
        print("预览CSV文件字段信息:")
        print(f"文件: {csv_file_path}")
        print(f"行数: {len(df)}")
        print(f"字段数: {len(df.columns)}")
        print("\n可用字段列表:")
        
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            # 获取示例值（非空值）
            sample_values = df[col].dropna().head(3).tolist()
            sample_str = ", ".join([str(v) for v in sample_values])
            
            print(f" {i:2d}. {col:<20} [{dtype}] 示例: {sample_str}")
            
    except Exception as e:
        print(f"读取文件失败: {e}")


def show_usage_example():
    """
    显示使用示例
    """
    print("\n=== 使用示例 ===")
    print("\n1. 预览CSV文件字段:")
    print('preview_csv_fields("path/to/your/file.csv")')
    print("\n2. 标准化CSV文件 - 单个字段:")
    print('standardize_csv("path/to/your/file.csv", ["field1"])')
    print("\n3. 标准化CSV文件 - 多个字段:")
    print('standardize_csv("path/to/your/file.csv", ["field1", "field2", "field3"])')
    print("\n4. 指定输出文件:")
    print('standardize_csv("input.csv", ["field1", "field2"], "output.csv")')
    print("\n5. 自定义分隔符:")
    print('standardize_csv("input.csv", ["field1", "field2"], separator="-")')


if __name__ == "__main__":
    # 显示使用示例
    show_usage_example()
    
    # 示例用法（注释掉，避免意外执行）
    # preview_csv_fields("example.csv")
    standardize_csv("D:\\PartTimeJob\\antibody_gen\\mcp_r\\Age_Bcells\\age_bcells_data\\AgeB_BCR_Test.csv", ["Batch", "barcode"])