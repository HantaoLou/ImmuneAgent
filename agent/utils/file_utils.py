"""
文件工具函数

提供文件相关的通用工具函数。
"""

import re
from typing import Dict, List, Optional
from pathlib import Path


def find_files(
    file_paths: Optional[Dict[str, str]], user_input: Optional[str]
) -> List[str]:
    """
    从 file_paths 和 user_input 中提取所有文件来源

    Args:
        file_paths: 文件路径字典，如 {"input_csv": "/path/to/file.csv"}
        user_input: 用户输入，可能包含 URL

    Returns:
        文件路径/URL 列表
    """
    files = []

    # 1. 从 file_paths 提取
    if file_paths:
        files.extend(file_paths.values())

    # 2. 从 user_input 提取 URL
    if user_input:
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, user_input)
        files.extend(urls)

    return files


def extract_urls(text: str) -> List[str]:
    """
    从文本中提取所有 URL

    Args:
        text: 输入文本

    Returns:
        URL 列表
    """
    if not text:
        return []

    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def get_file_extension(file_path: str) -> str:
    """
    获取文件扩展名

    Args:
        file_path: 文件路径或 URL

    Returns:
        文件扩展名（小写，不含点），如 "csv", "fasta"
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    # 处理 URL 中的查询参数
    if "?" in ext:
        ext = ext.split("?")[0]
    return ext.lstrip(".")


def is_bioinformatics_file(file_path: str) -> bool:
    """
    判断是否为生物信息学常见文件格式

    Args:
        file_path: 文件路径或 URL

    Returns:
        是否为生物信息学文件
    """
    bio_extensions = {
        "fasta",
        "fa",
        "fna",
        "faa",
        "ffn",
        "frn",
        "fastq",
        "fq",
        "bam",
        "sam",
        "cram",
        "vcf",
        "bcf",
        "bed",
        "gff",
        "gff3",
        "gtf",
        "pdb",
        "cif",
        "rds",
        "rdata",
        "h5",
        "h5ad",
        "hdf5",
        "mtx",
        "tsv",
        "csv",
    }
    ext = get_file_extension(file_path)
    return ext in bio_extensions


__all__ = [
    "find_files",
    "extract_urls",
    "get_file_extension",
    "is_bioinformatics_file",
]
