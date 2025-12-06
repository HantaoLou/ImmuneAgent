"""
SAbDab MCP Server

Downloads data from Structural Antibody Database.
Focus: Download CSV, PDB structures, and database dumps.

This server wraps the core functionality from sabdab_core.py with MCP decorators.
"""

from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, Optional

# Import core functions
from core.sabdab_core import (
    download_sabdab_summary_csv as _download_sabdab_summary_csv,
    download_pdb_structure as _download_pdb_structure,
    download_sabdab_dataset as _download_sabdab_dataset,
    get_sabdab_statistics as _get_sabdab_statistics
)

# Create MCP server
mcp = FastMCP("SAbDab Database Download Server")


@mcp.tool()
def download_sabdab_summary_csv(
    filters: Optional[Dict[str, str]] = None,
    save_file: bool = True
) -> Dict[str, Any]:
    """
    Download SAbDab summary data as CSV.

    Args:
        filters: Optional filters like {"resolution": "<2.5", "antigen": "yes"}
        save_file: Whether to save the file to disk (default: True)

    Returns:
        {
            "status": "success",
            "csv_content": "pdb,Hchain,Lchain,...",
            "num_entries": 1234,
            "file_size_bytes": 56789,
            "file_info": {
                "file_path": "/path/to/file.csv",
                "file_size_bytes": 56789,
                "created_at": "2024-12-01T14:30:22Z"
            }
        }
    """
    return _download_sabdab_summary_csv(filters, save_file)


@mcp.tool()
def download_pdb_structure(
    pdb_id: str,
    numbering_scheme: str = "chothia",
    save_file: bool = True
) -> Dict[str, Any]:
    """
    Download PDB structure from SAbDab with specified numbering.

    Args:
        pdb_id: PDB ID (e.g., "6m0j")
        numbering_scheme: chothia, kabat, or imgt
        save_file: Whether to save the file to disk (default: True)

    Returns:
        {
            "status": "success",
            "pdb_id": "6m0j",
            "pdb_content": "ATOM   1  N ...",
            "numbering_scheme": "chothia",
            "file_size_bytes": 123456,
            "file_info": {
                "file_path": "/path/to/file.pdb",
                "file_size_bytes": 123456,
                "created_at": "2024-12-01T14:30:22Z"
            }
        }
    """
    return _download_pdb_structure(pdb_id, numbering_scheme, save_file)


@mcp.tool()
def download_sabdab_dataset(
    dataset_type: str = "all",
    output_format: str = "csv",
    save_file: bool = True
) -> Dict[str, Any]:
    """
    Download complete SAbDab datasets.

    Args:
        dataset_type: all, antigen_bound, nanobodies, etc.
        output_format: csv, json, or fasta
        save_file: Whether to save the file to disk (default: True)

    Returns:
        Dataset content and metadata with optional file info
    """
    return _download_sabdab_dataset(dataset_type, output_format, save_file)


@mcp.tool()
def get_sabdab_statistics(save_file: bool = True) -> Dict[str, Any]:
    """
    Get SAbDab database statistics.

    Args:
        save_file: Whether to save the statistics to a JSON file (default: True)

    Returns:
        Database statistics and metadata with optional file info
    """
    return _get_sabdab_statistics(save_file)


# 添加生命周期管理
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

@asynccontextmanager
async def sabdab_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """处理服务器启动和关闭"""
    print("SAbDab MCP Server 正在初始化...")
    
    try:
        yield {"initialized": True}
    finally:
        print("SAbDab MCP Server 正在关闭...")

# 设置生命周期
mcp.lifespan = sabdab_lifespan


if __name__ == "__main__":
    print("启动SAbDab MCP服务器...")
    # 设置MCP标准路径
    # mcp.settings.sse_path = "/_mcp/v1/sse"
    # mcp.settings.message_path = "/_mcp/v1/messages/"
    # 设置网络参数
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8098
    
    # 使用SSE模式启动
    mcp.run(transport="sse")
