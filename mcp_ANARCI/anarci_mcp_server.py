"""
ANARCI MCP Server

Simple wrapper around ANARCI Python API.
Returns exactly what ANARCI provides - NO hardcoded logic.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from mcp.server.fastmcp import FastMCP

# 创建MCP服务器
mcp = FastMCP("ANARCI Antibody Numbering Server")

# 动态添加ANARCI到Python路径
current_dir = Path(__file__).parent
anarci_lib_path = current_dir / "ANARCI_lineage" / "ANARCI" / "lib"
if anarci_lib_path.exists():
    sys.path.insert(0, str(anarci_lib_path))
else:
    print(f"WARNING: ANARCI library path not found at {anarci_lib_path}")

# 导入ANARCI
try:
    from anarci import anarci, number as anarci_number
    ANARCI_AVAILABLE = True
except ImportError:
    ANARCI_AVAILABLE = False
    print("WARNING: ANARCI not available.")

@mcp.tool()
def number_antibody_batch(
    sequences: List[Dict[str, str]],
    scheme: str = "chothia",
    assign_germline: bool = True
) -> Dict[str, Any]:
    """
    Number antibody sequences using ANARCI API.
    Returns exactly what ANARCI provides.

    Args:
        sequences: [{"id": "seq1", "sequence": "EVQL..."}]
        scheme: chothia, kabat, imgt, martin, aho, wolfguy
        assign_germline: Assign germline genes

    Returns:
        ANARCI results with numbering, alignment details, and hit tables
    """
    start_time = time.time()

    try:
        # Format for ANARCI
        seq_list = [(seq['id'], seq['sequence']) for seq in sequences]

        # Call ANARCI
        numbering, alignment_details, hit_tables = anarci(
            seq_list,
            scheme=scheme,
            output=False,
            assign_germline=assign_germline
        )

        results = []
        for i, seq in enumerate(sequences):
            if numbering[i] is None:
                results.append({
                    "id": seq['id'],
                    "numbered": False,
                    "message": "Not an antibody/TCR sequence"
                })
            else:
                results.append({
                    "id": seq['id'],
                    "numbered": True,
                    "numbering": numbering[i],  # List of domains
                    "alignment_details": alignment_details[i],
                    "hit_tables": hit_tables[i],
                    "scheme": scheme
                })

        return {
            "status": "success",
            "results": results,
            "total": len(sequences),
            "processing_time_ms": (time.time() - start_time) * 1000
        }

    except (ValueError, KeyError, IndexError, RuntimeError) as e:
        return {
            "status": "error",
            "message": str(e),
            "results": []
        }


@mcp.tool()
def number_single_sequence(
    sequence: str,
    scheme: str = "imgt"
) -> Dict[str, Any]:
    """
    Quick numbering for single sequence using ANARCI.

    Args:
        sequence: Antibody sequence
        scheme: Numbering scheme

    Returns:
        ANARCI numbering and chain type
    """

    try:
        numbering, chain_type = anarci_number(sequence, scheme=scheme)

        if numbering is None:
            return {
                "status": "not_numbered",
                "message": "Not an antibody/TCR"
            }

        return {
            "status": "success",
            "numbering": numbering,
            "chain_type": chain_type,
            "scheme": scheme
        }

    except (ValueError, KeyError, IndexError, RuntimeError) as e:
        return {"status": "error", "message": str(e)}


@asynccontextmanager
async def anarci_server_lifespan(_server: FastMCP) -> AsyncIterator[dict]:
    """处理服务器启动和关闭生命周期"""
    print("ANARCI Antibody Numbering Server is initializing...")
    try:
        # 返回服务器状态信息
        yield {
            "initialized": True, 
        }
    finally:
        print("ANARCI Antibody Numbering Server is shutting down...")

# 设置生命周期
mcp.lifespan = anarci_server_lifespan

if __name__ == "__main__":
    print("Starting ANARCI Antibody Numbering Server...")

    # 设置网络参数
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8095

    # 使用SSE模式启动
    mcp.run(transport="sse")
