"""
IgBLAST MCP Server - V(D)J Analysis Tool Wrapper

This server exposes the IgBLAST + ChangeO pipeline via MCP protocol.
V(D)J recombination analysis using IgBLAST + ChangeO pipeline.
Returns AIRR format output - NO hardcoded V(D)J logic.
"""

from mcp.server.fastmcp import FastMCP
import subprocess
import tempfile
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import uuid
import time

# Import configuration
from config.config import TEMP_DIR as TEMP_FILES_DIR, IGBLAST_ROOT, IGBLAST_DB, IGBLAST_OPTIONAL

# Create MCP server
mcp = FastMCP("IgBLAST V(D)J Analysis Server")

# Configuration paths
IGBLAST_BIN = "igblastn"  # Use conda-installed igblastn
# Use ChangeO scripts - try multiple locations
# Check conda installation first, then local copy
_conda_makedb = Path("/data_new/lht/.conda/envs/antibody_venv/bin/MakeDb.py")
_local_makedb = Path("/data_new/hd/server/mcp_Igblast/igblast_changeO/MakeDb.py")

if _conda_makedb.exists():
    CHANGEO_MAKEDB = str(_conda_makedb)
elif _local_makedb.exists():
    CHANGEO_MAKEDB = str(_local_makedb)
else:
    CHANGEO_MAKEDB = "MakeDb.py"  # Fall back to PATH


@mcp.tool()
def analyze_vdj_batch(
    sequences: List[Dict[str, str]],
    organism: str = "human",
    receptor_type: str = "Ig",
    locus: str = "IGH"
) -> Dict[str, Any]:
    """
    V(D)J recombination analysis using IgBLAST + ChangeO.
    Returns AIRR format results.

    Args:
        sequences: [{"id": "seq1", "sequence": "ATGC..."}] - NUCLEOTIDE sequences!
        organism: human, mouse, rabbit, rat, rhesus, pig
        receptor_type: Ig or TCR
        locus: IGH, IGK, IGL, TRA, TRB, TRG, TRD

    Returns:
        AIRR format results from ChangeO
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]

    try:
        # 1. Write FASTA
        fasta_file = TEMP_FILES_DIR / f"igblast_input_{session_id}.fasta"
        with open(fasta_file, 'w') as f:
            for seq in sequences:
                f.write(f">{seq['id']}\n{seq['sequence']}\n")

        # 2. Run IgBLAST
        igblast_out = TEMP_FILES_DIR / f"igblast_output_{session_id}.txt"

        germline_v = IGBLAST_DB / f"imgt_{organism}_ig_v"
        germline_d = IGBLAST_DB / f"imgt_{organism}_ig_d"
        germline_j = IGBLAST_DB / f"imgt_{organism}_ig_j"
        aux_file = IGBLAST_OPTIONAL / f"{organism}_gl.aux"

        # Run IgBLAST - Simplified command without IGDATA environment variable
        # 构建auxiliary_data文件路径
        aux_file = IGBLAST_OPTIONAL / f"{organism}_gl.aux"
        
        igblast_cmd = [
            IGBLAST_BIN,
            "-germline_db_V", str(germline_v),
            "-germline_db_D", str(germline_d),
            "-germline_db_J", str(germline_j),
            "-organism", organism,
            "-domain_system", "imgt",
            "-ig_seqtype", receptor_type,
            "-auxiliary_data", str(aux_file),
            "-query", str(fasta_file),
            "-show_translation",
            "-outfmt", "7 std qseq sseq btop",
            "-out", str(igblast_out)
        ]

        subprocess.run(igblast_cmd, check=True, timeout=600, capture_output=True)

        # 3. Run ChangeO MakeDb to parse IgBLAST output
        changeo_prefix = f"changeo_{session_id}"

        v_fasta = IGBLAST_DB / f"imgt_{organism}_ig_v.fasta"
        d_fasta = IGBLAST_DB / f"imgt_{organism}_ig_d.fasta"
        j_fasta = IGBLAST_DB / f"imgt_{organism}_ig_j.fasta"

        makedb_cmd = [
            CHANGEO_MAKEDB,
            "igblast",
            "-i", str(igblast_out),
            "-r", str(v_fasta), str(d_fasta), str(j_fasta),
            "-s", str(fasta_file),
            "--format", "airr",
            "--partial",  # Allow sequences without full IMGT-gapped references
            "--outdir", str(TEMP_FILES_DIR),
            "--outname", changeo_prefix
        ]

        subprocess.run(makedb_cmd, check=True, timeout=600, capture_output=True)

        # 4. Read AIRR TSV output
        airr_file = TEMP_FILES_DIR / f"{changeo_prefix}_db-pass.tsv"

        if not airr_file.exists():
            return {
                "status": "error",
                "message": "ChangeO did not produce output file"
            }

        df = pd.read_csv(airr_file, sep='\t')

        # Convert to list of dictionaries (AIRR format)
        results = df.to_dict('records')

        # 5. Cleanup
        fasta_file.unlink(missing_ok=True)
        igblast_out.unlink(missing_ok=True)
        airr_file.unlink(missing_ok=True)

        return {
            "status": "success",
            "results": results,
            "total_sequences": len(sequences),
            "processed": len(results),
            "format": "AIRR",
            "processing_time_ms": (time.time() - start_time) * 1000
        }

    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "error_type": "subprocess_failed",
            "message": str(e),
            "stderr": e.stderr.decode() if e.stderr else "",
            "results": []
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": "unknown",
            "message": str(e),
            "results": []
        }


@mcp.tool()
def extract_cdr3_from_airr(
    airr_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extract CDR3 information from AIRR format results.

    Args:
        airr_results: AIRR format results from analyze_vdj_batch

    Returns:
        CDR3 sequences and metadata
    """
    try:
        cdr3_data = []

        for record in airr_results:
            cdr3_data.append({
                "sequence_id": record.get("sequence_id"),
                "junction": record.get("junction"),  # CDR3 nucleotide
                "junction_aa": record.get("junction_aa"),  # CDR3 amino acid
                "junction_length": record.get("junction_length"),
                "productive": record.get("productive"),
                "v_call": record.get("v_call"),
                "j_call": record.get("j_call"),
                "stop_codon": record.get("stop_codon"),
                "vj_in_frame": record.get("vj_in_frame")
            })

        return {
            "status": "success",
            "cdr3_results": cdr3_data,
            "total": len(cdr3_data)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# 添加生命周期管理
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

@asynccontextmanager
async def igblast_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """处理服务器启动和关闭"""
    print("IgBLAST MCP Server 正在初始化...")
    
    try:
        yield {"initialized": True}
    finally:
        print("IgBLAST MCP Server 正在关闭...")

# 设置生命周期
mcp.lifespan = igblast_lifespan

if __name__ == "__main__":
    print("启动IgBLAST MCP服务器...")
    # 设置网络参数
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8001
    
    # 使用SSE模式启动
    mcp.run(transport="sse")
