"""
Cell Type Annotation MCP Server

This server provides automated and manual cell type annotation tools for scRNA-seq data.
Supports SingleR reference-based annotation, marker detection, and validation.
"""

from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("Cell Type Annotation Server")

def load_config():
    """Load configuration from config.json"""
    import json
    from pathlib import Path
    
    config_path = Path(__file__).parent / "config" / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {
        "base_dir": str(Path(__file__).parent),
        "scripts_dir": "scripts",
        "output_dir": "output",
        "reference_data_dir": "reference_data"
    }

def run_r_script(
    script_name: str,
    input_file: str,
    timeout: int = 3600,
    **kwargs
):
    """
    Generic R script execution function

    Args:
        script_name: R script name (without .R extension)
        input_file: Input RDS file path
        timeout: Execution timeout in seconds (default 3600)
        **kwargs: Additional parameters passed as JSON

    Returns:
        Dictionary with status, generated files, and messages
    """
    import os
    import json
    import subprocess
    from pathlib import Path
    
    # Check input file
    if not os.path.exists(input_file):
        return {
            "status": "error",
            "message": f"Input file does not exist: {input_file}",
            "generated_files": []
        }

    # Load config
    config = load_config()
    working_dir = Path(__file__).parent

    # R script path
    r_script_path = working_dir / config["scripts_dir"] / f"{script_name}.R"

    # Check R script exists
    if not r_script_path.exists():
        return {
            "status": "error",
            "message": f"R script does not exist: {r_script_path}",
            "generated_files": []
        }

    # Prepare parameters JSON
    params = {"input_file": input_file, **kwargs}
    params_json = json.dumps(params)

    try:
        # Execute R script
        result = subprocess.run(
            ["Rscript", str(r_script_path), params_json],
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )

        # Check execution result
        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"R script execution failed (return code: {result.returncode})",
                "stderr": result.stderr,
                "stdout": result.stdout,
                "generated_files": []
            }

        # Collect generated files
        output_dir = working_dir / config["output_dir"] / script_name
        generated_files = []

        if output_dir.exists():
            # CSV files
            csv_files = list(output_dir.glob("*.csv"))
            generated_files.extend([str(f) for f in csv_files])

            # RDS files
            rds_files = list(output_dir.glob("*.rds"))
            generated_files.extend([str(f) for f in rds_files])

            # PDF files
            pdf_files = list((output_dir / "plots").glob("*.pdf")) if (output_dir / "plots").exists() else []
            generated_files.extend([str(f) for f in pdf_files])

            # PNG files
            png_files = list((output_dir / "plots").glob("*.png")) if (output_dir / "plots").exists() else []
            generated_files.extend([str(f) for f in png_files])

            # JSON files
            json_files = list(output_dir.glob("*.json"))
            generated_files.extend([str(f) for f in json_files])

        return {
            "status": "success",
            "message": f"{script_name} analysis completed successfully!",
            "generated_files": generated_files,
            "stdout": result.stdout,
            "output_dir": str(output_dir)
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": f"R script execution timeout (exceeded {timeout} seconds)",
            "generated_files": []
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error executing R script: {str(e)}",
            "generated_files": []
        }

@mcp.tool()
def run_singler_annotation(
    input_rds: str,
    reference_dataset: str = "HumanPrimaryCellAtlasData",
    label_type: str = "label.main",
    cluster_column: str = "seurat_clusters"
):
    """
    SingleR automated cell type annotation

    Performs reference-based automated annotation using SingleR with multiple
    reference datasets from celldex package.

    Args:
        input_rds: Path to input Seurat RDS file
        reference_dataset: Reference dataset name, one of:
            - HumanPrimaryCellAtlasData (default, general human cell types)
            - BlueprintEncodeData (immune and stromal cells)
            - MonacoImmuneData (immune cell types, detailed)
            - DatabaseImmuneCellExpressionData (immune cells)
        label_type: Annotation granularity:
            - label.main (broad cell types)
            - label.fine (detailed subtypes)
        cluster_column: Metadata column containing cluster IDs

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - message: Execution result
        - generated_files: List of output files (CSV, RDS with annotations)
        - annotation_summary: Cell type counts per cluster
    """
    return run_r_script(
        "run_singler_annotation",
        input_rds,
        reference_dataset=reference_dataset,
        label_type=label_type,
        cluster_column=cluster_column
    )

@mcp.tool()
def detect_cluster_markers(
    input_rds: str,
    test_use: str = "wilcox",
    only_pos: bool = True,
    min_pct: float = 0.25,
    logfc_threshold: float = 0.5,
    top_n: int = 10
):
    """
    Identify cluster-specific marker genes

    Uses Seurat's FindAllMarkers to detect differentially expressed genes
    for each cluster. Returns top markers for manual annotation.

    Args:
        input_rds: Path to input Seurat RDS file
        test_use: Statistical test, one of:
            - wilcox (default, Wilcoxon rank sum test)
            - bimod (likelihood-ratio test)
            - roc (ROC curve analysis)
            - t (t-test)
            - MAST (MAST framework)
        only_pos: Only return positive markers (upregulated)
        min_pct: Minimum percentage of cells expressing gene
        logfc_threshold: Minimum log2 fold change threshold
        top_n: Number of top markers per cluster to return

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - message: Execution result
        - generated_files: CSV with all markers, top markers per cluster
        - marker_summary: Top N markers per cluster
    """
    return run_r_script(
        "detect_cluster_markers",
        input_rds,
        test_use=test_use,
        only_pos=only_pos,
        min_pct=min_pct,
        logfc_threshold=logfc_threshold,
        top_n=top_n
    )

@mcp.tool()
def annotate_by_markers(
    input_rds: str,
    marker_list: dict,
    cluster_column: str = "seurat_clusters",
    new_column: str = "manual_celltype"
):
    """
    Manual cell type annotation based on marker genes

    Annotates clusters based on a provided marker gene list mapping
    cluster IDs to cell types.

    Args:
        input_rds: Path to input Seurat RDS file
        marker_list: Dictionary mapping cluster IDs to cell types
            Example: {"0": "T cells", "1": "B cells", "2": "Macrophages"}
        cluster_column: Metadata column containing cluster IDs
        new_column: Name for new annotation column in metadata

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - message: Execution result
        - generated_files: Updated RDS file with annotations
        - annotation_mapping: Applied marker-based annotations
    """
    return run_r_script(
        "annotate_by_markers",
        input_rds,
        marker_list=marker_list,
        cluster_column=cluster_column,
        new_column=new_column
    )

@mcp.tool()
def validate_annotation(
    input_rds: str,
    annotation_column1: str,
    annotation_column2: str = None,
    reference_dataset: str = "MonacoImmuneData"
):
    """
    Cross-reference annotation validation

    Validates existing annotations by comparing with SingleR predictions
    using a different reference dataset, or compares two annotation columns.

    Args:
        input_rds: Path to input Seurat RDS file with existing annotations
        annotation_column1: First annotation column to validate
        annotation_column2: Second annotation column for comparison (optional)
        reference_dataset: Reference for SingleR validation if column2 not provided

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - message: Execution result
        - generated_files: Validation report, confusion matrix
        - validation_metrics: Agreement scores, discrepancies
    """
    return run_r_script(
        "validate_annotation",
        input_rds,
        annotation_column1=annotation_column1,
        annotation_column2=annotation_column2,
        reference_dataset=reference_dataset
    )

@mcp.tool()
def score_annotation_confidence(
    input_rds: str,
    annotation_column: str,
    marker_genes: dict = None
):
    """
    Calculate annotation confidence scores

    Scores cell type annotations based on:
    - SingleR confidence scores
    - Marker gene expression consistency
    - Cluster homogeneity

    Args:
        input_rds: Path to input Seurat RDS file with annotations
        annotation_column: Column containing cell type annotations
        marker_genes: Optional dict mapping cell types to marker genes
            Example: {"T cells": ["CD3D", "CD3E"], "B cells": ["CD79A", "MS4A1"]}

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - message: Execution result
        - generated_files: Confidence scores CSV, visualization plots
        - confidence_summary: Per-cluster confidence metrics
    """
    return run_r_script(
        "score_annotation_confidence",
        input_rds,
        annotation_column=annotation_column,
        marker_genes=marker_genes
    )

@mcp.tool()
def export_annotations(
    input_rds: str,
    annotation_columns: list = None,
    export_format: str = "csv",
    include_umap: bool = True
):
    """
    Export cell type annotations to various formats

    Exports annotations and metadata for downstream analysis or visualization.

    Args:
        input_rds: Path to input Seurat RDS file with annotations
        annotation_columns: List of annotation columns to export
            If None, exports all annotation-related columns
        export_format: Export format, one of:
            - csv (default, metadata table)
            - h5ad (AnnData format for scanpy)
            - loom (Loom format)
            - tsv (tab-separated)
        include_umap: Include UMAP coordinates in export

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - message: Execution result
        - generated_files: Exported annotation files
        - export_summary: Statistics about exported data
    """
    return run_r_script(
        "export_annotations",
        input_rds,
        annotation_columns=annotation_columns,
        export_format=export_format,
        include_umap=include_umap
    )

# 添加生命周期管理
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

@asynccontextmanager
async def annotation_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """处理服务器启动和关闭"""
    print("Cell Type Annotation MCP Server 正在初始化...")
    
    # Check if reference data exists
    config = load_config()
    from pathlib import Path
    ref_dir = Path(__file__).parent / config["reference_data_dir"]
    if not ref_dir.exists():
        print(f"Warning: Reference data directory not found: {ref_dir}")
        print("Run download_references.R to download SingleR reference datasets")
    
    try:
        yield {"initialized": True}
    finally:
        print("Cell Type Annotation MCP Server 正在关闭...")

# 设置生命周期
mcp.lifespan = annotation_lifespan

if __name__ == "__main__":
    print("启动Cell Type Annotation MCP服务器...")
    # 设置MCP标准路径
    # mcp.settings.sse_path = "/_mcp/v1/sse"
    # mcp.settings.message_path = "/_mcp/v1/messages/"
    # 设置网络参数
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8095
    
    # 使用SSE模式启动
    mcp.run(transport="sse")
