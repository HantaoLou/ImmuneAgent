"""
scRNA-seq Analysis MCP Server

This server provides comprehensive single-cell RNA-seq preprocessing and analysis tools
using Seurat/R via external script execution.

Features:
- Quality control and cell filtering
- Normalization (LogNormalize, SCTransform)
- Integration (Harmony batch correction)
- Clustering analysis (Leiden/Louvain)
- Doublet detection (DoubletFinder)
- Differential expression analysis
- Marker gene detection
- Pathway enrichment analysis
- Dimensionality reduction (PCA, UMAP, tSNE)
- Cell subsetting and filtering

Design Principles:
- NEVER embed R logic in Python
- ALWAYS execute external R scripts via Rscript
- ALWAYS support batch processing
- ALWAYS return structured JSON with file paths
- Uses stdio transport (NOT SSE)
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("scRNA-seq Analysis Server")

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json"""
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {
        "base_dir": str(Path(__file__).parent),
        "output_dir": "output",
        "default_timeout": 3600
    }

def run_r_script(
    script_name: str,
    input_rds: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 3600
) -> Dict[str, Any]:
    """
    Generic R script execution function.

    Args:
        script_name: Name of R script (without .R extension)
        input_rds: Path to input Seurat RDS file
        params: Optional parameters to pass to R script (as JSON)
        timeout: Execution timeout in seconds

    Returns:
        Dict with status, generated_files, and message
    """
    # Validate input file
    if not os.path.exists(input_rds):
        return {
            "status": "error",
            "message": f"Input file does not exist: {input_rds}",
            "generated_files": []
        }

    # Load configuration
    config = load_config()
    working_dir = Path(__file__).parent
    base_dir = Path(config["base_dir"])

    # R script path
    r_script_path = working_dir / "scripts" / f"{script_name}.R"

    # Check if R script exists
    if not r_script_path.exists():
        return {
            "status": "error",
            "message": f"R script does not exist: {r_script_path}",
            "generated_files": []
        }

    # Prepare parameters
    params_json = json.dumps(params) if params else "{}"

    try:
        # Execute R script with input file and parameters
        result = subprocess.run(
            ["Rscript", str(r_script_path), input_rds, params_json],
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
        output_dir = base_dir / config["output_dir"] / script_name
        generated_files = []

        if output_dir.exists():
            # RDS files (processed Seurat objects)
            rds_files = list(output_dir.glob("*.rds"))
            generated_files.extend([str(f) for f in rds_files])

            # CSV files (tables, stats, DEG results)
            csv_files = list(output_dir.glob("*.csv"))
            generated_files.extend([str(f) for f in csv_files])

            # PDF/PNG files (plots)
            plot_files = list(output_dir.glob("*.pdf")) + list(output_dir.glob("*.png"))
            generated_files.extend([str(f) for f in plot_files])

            # TXT files (logs, summaries)
            txt_files = list(output_dir.glob("*.txt"))
            generated_files.extend([str(f) for f in txt_files])

        return {
            "status": "success",
            "message": f"{script_name} analysis completed successfully",
            "output_directory": str(output_dir),
            "generated_files": generated_files,
            "file_count": len(generated_files),
            "stdout": result.stdout
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

# ========== MCP Tool Definitions ==========

@mcp.tool()
def run_qc_filtering(
    input_rds: str,
    min_genes: int = 200,
    max_genes: int = 6000,
    min_counts: int = 1000,
    mt_percent: float = 20.0
) -> Dict[str, Any]:
    """
    Quality control and cell filtering for scRNA-seq data.

    Filters cells based on:
    - Feature counts (nCount_RNA): minimum counts per cell
    - Unique genes (nFeature_RNA): min/max genes per cell
    - Mitochondrial percentage: % of reads from mitochondrial genes

    Args:
        input_rds: Path to input Seurat RDS file
        min_genes: Minimum number of genes per cell (default: 200)
        max_genes: Maximum number of genes per cell (default: 6000)
        min_counts: Minimum UMI counts per cell (default: 1000)
        mt_percent: Maximum mitochondrial percentage (default: 20.0)

    Returns:
        Dict with status, filtered Seurat object path, and QC plots
    """
    params = {
        "min_genes": min_genes,
        "max_genes": max_genes,
        "min_counts": min_counts,
        "mt_percent": mt_percent
    }
    return run_r_script("qc_filtering", input_rds, params)

@mcp.tool()
def run_normalization_sct(
    input_rds: str,
    vars_to_regress: Optional[List[str]] = None,
    n_variable_features: int = 3000
) -> Dict[str, Any]:
    """
    SCTransform normalization for scRNA-seq data.

    Uses Seurat's SCTransform for:
    - Variance stabilization
    - Normalization
    - Feature selection
    - Optional regression of technical covariates

    Args:
        input_rds: Path to input Seurat RDS file (after QC filtering)
        vars_to_regress: Variables to regress out (e.g., ["percent.mt", "nCount_RNA"])
        n_variable_features: Number of variable features to select (default: 3000)

    Returns:
        Dict with status, normalized Seurat object, and diagnostic plots
    """
    params = {
        "vars_to_regress": vars_to_regress or [],
        "n_variable_features": n_variable_features
    }
    return run_r_script("normalization_sct", input_rds, params)

@mcp.tool()
def run_integration_harmony(
    input_rds: str,
    batch_variable: str = "orig.ident",
    dims: int = 30,
    theta: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Harmony batch correction and integration.

    Uses Harmony algorithm for:
    - Batch effect removal
    - Multi-sample integration
    - Preserving biological variation

    Args:
        input_rds: Path to input Seurat RDS file (after normalization)
        batch_variable: Metadata column to use for batch correction (default: "orig.ident")
        dims: Number of PCA dimensions to use (default: 30)
        theta: Diversity clustering penalty parameter (default: None, auto-tuned)

    Returns:
        Dict with status, integrated Seurat object, and integration QC plots
    """
    params = {
        "batch_variable": batch_variable,
        "dims": dims,
        "theta": theta or [2.0]
    }
    return run_r_script("integration_harmony", input_rds, params)

@mcp.tool()
def run_clustering_analysis(
    input_rds: str,
    resolution: float = 0.8,
    dims: int = 30,
    algorithm: str = "leiden"
) -> Dict[str, Any]:
    """
    Graph-based clustering analysis.

    Performs:
    - SNN graph construction
    - Community detection (Leiden or Louvain)
    - UMAP visualization

    Args:
        input_rds: Path to input Seurat RDS file (after integration or normalization)
        resolution: Clustering resolution (higher = more clusters, default: 0.8)
        dims: Number of dimensions to use for clustering (default: 30)
        algorithm: Clustering algorithm ("leiden" or "louvain", default: "leiden")

    Returns:
        Dict with status, clustered Seurat object, and UMAP plots
    """
    params = {
        "resolution": resolution,
        "dims": dims,
        "algorithm": algorithm
    }
    return run_r_script("clustering_analysis", input_rds, params)

@mcp.tool()
def run_doublet_detection(
    input_rds: str,
    expected_doublet_rate: float = 0.08,
    pN: float = 0.25,
    pK: float = 0.09,
    dims: int = 20
) -> Dict[str, Any]:
    """
    Doublet detection using DoubletFinder.

    Identifies and removes doublets (two cells captured together):
    - Simulates artificial doublets
    - Calculates doublet scores
    - Classifies cells as singlet/doublet

    Args:
        input_rds: Path to input Seurat RDS file (after clustering)
        expected_doublet_rate: Expected doublet formation rate (default: 0.08 = 8%)
        pN: Proportion of artificial doublets (default: 0.25)
        pK: PC neighborhood size (default: 0.09)
        dims: Number of PCs to use (default: 20)

    Returns:
        Dict with status, filtered Seurat object (doublets removed), and QC plots
    """
    params = {
        "expected_doublet_rate": expected_doublet_rate,
        "pN": pN,
        "pK": pK,
        "dims": dims
    }
    return run_r_script("doublet_detection", input_rds, params)

@mcp.tool()
def run_deg_analysis(
    input_rds: str,
    group_by: str = "seurat_clusters",
    ident_1: str = None,
    ident_2: str = None,
    test_use: str = "wilcox",
    logfc_threshold: float = 0.25,
    min_pct: float = 0.1
) -> Dict[str, Any]:
    """
    Differential expression gene (DEG) analysis.

    Compares gene expression between groups:
    - Wilcoxon rank-sum test (default)
    - MAST (hurdle model for scRNA-seq)
    - DESeq2 (for pseudobulk)

    Args:
        input_rds: Path to input Seurat RDS file (after clustering)
        group_by: Metadata column for grouping (default: "seurat_clusters")
        ident_1: First identity to compare (e.g., "cluster_0")
        ident_2: Second identity to compare (e.g., "cluster_1", None = all others)
        test_use: Statistical test ("wilcox", "MAST", "DESeq2", default: "wilcox")
        logfc_threshold: Minimum log fold-change threshold (default: 0.25)
        min_pct: Minimum percentage of cells expressing gene (default: 0.1)

    Returns:
        Dict with status, DEG tables (CSV), volcano plots, and heatmaps
    """
    params = {
        "group_by": group_by,
        "ident_1": ident_1,
        "ident_2": ident_2,
        "test_use": test_use,
        "logfc_threshold": logfc_threshold,
        "min_pct": min_pct
    }
    return run_r_script("deg_analysis", input_rds, params)

@mcp.tool()
def run_marker_detection(
    input_rds: str,
    group_by: str = "seurat_clusters",
    only_pos: bool = True,
    min_pct: float = 0.25,
    logfc_threshold: float = 0.5,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Marker gene detection for all clusters/groups.

    Uses FindAllMarkers to identify:
    - Cluster-specific marker genes
    - Top N markers per cluster
    - Specificity metrics (pct.1 vs pct.2)

    Args:
        input_rds: Path to input Seurat RDS file (after clustering)
        group_by: Metadata column for grouping (default: "seurat_clusters")
        only_pos: Only return positive markers (default: True)
        min_pct: Minimum percentage of cells expressing gene (default: 0.25)
        logfc_threshold: Minimum log fold-change threshold (default: 0.5)
        top_n: Number of top markers per cluster (default: 10)

    Returns:
        Dict with status, marker tables (CSV), dot plots, and heatmaps
    """
    params = {
        "group_by": group_by,
        "only_pos": only_pos,
        "min_pct": min_pct,
        "logfc_threshold": logfc_threshold,
        "top_n": top_n
    }
    return run_r_script("marker_detection", input_rds, params)

@mcp.tool()
def run_pathway_enrichment(
    input_rds: str,
    deg_csv: str,
    organism: str = "human",
    ontology: str = "BP",
    pvalue_cutoff: float = 0.05,
    qvalue_cutoff: float = 0.2
) -> Dict[str, Any]:
    """
    Gene set enrichment and pathway analysis.

    Uses clusterProfiler for:
    - GO enrichment (Biological Process, Molecular Function, Cellular Component)
    - KEGG pathway analysis
    - GSEA (Gene Set Enrichment Analysis)

    Args:
        input_rds: Path to input Seurat RDS file
        deg_csv: Path to DEG results CSV (from run_deg_analysis)
        organism: Species ("human" or "mouse", default: "human")
        ontology: GO ontology ("BP", "MF", "CC", default: "BP")
        pvalue_cutoff: P-value cutoff for enrichment (default: 0.05)
        qvalue_cutoff: Q-value cutoff for FDR correction (default: 0.2)

    Returns:
        Dict with status, enrichment tables (CSV), and dot plots
    """
    params = {
        "deg_csv": deg_csv,
        "organism": organism,
        "ontology": ontology,
        "pvalue_cutoff": pvalue_cutoff,
        "qvalue_cutoff": qvalue_cutoff
    }
    return run_r_script("pathway_enrichment", input_rds, params)

@mcp.tool()
def run_dim_reduction(
    input_rds: str,
    methods: Optional[List[str]] = None,
    dims: int = 30,
    n_neighbors: int = 30,
    min_dist: float = 0.3
) -> Dict[str, Any]:
    """
    Dimensionality reduction and visualization.

    Generates multiple embeddings:
    - PCA (Principal Component Analysis)
    - UMAP (Uniform Manifold Approximation and Projection)
    - tSNE (t-Distributed Stochastic Neighbor Embedding)

    Args:
        input_rds: Path to input Seurat RDS file (after normalization)
        methods: List of methods to run (default: ["PCA", "UMAP", "tSNE"])
        dims: Number of dimensions to compute (default: 30)
        n_neighbors: UMAP parameter for local neighborhood size (default: 30)
        min_dist: UMAP parameter for minimum distance (default: 0.3)

    Returns:
        Dict with status, Seurat object with embeddings, and visualization plots
    """
    params = {
        "methods": methods or ["PCA", "UMAP", "tSNE"],
        "dims": dims,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist
    }
    return run_r_script("dim_reduction", input_rds, params)

@mcp.tool()
def run_subset_cells(
    input_rds: str,
    subset_column: str,
    subset_values: List[str],
    invert: bool = False
) -> Dict[str, Any]:
    """
    Subset cells based on metadata criteria.

    Filters Seurat object to:
    - Keep specific cell types/clusters
    - Remove unwanted populations
    - Create focused subsets for downstream analysis

    Args:
        input_rds: Path to input Seurat RDS file
        subset_column: Metadata column to filter on (e.g., "seurat_clusters", "celltype")
        subset_values: Values to keep (e.g., ["0", "1", "3"] or ["B cell", "T cell"])
        invert: If True, remove specified values instead of keeping them (default: False)

    Returns:
        Dict with status, subsetted Seurat object, and summary statistics
    """
    params = {
        "subset_column": subset_column,
        "subset_values": subset_values,
        "invert": invert
    }
    return run_r_script("subset_cells", input_rds, params)

@mcp.tool()
def run_full_preprocessing_pipeline(
    input_rds: str,
    min_genes: int = 200,
    max_genes: int = 10000,
    min_counts: int = 1000,
    mt_percent: float = 25.0,
    n_variable_features: int = 3000,
    vars_to_regress: Optional[List[str]] = None,
    resolution: float = 0.8,
    dims: int = 30
) -> Dict[str, Any]:
    """
    Run complete preprocessing pipeline: QC → Normalize → Dim Reduce → Cluster

    This is a convenience tool that chains multiple preprocessing steps into a single
    end-to-end workflow. Perfect for preparing raw Seurat objects for downstream analysis.

    Pipeline Steps:
    1. Quality Control: Filter low-quality cells based on gene counts and MT%
    2. Normalization: SCTransform normalization with optional regression
    3. Dimensionality Reduction: PCA and UMAP computation
    4. Clustering: Graph-based clustering with specified resolution

    Args:
        input_rds: Path to raw Seurat RDS file
        min_genes: Minimum genes per cell (QC, default: 200)
        max_genes: Maximum genes per cell (QC, default: 10000)
        min_counts: Minimum UMI counts per cell (QC, default: 1000)
        mt_percent: Maximum mitochondrial percentage (QC, default: 25.0)
        n_variable_features: Number of variable features for SCTransform (default: 3000)
        vars_to_regress: Variables to regress out during normalization (default: None)
        resolution: Clustering resolution (higher = more clusters, default: 0.8)
        dims: Number of PCA dimensions for clustering (default: 30)

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - pipeline_steps: List of completed steps
        - final_output: Path to final preprocessed RDS file
        - intermediate_files: Dict mapping step names to output files
        - message: Summary message
    """
    try:
        results = {
            "status": "success",
            "pipeline_steps": [],
            "final_output": None,
            "intermediate_files": {},
            "errors": []
        }

        current_rds = input_rds

        # Step 1: QC Filtering
        print("Step 1/4: Running QC filtering...")
        qc_result = run_qc_filtering(current_rds, min_genes, max_genes, min_counts, mt_percent)
        if qc_result["status"] == "error":
            results["status"] = "error"
            results["message"] = f"Pipeline failed at QC filtering: {qc_result.get('message', 'Unknown error')}"
            results["errors"].append({"step": "qc_filtering", "error": qc_result.get("message")})
            return results

        results["pipeline_steps"].append("qc_filtering")
        results["intermediate_files"]["qc"] = qc_result.get("generated_files", [])

        # Get the processed RDS from QC step
        rds_files = [f for f in qc_result.get("generated_files", []) if f.endswith('.rds')]
        if not rds_files:
            results["status"] = "error"
            results["message"] = "QC filtering did not produce an RDS file"
            return results
        current_rds = rds_files[0]

        # Step 2: Normalization (SCTransform)
        print("Step 2/4: Running SCTransform normalization...")
        norm_result = run_normalization_sct(current_rds, vars_to_regress or [], n_variable_features)
        if norm_result["status"] == "error":
            results["status"] = "error"
            results["message"] = f"Pipeline failed at normalization: {norm_result.get('message', 'Unknown error')}"
            results["errors"].append({"step": "normalization", "error": norm_result.get("message")})
            return results

        results["pipeline_steps"].append("normalization_sct")
        results["intermediate_files"]["normalize"] = norm_result.get("generated_files", [])

        # Get the normalized RDS
        rds_files = [f for f in norm_result.get("generated_files", []) if f.endswith('.rds')]
        if not rds_files:
            results["status"] = "error"
            results["message"] = "Normalization did not produce an RDS file"
            return results
        current_rds = rds_files[0]

        # Step 3: Dimensionality Reduction
        print("Step 3/4: Running dimensionality reduction...")
        dim_result = run_dim_reduction(current_rds, ["PCA", "UMAP"], dims, 30, 0.3)
        if dim_result["status"] == "error":
            results["status"] = "error"
            results["message"] = f"Pipeline failed at dim reduction: {dim_result.get('message', 'Unknown error')}"
            results["errors"].append({"step": "dim_reduction", "error": dim_result.get("message")})
            return results

        results["pipeline_steps"].append("dim_reduction")
        results["intermediate_files"]["dimred"] = dim_result.get("generated_files", [])

        # Get the reduced RDS
        rds_files = [f for f in dim_result.get("generated_files", []) if f.endswith('.rds')]
        if not rds_files:
            results["status"] = "error"
            results["message"] = "Dim reduction did not produce an RDS file"
            return results
        current_rds = rds_files[0]

        # Step 4: Clustering
        print("Step 4/4: Running clustering analysis...")
        cluster_result = run_clustering_analysis(current_rds, resolution, dims, "leiden")
        if cluster_result["status"] == "error":
            results["status"] = "error"
            results["message"] = f"Pipeline failed at clustering: {cluster_result.get('message', 'Unknown error')}"
            results["errors"].append({"step": "clustering", "error": cluster_result.get("message")})
            return results

        results["pipeline_steps"].append("clustering")
        results["intermediate_files"]["cluster"] = cluster_result.get("generated_files", [])

        # Get final clustered RDS
        rds_files = [f for f in cluster_result.get("generated_files", []) if f.endswith('.rds')]
        if rds_files:
            results["final_output"] = rds_files[0]

        results["message"] = f"Full preprocessing pipeline completed successfully: {' → '.join(results['pipeline_steps'])}"

        return results

    except Exception as e:
        return {
            "status": "error",
            "message": f"Pipeline failed with exception: {str(e)}",
            "pipeline_steps": results.get("pipeline_steps", []),
            "final_output": None,
            "intermediate_files": results.get("intermediate_files", {}),
            "errors": results.get("errors", []) + [{"step": "exception", "error": str(e)}]
        }

# ========== Server Lifecycle Management ==========

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

@asynccontextmanager
async def scrna_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Handle server startup and shutdown"""
    print("scRNA-seq Analysis MCP Server initializing...")

    # Ensure output directory exists
    config = load_config()
    output_dir = Path(config["base_dir"]) / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print("Server ready on stdio transport")

    try:
        yield {"initialized": True}
    finally:
        print("scRNA-seq Analysis MCP Server shutting down...")

# Set lifecycle
mcp.lifespan = scrna_lifespan

if __name__ == "__main__":
    print("Starting scRNA-seq Analysis MCP Server...")
    print("Transport: stdio (NOT SSE)")

    # Use stdio transport (consistent with sessions 3-5)
    mcp.run(transport="stdio")
