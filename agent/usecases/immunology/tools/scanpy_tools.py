"""
Scanpy Tool Wrapper for Single-Cell Analysis
Real implementation using scanpy library
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from langchain_core.tools import tool


@tool
def load_single_cell_data(file_path: str, file_format: str = "h5ad") -> Dict[str, Any]:
    """
    Load single-cell RNA-seq data from various formats

    Args:
        file_path: Path to the data file
        file_format: Format of the file (h5ad, csv, mtx, loom)

    Returns:
        Dictionary with data summary and path to processed h5ad file
    """
    try:
        # Load data based on format
        if file_format == "h5ad":
            adata = sc.read_h5ad(file_path)
        elif file_format == "csv":
            adata = sc.read_csv(file_path)
        elif file_format == "mtx":
            adata = sc.read_mtx(file_path)
        elif file_format == "loom":
            adata = sc.read_loom(file_path)
        else:
            return {"error": f"Unsupported format: {file_format}"}

        # Basic info
        n_obs, n_vars = adata.shape

        # Save processed data
        output_path = file_path.replace(f".{file_format}", "_processed.h5ad")
        adata.write_h5ad(output_path)

        return {
            "success": True,
            "n_cells": n_obs,
            "n_genes": n_vars,
            "output_path": output_path,
            "obs_columns": list(adata.obs.columns),
            "var_columns": list(adata.var.columns),
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def quality_control_filtering(
    adata_path: str,
    min_genes: int = 200,
    min_cells: int = 3,
    max_mt_percent: float = 5.0,
    max_genes: int = 2500,
) -> Dict[str, Any]:
    """
    Perform quality control and filtering on single-cell data

    Args:
        adata_path: Path to AnnData object (h5ad file)
        min_genes: Minimum number of genes per cell
        min_cells: Minimum number of cells per gene
        max_mt_percent: Maximum mitochondrial gene percentage
        max_genes: Maximum number of genes per cell

    Returns:
        QC metrics and filtered data path
    """
    try:
        # Load data
        adata = sc.read_h5ad(adata_path)
        initial_cells = adata.n_obs
        initial_genes = adata.n_vars

        # Calculate QC metrics
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
        )

        # Filter cells and genes
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)

        # Filter based on QC metrics
        adata = adata[adata.obs.n_genes_by_counts < max_genes, :]
        adata = adata[adata.obs.pct_counts_mt < max_mt_percent, :]

        # Save filtered data
        output_path = adata_path.replace(".h5ad", "_qc_filtered.h5ad")
        adata.write_h5ad(output_path)

        return {
            "success": True,
            "initial_cells": initial_cells,
            "initial_genes": initial_genes,
            "filtered_cells": adata.n_obs,
            "filtered_genes": adata.n_vars,
            "cells_removed": initial_cells - adata.n_obs,
            "genes_removed": initial_genes - adata.n_vars,
            "output_path": output_path,
            "mean_genes_per_cell": float(np.mean(adata.obs.n_genes_by_counts)),
            "mean_counts_per_cell": float(np.mean(adata.obs.total_counts)),
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def normalize_and_scale(
    adata_path: str,
    target_sum: float = 1e4,
    n_highly_variable: int = 2000,
    max_value: float = 10,
) -> Dict[str, Any]:
    """
    Normalize and scale single-cell data

    Args:
        adata_path: Path to filtered AnnData object
        target_sum: Target sum for normalization
        n_highly_variable: Number of highly variable genes to keep
        max_value: Maximum value for scaling

    Returns:
        Normalized data path and statistics
    """
    try:
        # Load data
        adata = sc.read_h5ad(adata_path)

        # Store raw counts
        adata.raw = adata

        # Normalization
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)

        # Find highly variable genes
        sc.pp.highly_variable_genes(adata, n_top_genes=n_highly_variable, subset=True)

        # Scale data
        sc.pp.scale(adata, max_value=max_value)

        # Save normalized data
        output_path = adata_path.replace(".h5ad", "_normalized.h5ad")
        adata.write_h5ad(output_path)

        return {
            "success": True,
            "n_highly_variable_genes": n_highly_variable,
            "total_genes_after_hvg": adata.n_vars,
            "output_path": output_path,
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def perform_pca(
    adata_path: str, n_comps: int = 50, svd_solver: str = "arpack"
) -> Dict[str, Any]:
    """
    Perform Principal Component Analysis

    Args:
        adata_path: Path to normalized AnnData object
        n_comps: Number of principal components
        svd_solver: SVD solver to use

    Returns:
        PCA results and variance explained
    """
    try:
        # Load data
        adata = sc.read_h5ad(adata_path)

        # Run PCA
        sc.tl.pca(adata, n_comps=n_comps, svd_solver=svd_solver)

        # Calculate variance ratio
        variance_ratio = adata.uns["pca"]["variance_ratio"]
        cumulative_variance = np.cumsum(variance_ratio)

        # Save with PCA
        output_path = adata_path.replace(".h5ad", "_pca.h5ad")
        adata.write_h5ad(output_path)

        return {
            "success": True,
            "n_components": n_comps,
            "variance_explained_pc1": float(variance_ratio[0]),
            "variance_explained_pc2": float(variance_ratio[1]),
            "cumulative_variance_10pc": float(cumulative_variance[9])
            if len(cumulative_variance) > 9
            else float(cumulative_variance[-1]),
            "output_path": output_path,
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def compute_neighbors_and_umap(
    adata_path: str, n_neighbors: int = 15, n_pcs: int = 40, min_dist: float = 0.3
) -> Dict[str, Any]:
    """
    Compute neighbor graph and UMAP embedding

    Args:
        adata_path: Path to PCA-processed AnnData object
        n_neighbors: Number of neighbors for graph
        n_pcs: Number of PCs to use
        min_dist: Minimum distance for UMAP

    Returns:
        UMAP coordinates and graph statistics
    """
    try:
        # Load data
        adata = sc.read_h5ad(adata_path)

        # Compute neighbor graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

        # Compute UMAP
        sc.tl.umap(adata, min_dist=min_dist)

        # Save with UMAP
        output_path = adata_path.replace(".h5ad", "_umap.h5ad")
        adata.write_h5ad(output_path)

        # Get UMAP coordinates range
        umap_coords = adata.obsm["X_umap"]

        return {
            "success": True,
            "n_neighbors": n_neighbors,
            "n_pcs_used": n_pcs,
            "umap_x_range": [
                float(umap_coords[:, 0].min()),
                float(umap_coords[:, 0].max()),
            ],
            "umap_y_range": [
                float(umap_coords[:, 1].min()),
                float(umap_coords[:, 1].max()),
            ],
            "output_path": output_path,
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def leiden_clustering(
    adata_path: str, resolution: float = 1.0, random_state: int = 0
) -> Dict[str, Any]:
    """
    Perform Leiden clustering

    Args:
        adata_path: Path to UMAP-processed AnnData object
        resolution: Resolution parameter for clustering
        random_state: Random seed for reproducibility

    Returns:
        Clustering results with cluster statistics
    """
    try:
        # Load data
        adata = sc.read_h5ad(adata_path)

        # Run Leiden clustering
        sc.tl.leiden(adata, resolution=resolution, random_state=random_state)

        # Get cluster statistics
        cluster_counts = adata.obs["leiden"].value_counts().to_dict()
        n_clusters = len(cluster_counts)

        # Save with clusters
        output_path = adata_path.replace(".h5ad", "_clustered.h5ad")
        adata.write_h5ad(output_path)

        return {
            "success": True,
            "n_clusters": n_clusters,
            "resolution": resolution,
            "cluster_sizes": {str(k): int(v) for k, v in cluster_counts.items()},
            "smallest_cluster_size": int(min(cluster_counts.values())),
            "largest_cluster_size": int(max(cluster_counts.values())),
            "output_path": output_path,
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def find_marker_genes(
    adata_path: str,
    groupby: str = "leiden",
    method: str = "wilcoxon",
    n_genes: int = 25,
) -> Dict[str, Any]:
    """
    Find marker genes for each cluster

    Args:
        adata_path: Path to clustered AnnData object
        groupby: Column to group by (usually 'leiden')
        method: Statistical test method
        n_genes: Number of top genes per group

    Returns:
        Marker genes for each cluster
    """
    try:
        # Load data
        adata = sc.read_h5ad(adata_path)

        # Find marker genes
        sc.tl.rank_genes_groups(adata, groupby=groupby, method=method, n_genes=n_genes)

        # Extract top markers
        markers = {}
        result = adata.uns["rank_genes_groups"]
        groups = result["names"].dtype.names

        for group in groups:
            markers[str(group)] = {
                "genes": list(result["names"][group][:10]),
                "scores": [float(x) for x in result["scores"][group][:10]],
                "pvals": [float(x) for x in result["pvals"][group][:10]],
            }

        # Save with markers
        output_path = adata_path.replace(".h5ad", "_markers.h5ad")
        adata.write_h5ad(output_path)

        # Save marker genes as CSV
        marker_df = pd.DataFrame(result["names"])
        csv_path = output_path.replace(".h5ad", "_markers.csv")
        marker_df.to_csv(csv_path)

        return {
            "success": True,
            "n_clusters": len(groups),
            "method": method,
            "top_markers": markers,
            "output_path": output_path,
            "markers_csv": csv_path,
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def annotate_cell_types(
    adata_path: str,
    marker_genes: Dict[str, List[str]],
    annotation_key: str = "cell_type",
) -> Dict[str, Any]:
    """
    Annotate cell types based on marker genes

    Args:
        adata_path: Path to clustered AnnData object
        marker_genes: Dictionary mapping cell types to marker genes
        annotation_key: Key to store annotations

    Returns:
        Annotated data with cell type assignments
    """
    try:
        # Load data
        adata = sc.read_h5ad(adata_path)

        # Score each cell type
        for cell_type, genes in marker_genes.items():
            # Filter genes that exist in the data
            genes_present = [g for g in genes if g in adata.var_names]
            if genes_present:
                sc.tl.score_genes(adata, genes_present, score_name=f"{cell_type}_score")

        # Assign cell types based on highest score
        score_cols = [col for col in adata.obs.columns if col.endswith("_score")]
        if score_cols:
            scores = adata.obs[score_cols]
            adata.obs[annotation_key] = scores.idxmax(axis=1).str.replace("_score", "")
        else:
            # Fallback to cluster-based annotation
            adata.obs[annotation_key] = "Unknown"

        # Get cell type statistics
        cell_type_counts = adata.obs[annotation_key].value_counts().to_dict()

        # Save annotated data
        output_path = adata_path.replace(".h5ad", "_annotated.h5ad")
        adata.write_h5ad(output_path)

        return {
            "success": True,
            "cell_types": {str(k): int(v) for k, v in cell_type_counts.items()},
            "n_cell_types": len(cell_type_counts),
            "output_path": output_path,
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def differential_expression(
    adata_path: str,
    group1: str,
    group2: str,
    groupby: str = "cell_type",
    method: str = "wilcoxon",
) -> Dict[str, Any]:
    """
    Perform differential expression between two groups

    Args:
        adata_path: Path to annotated AnnData object
        group1: First group for comparison
        group2: Second group for comparison
        groupby: Column containing group labels
        method: Statistical test method

    Returns:
        Differentially expressed genes between groups
    """
    try:
        # Load data
        adata = sc.read_h5ad(adata_path)

        # Subset to groups of interest
        mask = adata.obs[groupby].isin([group1, group2])
        adata_subset = adata[mask].copy()

        # Run differential expression
        sc.tl.rank_genes_groups(
            adata_subset,
            groupby=groupby,
            groups=[group1],
            reference=group2,
            method=method,
        )

        # Extract results
        result = adata_subset.uns["rank_genes_groups"]
        de_genes = pd.DataFrame(
            {
                "gene": result["names"][group1],
                "score": result["scores"][group1],
                "pval": result["pvals"][group1],
                "pval_adj": result["pvals_adj"][group1],
                "logfoldchange": result["logfoldchanges"][group1],
            }
        )

        # Filter significant genes
        sig_genes = de_genes[de_genes["pval_adj"] < 0.05]

        # Save DE results
        output_path = adata_path.replace(".h5ad", f"_DE_{group1}_vs_{group2}.csv")
        de_genes.to_csv(output_path, index=False)

        return {
            "success": True,
            "comparison": f"{group1} vs {group2}",
            "total_genes_tested": len(de_genes),
            "significant_genes": len(sig_genes),
            "top_upregulated": sig_genes.nlargest(10, "logfoldchange")["gene"].tolist(),
            "top_downregulated": sig_genes.nsmallest(10, "logfoldchange")[
                "gene"
            ].tolist(),
            "output_path": output_path,
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def trajectory_analysis(
    adata_path: str, root_cluster: Optional[str] = None, n_dcs: int = 15
) -> Dict[str, Any]:
    """
    Perform trajectory inference using diffusion maps and PAGA

    Args:
        adata_path: Path to clustered AnnData object
        root_cluster: Starting cluster for trajectory
        n_dcs: Number of diffusion components

    Returns:
        Trajectory analysis results with pseudotime
    """
    try:
        # Load data
        adata = sc.read_h5ad(adata_path)

        # Compute diffusion maps
        sc.tl.diffmap(adata, n_comps=n_dcs)

        # Run PAGA
        sc.tl.paga(adata, groups="leiden")

        # Compute pseudotime if root is specified
        if root_cluster is not None:
            # Set root cell
            root_idx = adata.obs["leiden"] == str(root_cluster)
            if root_idx.any():
                iroot = np.where(root_idx)[0][0]
                adata.uns["iroot"] = iroot

                # Compute diffusion pseudotime
                sc.tl.dpt(adata)

                # Get pseudotime statistics
                pseudotime_stats = {
                    "min": float(adata.obs["dpt_pseudotime"].min()),
                    "max": float(adata.obs["dpt_pseudotime"].max()),
                    "mean": float(adata.obs["dpt_pseudotime"].mean()),
                }
            else:
                pseudotime_stats = {"error": "Root cluster not found"}
        else:
            pseudotime_stats = {"note": "No root specified, pseudotime not computed"}

        # Save with trajectory
        output_path = adata_path.replace(".h5ad", "_trajectory.h5ad")
        adata.write_h5ad(output_path)

        return {
            "success": True,
            "n_diffusion_components": n_dcs,
            "paga_connectivity_computed": True,
            "pseudotime": pseudotime_stats,
            "output_path": output_path,
        }

    except Exception as e:
        return {"error": str(e)}


# Tool collections
scanpy_basic_tools = [
    load_single_cell_data,
    quality_control_filtering,
    normalize_and_scale,
]

scanpy_analysis_tools = [
    perform_pca,
    compute_neighbors_and_umap,
    leiden_clustering,
    find_marker_genes,
]

scanpy_advanced_tools = [
    annotate_cell_types,
    differential_expression,
    trajectory_analysis,
]

all_scanpy_tools = scanpy_basic_tools + scanpy_analysis_tools + scanpy_advanced_tools

# Create tool dictionary for easy access
scanpy_tools_dict = {
    "load_single_cell_data": load_single_cell_data,
    "quality_control_filtering": quality_control_filtering,
    "normalize_and_scale": normalize_and_scale,
    "perform_pca": perform_pca,
    "compute_neighbors_and_umap": compute_neighbors_and_umap,
    "leiden_clustering": leiden_clustering,
    "find_marker_genes": find_marker_genes,
    "annotate_cell_types": annotate_cell_types,
    "differential_expression": differential_expression,
    "trajectory_analysis": trajectory_analysis,
}

# Export
__all__ = [
    "load_single_cell_data",
    "quality_control_filtering",
    "normalize_and_scale",
    "perform_pca",
    "compute_neighbors_and_umap",
    "leiden_clustering",
    "find_marker_genes",
    "annotate_cell_types",
    "differential_expression",
    "trajectory_analysis",
    "all_scanpy_tools",
    "scanpy_tools_dict",
]
