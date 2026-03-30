"""
08_tcr_analysis.py — T cell receptor (TCR) analysis.
=====================================================
CDR3 length distribution, clonotype expansion, CD4/CD8 ratio, and T cell
exhaustion marker scoring.

Prerequisites:
    adata loaded + QC + normalized + clustered + cell-type annotated (01-07)
    TCR metadata columns must exist in adata.obs.

Input contract:
    --input   Path to annotated h5ad file
    --output  Directory for results
    --params  JSON: {"cell_type_col": "cell_type",
               "t_cell_labels": ["T cell", "CD4 T", "CD8 T"],
               "cdr3_alpha_col": "cdr3_aa_alpha", "cdr3_beta_col": "cdr3_aa_beta",
               "clone_id_col": "clone_id", "top_n_clones": 20}

Output contract:
    <output_dir>/tcr_summary.json          — CDR3 stats, CD4/CD8, exhaustion scores
    <output_dir>/clonotype_expansion.json   — Top expanded TCR clonotypes
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import scanpy as sc


### ADAPT ### — column names for TCR metadata; varies per dataset
DEFAULT_PARAMS = {
    "cell_type_col": "cell_type",
    "t_cell_labels": ["T cell", "T cells", "CD4 T", "CD4 T cell", "CD8 T", "CD8 T cell"],
    "cdr3_alpha_col": "cdr3_aa_alpha",
    "cdr3_beta_col": "cdr3_aa_beta",
    "clone_id_col": "clone_id",
    "top_n_clones": 20,
}

### ADAPT ### — exhaustion markers; human gene symbols shown, adjust for mouse
EXHAUSTION_MARKERS = ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX"]


def _subset_t_cells(adata: sc.AnnData, params: dict) -> sc.AnnData:
    col = params["cell_type_col"]
    labels = params["t_cell_labels"]
    if col not in adata.obs.columns:
        print(f"WARNING: '{col}' not in obs. Using all cells.")
        return adata
    mask = adata.obs[col].isin(labels)
    subset = adata[mask].copy()
    print(f"T cell subset: {subset.n_obs} / {adata.n_obs} cells")
    return subset


def _cdr3_length_distribution(obs: pd.DataFrame, col: str) -> dict:
    """CDR3 amino acid length distribution for one chain."""
    if col not in obs.columns:
        return {"error": f"Column '{col}' not found"}
    lengths = obs[col].dropna().astype(str).str.len()
    if len(lengths) == 0:
        return {"n_cells_with_cdr3": 0}
    return {
        "n_cells_with_cdr3": int(len(lengths)),
        "mean_length": round(float(lengths.mean()), 2),
        "median_length": int(lengths.median()),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "length_histogram": {
            str(k): int(v) for k, v in sorted(Counter(lengths).items())
        },
    }


def _clonotype_expansion(obs: pd.DataFrame, clone_col: str, top_n: int) -> dict:
    if clone_col not in obs.columns:
        return {"error": f"Column '{clone_col}' not found"}

    clone_counts = obs[clone_col].dropna().value_counts()
    n_unique = len(clone_counts)
    n_expanded = int((clone_counts > 1).sum())

    size_bins = {"singleton_1": 0, "small_2_5": 0, "medium_6_20": 0,
                 "large_21_100": 0, "hyperexpanded_100plus": 0}
    for count in clone_counts.values:
        if count == 1:
            size_bins["singleton_1"] += 1
        elif count <= 5:
            size_bins["small_2_5"] += 1
        elif count <= 20:
            size_bins["medium_6_20"] += 1
        elif count <= 100:
            size_bins["large_21_100"] += 1
        else:
            size_bins["hyperexpanded_100plus"] += 1

    top_clones = [
        {"clone_id": str(cid), "size": int(cnt)}
        for cid, cnt in clone_counts.head(top_n).items()
    ]

    return {
        "n_unique_clonotypes": int(n_unique),
        "n_expanded": n_expanded,
        "n_singleton": int((clone_counts == 1).sum()),
        "clonal_expansion_index": round(n_expanded / n_unique, 4) if n_unique else 0,
        "size_distribution": size_bins,
        "top_clones": top_clones,
    }


def _cd4_cd8_ratio(adata: sc.AnnData, cell_type_col: str) -> dict:
    """Estimate CD4/CD8 ratio from cell type annotations or marker expression."""
    if cell_type_col in adata.obs.columns:
        labels = adata.obs[cell_type_col].astype(str).str.lower()
        n_cd4 = int(labels.str.contains("cd4").sum())
        n_cd8 = int(labels.str.contains("cd8").sum())
        if n_cd4 > 0 or n_cd8 > 0:
            return {
                "n_cd4": n_cd4,
                "n_cd8": n_cd8,
                "ratio": round(n_cd4 / n_cd8, 4) if n_cd8 > 0 else float("inf"),
                "source": "cell_type_annotation",
            }

    cd4_genes = [g for g in ["CD4"] if g in adata.var_names]
    cd8_genes = [g for g in ["CD8A", "CD8B"] if g in adata.var_names]
    if cd4_genes and cd8_genes:
        X = adata[:, cd4_genes + cd8_genes].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        cd4_expr = X[:, :len(cd4_genes)].mean(axis=1)
        cd8_expr = X[:, len(cd4_genes):].mean(axis=1)
        n_cd4 = int((cd4_expr > cd8_expr).sum())
        n_cd8 = int((cd8_expr > cd4_expr).sum())
        return {
            "n_cd4": n_cd4,
            "n_cd8": n_cd8,
            "ratio": round(n_cd4 / n_cd8, 4) if n_cd8 > 0 else float("inf"),
            "source": "marker_expression",
        }

    return {"error": "Cannot determine CD4/CD8 ratio"}


### ADAPT ### — exhaustion gene list; adjust for species
def _exhaustion_score(adata: sc.AnnData) -> dict:
    """Score exhaustion markers per cell, return summary statistics."""
    available = [g for g in EXHAUSTION_MARKERS if g in adata.var_names]
    if not available:
        return {"available_markers": [], "error": "No exhaustion markers found in var_names"}

    sc.tl.score_genes(adata, gene_list=available, score_name="exhaustion_score")
    scores = adata.obs["exhaustion_score"]

    per_marker = {}
    for gene in available:
        col_idx = list(adata.var_names).index(gene)
        vals = adata.X[:, col_idx]
        if hasattr(vals, "toarray"):
            vals = vals.toarray().flatten()
        else:
            vals = np.asarray(vals).flatten()
        per_marker[gene] = {
            "mean_expression": round(float(np.mean(vals)), 4),
            "pct_expressing": round(float((vals > 0).mean()) * 100, 2),
        }

    return {
        "available_markers": available,
        "composite_score": {
            "mean": round(float(scores.mean()), 4),
            "median": round(float(scores.median()), 4),
            "std": round(float(scores.std()), 4),
        },
        "per_marker": per_marker,
    }


def main(input_path: str, output_dir: str, params: dict = None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    os.makedirs(output_dir, exist_ok=True)

    adata = sc.read_h5ad(input_path)
    t_adata = _subset_t_cells(adata, p)

    cdr3_alpha = _cdr3_length_distribution(t_adata.obs, p["cdr3_alpha_col"])
    cdr3_beta = _cdr3_length_distribution(t_adata.obs, p["cdr3_beta_col"])
    clone_exp = _clonotype_expansion(t_adata.obs, p["clone_id_col"], p["top_n_clones"])
    cd4_cd8 = _cd4_cd8_ratio(t_adata, p["cell_type_col"])
    exhaustion = _exhaustion_score(t_adata)

    summary = {
        "status": "success",
        "n_t_cells": int(t_adata.n_obs),
        "cdr3_alpha": cdr3_alpha,
        "cdr3_beta": cdr3_beta,
        "cd4_cd8_ratio": cd4_cd8,
        "exhaustion": exhaustion,
    }

    with open(os.path.join(output_dir, "tcr_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "clonotype_expansion.json"), "w") as f:
        json.dump(clone_exp, f, indent=2)

    print(f"TCR analysis complete: {t_adata.n_obs} T cells")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="08_tcr_analysis: TCR repertoire analysis")
    parser.add_argument("--input", required=True, help="Path to annotated h5ad")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--params", default="{}", help="JSON string of parameters")
    args = parser.parse_args()

    result = main(args.input, args.output, json.loads(args.params))
    print(json.dumps(result, indent=2))
