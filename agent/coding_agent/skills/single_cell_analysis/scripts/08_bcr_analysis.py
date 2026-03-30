"""
08_bcr_analysis.py — B cell receptor (BCR) analysis.
=====================================================
Isotype distribution, clonotype expansion, V(D)J gene usage, and optional
somatic hypermutation summary for B cell populations.

Prerequisites:
    adata loaded + QC + normalized + clustered + cell-type annotated (01-07)
    BCR metadata columns must exist in adata.obs (from 14_data_integration
    or pre-merged in the original dataset).

Input contract:
    --input   Path to annotated h5ad file
    --output  Directory for results
    --params  JSON: {"cell_type_col": "cell_type", "b_cell_labels": ["B cell"],
               "isotype_col": "isotype", "clone_id_col": "clone_id",
               "v_gene_col": "v_call", "j_gene_col": "j_call",
               "shm_col": null, "top_n_clones": 20}

Output contract:
    <output_dir>/bcr_summary.json         — Isotype counts, V/J usage, SHM stats
    <output_dir>/clonotype_expansion.json  — Top expanded clonotypes
    <output_dir>/bcr_vgene_usage.csv       — V gene counts table
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import scanpy as sc


### ADAPT ### — column names for BCR metadata; varies per dataset
DEFAULT_PARAMS = {
    "cell_type_col": "cell_type",
    "b_cell_labels": ["B cell", "B cells", "Plasma cell", "Plasma cells"],
    "isotype_col": "isotype",
    "clone_id_col": "clone_id",
    "v_gene_col": "v_call",
    "j_gene_col": "j_call",
    "shm_col": None,
    "top_n_clones": 20,
}


def _subset_b_cells(adata: sc.AnnData, params: dict) -> sc.AnnData:
    """Subset to B-lineage cells only."""
    col = params["cell_type_col"]
    labels = params["b_cell_labels"]
    if col not in adata.obs.columns:
        print(f"WARNING: '{col}' not in obs. Using all cells.")
        return adata
    mask = adata.obs[col].isin(labels)
    subset = adata[mask].copy()
    print(f"B cell subset: {subset.n_obs} / {adata.n_obs} cells")
    return subset


def _isotype_distribution(obs: pd.DataFrame, col: str) -> dict:
    """Count cells per isotype."""
    if col not in obs.columns:
        return {"error": f"Column '{col}' not found in obs"}
    counts = obs[col].value_counts().to_dict()
    total = sum(counts.values())
    distribution = {
        iso: {"count": int(c), "fraction": round(c / total, 4) if total else 0}
        for iso, c in sorted(counts.items(), key=lambda x: -x[1])
    }
    return {"total_cells": int(total), "isotypes": distribution}


def _clonotype_expansion(obs: pd.DataFrame, clone_col: str, top_n: int) -> dict:
    """Analyze clonotype expansion — top N clones and size distribution."""
    if clone_col not in obs.columns:
        return {"error": f"Column '{clone_col}' not found in obs"}

    clone_counts = obs[clone_col].dropna().value_counts()
    n_unique = len(clone_counts)
    n_expanded = int((clone_counts > 1).sum())
    n_singleton = int((clone_counts == 1).sum())

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
        "n_singleton": n_singleton,
        "clonal_expansion_index": round(n_expanded / n_unique, 4) if n_unique else 0,
        "size_distribution": size_bins,
        "top_clones": top_clones,
    }


def _gene_usage(obs: pd.DataFrame, col: str) -> list[dict]:
    """Count gene usage frequencies."""
    if col not in obs.columns:
        return []
    counts = obs[col].dropna().value_counts()
    total = counts.sum()
    return [
        {"gene": str(g), "count": int(c), "fraction": round(c / total, 4)}
        for g, c in counts.items()
    ]


### ADAPT ### — SHM column may have different names or may not exist
def _shm_analysis(obs: pd.DataFrame, shm_col: str, isotype_col: str) -> dict:
    """Somatic hypermutation rate per isotype (if available)."""
    if shm_col is None or shm_col not in obs.columns:
        return {"available": False, "note": "No SHM column found"}

    result = {"available": True, "per_isotype": {}}
    if isotype_col in obs.columns:
        for iso, grp in obs.groupby(isotype_col):
            vals = grp[shm_col].dropna()
            if len(vals) > 0:
                result["per_isotype"][str(iso)] = {
                    "mean": round(float(vals.mean()), 4),
                    "median": round(float(vals.median()), 4),
                    "std": round(float(vals.std()), 4),
                    "n_cells": int(len(vals)),
                }
    else:
        vals = obs[shm_col].dropna()
        result["overall"] = {
            "mean": round(float(vals.mean()), 4),
            "median": round(float(vals.median()), 4),
            "n_cells": int(len(vals)),
        }
    return result


def main(input_path: str, output_dir: str, params: dict = None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    os.makedirs(output_dir, exist_ok=True)

    adata = sc.read_h5ad(input_path)
    b_adata = _subset_b_cells(adata, p)
    obs = b_adata.obs

    isotype_dist = _isotype_distribution(obs, p["isotype_col"])
    clone_exp = _clonotype_expansion(obs, p["clone_id_col"], p["top_n_clones"])
    v_usage = _gene_usage(obs, p["v_gene_col"])
    j_usage = _gene_usage(obs, p["j_gene_col"])
    shm = _shm_analysis(obs, p["shm_col"], p["isotype_col"])

    summary = {
        "status": "success",
        "n_b_cells": int(b_adata.n_obs),
        "isotype_distribution": isotype_dist,
        "vdj_gene_usage": {
            "v_gene_col": p["v_gene_col"],
            "j_gene_col": p["j_gene_col"],
            "n_v_genes": len(v_usage),
            "n_j_genes": len(j_usage),
        },
        "somatic_hypermutation": shm,
    }

    with open(os.path.join(output_dir, "bcr_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "clonotype_expansion.json"), "w") as f:
        json.dump(clone_exp, f, indent=2)

    if v_usage:
        pd.DataFrame(v_usage).to_csv(
            os.path.join(output_dir, "bcr_vgene_usage.csv"), index=False
        )

    print(f"BCR analysis complete: {b_adata.n_obs} B cells")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="08_bcr_analysis: BCR repertoire analysis")
    parser.add_argument("--input", required=True, help="Path to annotated h5ad")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--params", default="{}", help="JSON string of parameters")
    args = parser.parse_args()

    result = main(args.input, args.output, json.loads(args.params))
    print(json.dumps(result, indent=2))
