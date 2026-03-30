"""
04_dim_reduction.py — Dimensionality reduction (PCA, neighbors, UMAP).
=======================================================================
Scales data, computes PCA, builds a neighbor graph, and generates UMAP
embeddings for visualization.

Input contract:
    --input   Path to h5ad file (typically normalized_adata.h5ad from 03_normalize)
    --output  Directory to write adata with embeddings
    --params  JSON string:
              {
                  "max_scale_value": 10,   (clip scaled values)
                  "n_comps": 50,           (PCA components)
                  "n_neighbors": 15,       (neighbor graph k)
                  "n_pcs": 30              (PCs used for neighbors/UMAP)
              }

Output contract:
    <output_dir>/dimred_adata.h5ad       — AnnData with PCA + UMAP in obsm
    <output_dir>/dimred_report.json      — Summary of embeddings computed
"""

import argparse
import json
import os

import scanpy as sc


def main(input_path: str, output_dir: str, params: dict = None) -> dict:
    params = params or {}
    os.makedirs(output_dir, exist_ok=True)

    adata = sc.read_h5ad(input_path)
    print(f"Input: {adata.n_obs} cells x {adata.n_vars} genes")

    max_scale = params.get("max_scale_value", 10)
    n_comps = params.get("n_comps", 50)
    n_neighbors = params.get("n_neighbors", 15)
    n_pcs = params.get("n_pcs", 30)

    sc.pp.scale(adata, max_value=max_scale)
    sc.pp.pca(adata, n_comps=n_comps)

    variance_ratio = adata.uns["pca"]["variance_ratio"]
    cumulative_var = float(sum(variance_ratio[:n_pcs]))
    print(f"PCA: {n_comps} components, top {n_pcs} explain {cumulative_var:.1%} variance")

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata)
    print("UMAP computed")

    out_path = os.path.join(output_dir, "dimred_adata.h5ad")
    adata.write_h5ad(out_path)

    report = {
        "status": "success",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "embeddings": list(adata.obsm.keys()),
        "params": {
            "max_scale_value": max_scale,
            "n_comps": n_comps,
            "n_neighbors": n_neighbors,
            "n_pcs": n_pcs,
        },
        "pca_variance_explained_top_n": round(cumulative_var, 4),
        "output_file": out_path,
    }

    report_path = os.path.join(output_dir, "dimred_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="04_dim_reduction: PCA + UMAP")
    parser.add_argument("--input", required=True, help="Path to h5ad file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--params", default="{}", help="JSON string of parameters")
    args = parser.parse_args()

    params = json.loads(args.params)
    result = main(args.input, args.output, params)
    print(json.dumps(result, indent=2))
