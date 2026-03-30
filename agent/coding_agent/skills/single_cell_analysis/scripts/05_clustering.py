"""
05_clustering.py — Leiden clustering.
======================================
Performs graph-based clustering using the Leiden algorithm on the
precomputed neighbor graph from 04_dim_reduction.

Input contract:
    --input   Path to h5ad file (typically dimred_adata.h5ad from 04_dim_reduction)
    --output  Directory to write clustered adata
    --params  JSON string:
              {
                  "resolution": 0.8   (### ADAPT ### higher=more clusters)
              }

Output contract:
    <output_dir>/clustered_adata.h5ad    — AnnData with obs["leiden"] column
    <output_dir>/cluster_summary.json    — Per-cluster cell counts
"""

import argparse
import json
import os

import scanpy as sc


def main(input_path: str, output_dir: str, params: dict = None) -> dict:
    params = params or {}
    os.makedirs(output_dir, exist_ok=True)

    adata = sc.read_h5ad(input_path)
    print(f"Input: {adata.n_obs} cells")

    ### ADAPT ### — resolution controls granularity of clustering
    # Lower resolution (0.3-0.5) gives fewer, broader clusters
    # Higher resolution (1.0-2.0) gives more, finer clusters
    # Adjust based on expected cell type diversity and dataset size
    resolution = params.get("resolution", 0.8)

    sc.tl.leiden(adata, resolution=resolution, key_added="leiden")

    clusters = sorted(adata.obs["leiden"].unique(), key=lambda x: int(x))
    n_clusters = len(clusters)
    print(f"Found {n_clusters} clusters at resolution {resolution}")

    cluster_summary = []
    for cluster in clusters:
        n_cells = int((adata.obs["leiden"] == cluster).sum())
        cluster_summary.append({
            "cluster_id": str(cluster),
            "n_cells": n_cells,
        })

    out_path = os.path.join(output_dir, "clustered_adata.h5ad")
    adata.write_h5ad(out_path)

    summary_path = os.path.join(output_dir, "cluster_summary.json")
    with open(summary_path, "w") as f:
        json.dump(cluster_summary, f, indent=2)

    report = {
        "status": "success",
        "n_cells": int(adata.n_obs),
        "n_clusters": n_clusters,
        "resolution": resolution,
        "clusters": cluster_summary,
        "output_file": out_path,
    }

    report_path = os.path.join(output_dir, "clustering_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="05_clustering: Leiden clustering")
    parser.add_argument("--input", required=True, help="Path to h5ad file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--params", default="{}", help="JSON string of parameters")
    args = parser.parse_args()

    params = json.loads(args.params)
    result = main(args.input, args.output, params)
    print(json.dumps(result, indent=2))
