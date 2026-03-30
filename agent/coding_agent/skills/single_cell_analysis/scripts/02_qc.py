"""
02_qc.py — Quality control filtering for single-cell data.
============================================================
Filters cells and genes by standard QC metrics: minimum genes per cell,
minimum cells per gene, and mitochondrial gene percentage.

Input contract:
    --input   Path to h5ad file (typically loaded_adata.h5ad from 01_load)
    --output  Directory to write filtered adata
    --params  JSON string:
              {
                  "min_genes": 200,       (min genes per cell)
                  "min_cells": 3,         (min cells per gene)
                  "mito_prefix": "MT-",   (### ADAPT ### human=MT-, mouse=mt-)
                  "max_mito_percent": 20
              }

Output contract:
    <output_dir>/qc_adata.h5ad    — Filtered AnnData
    <output_dir>/qc_report.json   — QC summary (cells before/after, genes before/after)
"""

import argparse
import json
import os

import scanpy as sc


def main(input_path: str, output_dir: str, params: dict = None) -> dict:
    params = params or {}
    os.makedirs(output_dir, exist_ok=True)

    adata = sc.read_h5ad(input_path)
    n_cells_before = adata.n_obs
    n_genes_before = adata.n_vars
    print(f"Before QC: {n_cells_before} cells x {n_genes_before} genes")

    min_genes = params.get("min_genes", 200)
    min_cells = params.get("min_cells", 3)

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    n_after_basic = adata.n_obs
    print(f"After basic filters (min_genes={min_genes}, min_cells={min_cells}): "
          f"{n_after_basic} cells")

    ### ADAPT ### — mitochondrial gene prefix depends on species
    # Human genes: MT-CO1, MT-ND1, etc. (prefix "MT-")
    # Mouse genes: mt-Co1, mt-Nd1, etc. (prefix "mt-")
    mito_prefix = params.get("mito_prefix", "MT-")
    adata.var["mt"] = adata.var_names.str.startswith(mito_prefix)

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, inplace=True
    )

    max_mito = params.get("max_mito_percent", 20)
    mito_mask = adata.obs["pct_counts_mt"] < max_mito
    n_mito_filtered = int((~mito_mask).sum())
    adata = adata[mito_mask, :].copy()

    print(f"Removed {n_mito_filtered} cells with >={max_mito}% mitochondrial reads")
    print(f"After QC: {adata.n_obs} cells x {adata.n_vars} genes")

    out_path = os.path.join(output_dir, "qc_adata.h5ad")
    adata.write_h5ad(out_path)

    report = {
        "status": "success",
        "n_cells_before": n_cells_before,
        "n_genes_before": n_genes_before,
        "n_cells_after_basic": n_after_basic,
        "n_cells_mito_filtered": n_mito_filtered,
        "n_cells_after": int(adata.n_obs),
        "n_genes_after": int(adata.n_vars),
        "params": {
            "min_genes": min_genes,
            "min_cells": min_cells,
            "mito_prefix": mito_prefix,
            "max_mito_percent": max_mito,
        },
        "qc_metrics": {
            "median_genes_per_cell": float(adata.obs["n_genes_by_counts"].median()),
            "median_counts_per_cell": float(adata.obs["total_counts"].median()),
            "median_pct_mito": float(adata.obs["pct_counts_mt"].median()),
        },
        "output_file": out_path,
    }

    report_path = os.path.join(output_dir, "qc_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="02_qc: Quality control filtering")
    parser.add_argument("--input", required=True, help="Path to h5ad file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--params", default="{}", help="JSON string of parameters")
    args = parser.parse_args()

    params = json.loads(args.params)
    result = main(args.input, args.output, params)
    print(json.dumps(result, indent=2))
