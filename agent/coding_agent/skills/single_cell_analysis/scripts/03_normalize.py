"""
03_normalize.py — Normalization and highly variable gene selection.
===================================================================
Normalizes total counts, applies log1p transform, and identifies
highly variable genes for downstream dimensionality reduction.

Input contract:
    --input   Path to h5ad file (typically qc_adata.h5ad from 02_qc)
    --output  Directory to write normalized adata
    --params  JSON string:
              {
                  "target_sum": 10000,    (normalization target)
                  "n_top_genes": 2000     (number of HVGs to select)
              }

Output contract:
    <output_dir>/normalized_adata.h5ad  — Normalized AnnData with HVG annotations
    <output_dir>/normalize_report.json  — Normalization summary
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

    target_sum = params.get("target_sum", 1e4)
    n_top_genes = params.get("n_top_genes", 2000)

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    n_hvg = int(adata.var["highly_variable"].sum())
    print(f"Selected {n_hvg} highly variable genes (requested {n_top_genes})")

    out_path = os.path.join(output_dir, "normalized_adata.h5ad")
    adata.write_h5ad(out_path)

    report = {
        "status": "success",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_highly_variable_genes": n_hvg,
        "params": {
            "target_sum": target_sum,
            "n_top_genes": n_top_genes,
        },
        "output_file": out_path,
    }

    report_path = os.path.join(output_dir, "normalize_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="03_normalize: Normalization and HVG selection")
    parser.add_argument("--input", required=True, help="Path to h5ad file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--params", default="{}", help="JSON string of parameters")
    args = parser.parse_args()

    params = json.loads(args.params)
    result = main(args.input, args.output, params)
    print(json.dumps(result, indent=2))
