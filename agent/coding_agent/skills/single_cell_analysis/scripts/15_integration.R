# 15_integration.R — Integrate CSV + RDS into a merged Seurat object
# Input: --csv data.csv --rds data.rds --csv-fields "Batch,barcode" --rds-fields "rownames" --output result.rds
# Output: Merged Seurat RDS
#
# Composable block. The agent can adapt the MODULE_DIR or parameters.
# NOTE: integrate_all.R uses optparse (argparse-style); args pass through directly.

### ADAPT ### — module path (resolve relative to project root)
MODULE_DIR <- Sys.getenv("MCP_BIOINFORMATICS_DIR",
    normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."),
        "..", "..", "..", "..", "..", "..", "mcp_servers", "mcp_bioinformatics"),
        mustWork = FALSE))

source(file.path(MODULE_DIR, "scripts/combine/integrate_all.R"))
