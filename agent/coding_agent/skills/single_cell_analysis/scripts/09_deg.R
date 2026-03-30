# 09_deg.R — Differential expression analysis between cell types
# Input: <input_rds> <output_dir> [group1] [group2] [logfc_threshold] [min_pct]
# Output: DEG CSV + volcano PDF
#
# Composable block. Agent can adapt parameters or replace module path.

### ADAPT ### — module path
MODULE_DIR <- Sys.getenv("MCP_BIOINFORMATICS_DIR",
    normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."),
        "..", "..", "..", "..", "..", "..", "mcp_servers", "mcp_bioinformatics"),
        mustWork = FALSE))

source(file.path(MODULE_DIR, "scripts/common/immune_modules/Immune_DEG.R"))
