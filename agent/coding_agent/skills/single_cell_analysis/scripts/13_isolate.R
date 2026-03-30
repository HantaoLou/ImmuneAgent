# 13_isolate.R — Isolate/subset cells by lineage
# Input: <input_rds> <output_dir> <lineage> [output_rds]
# Output: Isolated RDS subset + subtype distribution CSV
#
# Composable block. Agent can adapt parameters or replace module path.

### ADAPT ### — module path
MODULE_DIR <- Sys.getenv("MCP_BIOINFORMATICS_DIR",
    normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."),
        "..", "..", "..", "..", "..", "..", "mcp_servers", "mcp_bioinformatics"),
        mustWork = FALSE))

source(file.path(MODULE_DIR, "scripts/common/immune_modules/Immune_Isolate.R"))
