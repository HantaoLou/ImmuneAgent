# 12_cellchat.R — Cell-cell communication analysis via CellChat
# Input: <input_rds> <output_dir> [db_type]
# Output: Network PDF + interaction CSVs
#
# Composable block. Agent can adapt parameters or replace module path.

### ADAPT ### — module path
MODULE_DIR <- Sys.getenv("MCP_BIOINFORMATICS_DIR",
    normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."),
        "..", "..", "..", "..", "..", "..", "mcp_servers", "mcp_bioinformatics"),
        mustWork = FALSE))

source(file.path(MODULE_DIR, "scripts/common/immune_modules/Immune_CellChat.R"))
