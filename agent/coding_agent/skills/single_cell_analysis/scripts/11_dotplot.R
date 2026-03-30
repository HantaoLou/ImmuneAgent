# 11_dotplot.R — Marker gene dotplot visualization
# Input: <input_rds> <output_dir> [marker_set: all|tcell|bcell|myeloid|nk|granulocyte|stromal]
# Output: Dotplot PDF + marker gene CSVs
#
# Composable block. Agent can adapt parameters or replace module path.

### ADAPT ### — module path
MODULE_DIR <- Sys.getenv("MCP_BIOINFORMATICS_DIR",
    normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."),
        "..", "..", "..", "..", "..", "..", "mcp_servers", "mcp_bioinformatics"),
        mustWork = FALSE))

source(file.path(MODULE_DIR, "scripts/common/immune_modules/Immune_DotPlot.R"))
