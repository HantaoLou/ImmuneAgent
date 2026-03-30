# 18_tcr_clonotype.R — TCR clonotype expansion analysis
# Input: Seurat RDS + output_dir
# Output: Clonotype expansion analysis
#
# Composable block. The agent can adapt the MODULE_DIR or parameters.

### ADAPT ### — module path (resolve relative to project root)
MODULE_DIR <- Sys.getenv("MCP_BIOINFORMATICS_DIR",
    normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."),
        "..", "..", "..", "..", "..", "..", "mcp_servers", "mcp_bioinformatics"),
        mustWork = FALSE))

source(file.path(MODULE_DIR, "scripts/common/tcell_modules/TCR_Clonotype.R"))
