# 16_tcr_celltype.R — T cell subtype UMAP + distribution stats
# Input: Seurat RDS + output_dir + [annotation_level: tcell|immune]
# Output: T cell UMAP PDF + stats CSV
#
# Composable block. The agent can adapt the MODULE_DIR or parameters.

### ADAPT ### — module path (resolve relative to project root)
MODULE_DIR <- Sys.getenv("MCP_BIOINFORMATICS_DIR",
    normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."),
        "..", "..", "..", "..", "..", "..", "mcp_servers", "mcp_bioinformatics"),
        mustWork = FALSE))

source(file.path(MODULE_DIR, "scripts/common/tcell_modules/TCR_CellType.R"))
