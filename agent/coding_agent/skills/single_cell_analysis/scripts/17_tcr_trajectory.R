# 17_tcr_trajectory.R — Pseudotime trajectory analysis of T cells
# Input: Seurat RDS + output_dir + [num_dim] [cluster_resolution] [min_gene_cells] [root_celltype]
# Output: Pseudotime trajectory PDF
#
# Composable block. The agent can adapt the MODULE_DIR or parameters.

### ADAPT ### — module path (resolve relative to project root)
MODULE_DIR <- Sys.getenv("MCP_BIOINFORMATICS_DIR",
    normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."),
        "..", "..", "..", "..", "..", "..", "mcp_servers", "mcp_bioinformatics"),
        mustWork = FALSE))

source(file.path(MODULE_DIR, "scripts/common/tcell_modules/TCR_Trajectory.R"))
