# 10_trajectory.R — Pseudotime trajectory analysis via monocle3
# Input: <input_rds> <output_dir> [lineage] [num_dim] [cluster_resolution] [root_celltype]
# Output: Pseudotime UMAP PDF + monocle CDS object
#
# Composable block. Agent can adapt parameters or replace module path.

### ADAPT ### — module path
MODULE_DIR <- Sys.getenv("MCP_BIOINFORMATICS_DIR",
    normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."),
        "..", "..", "..", "..", "..", "..", "mcp_servers", "mcp_bioinformatics"),
        mustWork = FALSE))

source(file.path(MODULE_DIR, "scripts/common/immune_modules/Immune_Trajectory.R"))
