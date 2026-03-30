#!/usr/bin/env Rscript

# 1. Environment Setup
suppressMessages({
  library(optparse)
  library(monocle3)
  library(Seurat)
  library(ggplot2)
  library(dplyr)
  library(magrittr)
})

# Dynamically locate and source the custom utility library
args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- sub(file_arg, "", args[grep(file_arg, args)])
if (length(script_path) > 0) {
  script_dir <- dirname(normalizePath(script_path))
  source(file.path(script_dir, "utils.R"))
} else {
  # Fallback for interactive sessions
  source("utils.R")
}

# 2. Define General Command-Line Arguments
option_list <- list(
  make_option(c("-i", "--input"), type="character", help="Path to input Seurat RDS file"),
  make_option(c("-g", "--group_col"), type="character", default="orig.ident", help="Metadata column for grouping"),
  make_option(c("-c", "--condition"), type="character", help="Specific group/condition to analyze"),
  make_option(c("-l", "--label_col"), type="character", default="celltype", help="Metadata column for cell type labels"),
  make_option(c("-r", "--root_type"), type="character", help="Cell type designated as the trajectory root"),
  make_option(c("-o", "--outdir"), type="character", default="./results", help="Output directory")
)

opt <- parse_args(OptionParser(option_list=option_list))

# 3. Data Loading and Subsetting
if (!file.exists(opt$input)) stop("Input file not found: ", opt$input)
cat(paste0("Reading data and filtering for group: ", opt$condition, "...\n"))

scRNA <- readRDS(opt$input)
# Dynamically subset based on provided group column
scRNA$temp_group <- scRNA[[opt$group_col]]
sub_obj <- subset(scRNA, temp_group == opt$condition)

# 4. Construct Monocle3 CDS Object
counts <- GetAssayData(sub_obj, assay = "RNA", slot = "counts")
# Filter out lowly expressed genes (detected in fewer than 3 cells)
counts <- counts[rowSums(counts > 0) >= 3, ]

gene_metadata <- data.frame(gene_short_name = rownames(counts), row.names = rownames(counts))
cds <- new_cell_data_set(counts, 
                         cell_metadata = sub_obj@meta.data, 
                         gene_metadata = gene_metadata)

# 5. Coordinate Synchronization (Using utils.R functions)
# Auto-detect available reduction: UMAP > t-SNE > PCA
available_reds <- names(sub_obj@reductions)
target_red <- if("umap" %in% available_reds) "umap" else if("tsne" %in% available_reds) "tsne" else "pca"

cat(paste0("Synchronizing coordinates using: ", target_red, "\n"))
cds <- preprocess_cds(cds, num_dim = 50) # Pre-processing is required before sync
cds <- sync_coordinates(cds, sub_obj, reduction = target_red)

# 6. Learn Trajectory Graph
# Low resolution is used to maintain trajectory continuity
cds <- cluster_cells(cds, resolution = 1e-4) 
cds <- learn_graph(cds, use_partition = TRUE, close_loop = FALSE)

# 7. Identify Root and Calculate Pseudotime (Using utils.R functions)
cat(paste0("Identifying trajectory root for cell type: ", opt$root_type, "...\n"))
root_node <- get_root_nodes(cds, 
                            label_col = opt$label_col, 
                            root_type = opt$root_type, 
                            reduction = toupper(target_red))

cds <- order_cells(cds, root_pr_nodes = root_node)

# 8. Result Export and Visualization
if(!dir.exists(opt$outdir)) dir.create(opt$outdir, recursive = TRUE)

# Save CDS object for downstream analysis
saveRDS(cds, file = file.path(opt$outdir, "trajectory_result.rds"))

# Plotting
p <- plot_cells(cds, 
                color_cells_by = "pseudotime", 
                label_cell_groups = FALSE, 
                label_leaves = FALSE, 
                label_branch_points = FALSE,
                graph_label_size = 1.5) +
     ggtitle(paste0("Trajectory: ", opt$condition, " (Root: ", opt$root_type, ")"))

# Save as PDF for high-quality publication-ready output
ggsave(file.path(opt$outdir, "pseudotime_umap.pdf"), p, width = 8, height = 7)
# Save as PNG for quick preview in the Markdown report
ggsave(file.path(opt$outdir, "pseudotime_umap.png"), p, width = 8, height = 7)

cat("Analysis successfully completed. Results saved in: ", opt$outdir, "\n")