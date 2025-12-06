#!/usr/bin/env Rscript
# Differential Expression Gene (DEG) Analysis
# Uses Seurat's FindMarkers for statistical testing

suppressPackageStartupMessages({
  library(Seurat)
  library(ggplot2)
  library(dplyr)
  library(jsonlite)
})

# Set language to English for error messages
Sys.setenv(LANGUAGE = "en")
options(stringsAsFactors = FALSE)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript deg_analysis.R <input_rds> <params_json>")
}

input_rds <- args[1]
params_json <- args[2]

# Parse parameters
params <- fromJSON(params_json)
group_by <- if(!is.null(params$group_by)) params$group_by else "seurat_clusters"
ident_1 <- params$ident_1  # Can be NULL
ident_2 <- params$ident_2  # Can be NULL
test_use <- if(!is.null(params$test_use)) params$test_use else "wilcox"
logfc_threshold <- if(!is.null(params$logfc_threshold)) params$logfc_threshold else 0.25
min_pct <- if(!is.null(params$min_pct)) params$min_pct else 0.1

# Load Seurat object
cat("Loading Seurat object from:", input_rds, "\n")
seurat_obj <- readRDS(input_rds)

# Setup output directory
base_dir <- getwd()  # Server already sets cwd correctly
config_path <- file.path(base_dir, "config.json")
config <- fromJSON(config_path)
output_dir <- file.path(config$base_dir, config$output_dir, "deg_analysis")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Set identity
if (group_by %in% colnames(seurat_obj@meta.data)) {
  Idents(seurat_obj) <- group_by
  cat("Grouping by:", group_by, "\n")
} else {
  stop(paste("Column", group_by, "not found in metadata"))
}

cat("DEG analysis parameters:\n")
cat("  - Test:", test_use, "\n")
cat("  - Log FC threshold:", logfc_threshold, "\n")
cat("  - Min percent:", min_pct, "\n")

# Determine comparison
if (!is.null(ident_1) && ident_1 != "") {
  cat("  - Identity 1:", ident_1, "\n")
  if (!is.null(ident_2) && ident_2 != "") {
    cat("  - Identity 2:", ident_2, "\n")
    comparison_name <- paste0(ident_1, "_vs_", ident_2)
  } else {
    cat("  - Identity 2: all others\n")
    comparison_name <- paste0(ident_1, "_vs_rest")
  }
} else {
  stop("ident_1 must be specified")
}

# Run differential expression
cat("Running DEG analysis...\n")
deg_results <- FindMarkers(seurat_obj,
                          ident.1 = ident_1,
                          ident.2 = if(!is.null(ident_2) && ident_2 != "") ident_2 else NULL,
                          test.use = test_use,
                          logfc.threshold = logfc_threshold,
                          min.pct = min_pct)

# Add gene names as column
deg_results$gene <- rownames(deg_results)

# Reorder columns
deg_results <- deg_results[, c("gene", "p_val", "avg_log2FC", "pct.1", "pct.2", "p_val_adj")]

# Sort by adjusted p-value
deg_results <- deg_results[order(deg_results$p_val_adj), ]

cat("Total DEGs found:", nrow(deg_results), "\n")
cat("Significant DEGs (p_adj < 0.05):", sum(deg_results$p_val_adj < 0.05), "\n")
cat("Upregulated (log2FC > 0):", sum(deg_results$avg_log2FC > 0 & deg_results$p_val_adj < 0.05), "\n")
cat("Downregulated (log2FC < 0):", sum(deg_results$avg_log2FC < 0 & deg_results$p_val_adj < 0.05), "\n")

# Save DEG results
write.csv(deg_results,
          file.path(output_dir, paste0("deg_", comparison_name, ".csv")),
          row.names = FALSE)

# Get top DEGs
top_up <- deg_results %>%
  filter(avg_log2FC > 0, p_val_adj < 0.05) %>%
  head(20)

top_down <- deg_results %>%
  filter(avg_log2FC < 0, p_val_adj < 0.05) %>%
  head(20)

# Volcano plot
pdf(file.path(output_dir, paste0("volcano_", comparison_name, ".pdf")), width = 10, height = 8)
deg_results$significance <- ifelse(deg_results$p_val_adj < 0.05,
                                  ifelse(deg_results$avg_log2FC > 0, "Upregulated", "Downregulated"),
                                  "Not significant")

p <- ggplot(deg_results, aes(x = avg_log2FC, y = -log10(p_val_adj), color = significance)) +
  geom_point(alpha = 0.5, size = 1.5) +
  scale_color_manual(values = c("Upregulated" = "red",
                                "Downregulated" = "blue",
                                "Not significant" = "grey")) +
  geom_vline(xintercept = c(-logfc_threshold, logfc_threshold), linetype = "dashed") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed") +
  labs(title = paste("Volcano Plot:", comparison_name),
       x = "Log2 Fold Change",
       y = "-Log10 Adjusted P-value") +
  theme_minimal() +
  theme(legend.position = "bottom")

print(p)
dev.off()

# Heatmap of top DEGs
if (nrow(deg_results[deg_results$p_val_adj < 0.05, ]) > 0) {
  top_genes <- head(deg_results[deg_results$p_val_adj < 0.05, "gene"], 50)

  pdf(file.path(output_dir, paste0("heatmap_", comparison_name, ".pdf")), width = 10, height = 12)
  print(DoHeatmap(seurat_obj,
                  features = top_genes,
                  cells = WhichCells(seurat_obj, idents = c(ident_1,
                                                           if(!is.null(ident_2) && ident_2 != "") ident_2 else NULL))) +
        ggtitle(paste("Top 50 DEGs:", comparison_name)))
  dev.off()
}

# Feature plots for top DEGs
if (length(top_up$gene) > 0) {
  top_genes_plot <- head(c(top_up$gene, top_down$gene), 9)

  pdf(file.path(output_dir, paste0("features_", comparison_name, ".pdf")), width = 15, height = 15)
  print(FeaturePlot(seurat_obj, features = top_genes_plot, ncol = 3))
  dev.off()
}

# Save summary statistics
deg_summary <- data.frame(
  metric = c("comparison", "total_degs", "significant_degs", "upregulated", "downregulated",
             "test_method", "logfc_threshold", "min_pct"),
  value = c(comparison_name,
            nrow(deg_results),
            sum(deg_results$p_val_adj < 0.05),
            sum(deg_results$avg_log2FC > 0 & deg_results$p_val_adj < 0.05),
            sum(deg_results$avg_log2FC < 0 & deg_results$p_val_adj < 0.05),
            test_use,
            logfc_threshold,
            min_pct)
)
write.csv(deg_summary, file.path(output_dir, paste0("summary_", comparison_name, ".csv")), row.names = FALSE)

cat("DEG analysis completed successfully!\n")
