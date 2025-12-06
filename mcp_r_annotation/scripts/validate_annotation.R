#!/usr/bin/env Rscript

# Cross-Reference Annotation Validation
# Validates annotations by comparing with SingleR or another annotation column

suppressPackageStartupMessages({
  library(Seurat)
  library(SingleR)
  library(celldex)
  library(jsonlite)
  library(ggplot2)
  library(SingleCellExperiment)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("No parameters provided. Expecting JSON string.")
}

# Parse JSON parameters
params <- fromJSON(args[1])
input_file <- params$input_file
annotation_column1 <- params$annotation_column1
annotation_column2 <- params$annotation_column2
reference_dataset <- ifelse(is.null(params$reference_dataset),
                            "MonacoImmuneData",
                            params$reference_dataset)

# Setup output directory
script_name <- "validate_annotation"
# 使用当前工作目录作为基础目录
base_dir <- getwd()
output_dir <- file.path(base_dir, "output", script_name)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "plots"), recursive = TRUE, showWarnings = FALSE)

# Load Seurat object
cat(sprintf("Loading Seurat object from: %s\n", input_file))
seurat_obj <- readRDS(input_file)

# Check if annotation_column1 exists
if (!annotation_column1 %in% colnames(seurat_obj@meta.data)) {
  stop(sprintf("Annotation column '%s' not found in metadata", annotation_column1))
}

# Determine validation strategy
if (!is.null(annotation_column2) && annotation_column2 != "") {
  # Compare two existing annotation columns
  cat(sprintf("Comparing annotations: %s vs %s\n",
              annotation_column1, annotation_column2))

  if (!annotation_column2 %in% colnames(seurat_obj@meta.data)) {
    stop(sprintf("Annotation column '%s' not found in metadata", annotation_column2))
  }

  annot1 <- seurat_obj@meta.data[[annotation_column1]]
  annot2 <- seurat_obj@meta.data[[annotation_column2]]

} else {
  # Validate against SingleR predictions
  cat(sprintf("Validating %s against SingleR (%s)...\n",
              annotation_column1, reference_dataset))

  # Load reference
  ref_data <- switch(reference_dataset,
    "HumanPrimaryCellAtlasData" = celldex::HumanPrimaryCellAtlasData(),
    "BlueprintEncodeData" = celldex::BlueprintEncodeData(),
    "MonacoImmuneData" = celldex::MonacoImmuneData(),
    "DatabaseImmuneCellExpressionData" = celldex::DatabaseImmuneCellExpressionData(),
    stop(sprintf("Unknown reference dataset: %s", reference_dataset))
  )

  # Run SingleR at cell level for validation
  test_data <- GetAssayData(seurat_obj, slot = "data")
  singler_pred <- SingleR(
    test = test_data,
    ref = ref_data,
    labels = ref_data$label.main,
    assay.type.test = "logcounts",
    assay.type.ref = "logcounts"
  )

  annot1 <- seurat_obj@meta.data[[annotation_column1]]
  annot2 <- singler_pred$labels
  annotation_column2 <- sprintf("SingleR_%s", reference_dataset)
}

# Calculate confusion matrix
confusion_matrix <- table(annot1, annot2)
confusion_file <- file.path(output_dir, "confusion_matrix.csv")
write.csv(as.data.frame.matrix(confusion_matrix), confusion_file)
cat(sprintf("Saved confusion matrix to: %s\n", confusion_file))

# Calculate agreement metrics
total_cells <- length(annot1)
exact_matches <- sum(annot1 == annot2, na.rm = TRUE)
agreement_rate <- round(exact_matches / total_cells * 100, 2)

cat(sprintf("\nAgreement: %d/%d cells (%.2f%%)\n",
            exact_matches, total_cells, agreement_rate))

# Identify discrepancies
discrepancies <- data.frame(
  cell_barcode = colnames(seurat_obj),
  annotation1 = annot1,
  annotation2 = annot2,
  match = annot1 == annot2,
  stringsAsFactors = FALSE
)

discrepancies_only <- discrepancies[!discrepancies$match, ]
discrepancy_file <- file.path(output_dir, "discrepancies.csv")
write.csv(discrepancies_only, discrepancy_file, row.names = FALSE)
cat(sprintf("Saved %d discrepancies to: %s\n",
            nrow(discrepancies_only), discrepancy_file))

# Calculate per-celltype agreement
celltype_agreement <- data.frame(
  celltype = unique(annot1),
  total_cells = sapply(unique(annot1), function(ct) sum(annot1 == ct)),
  matching_cells = sapply(unique(annot1),
                          function(ct) sum(annot1 == ct & annot2 == ct, na.rm = TRUE)),
  stringsAsFactors = FALSE
)
celltype_agreement$agreement_rate <- round(
  celltype_agreement$matching_cells / celltype_agreement$total_cells * 100, 2
)

agreement_file <- file.path(output_dir, "celltype_agreement.csv")
write.csv(celltype_agreement, agreement_file, row.names = FALSE)
cat(sprintf("Saved per-celltype agreement to: %s\n", agreement_file))

# Generate confusion matrix heatmap
cat("Generating confusion matrix heatmap...\n")
pdf(file.path(output_dir, "plots", "confusion_heatmap.pdf"),
    width = 12, height = 10)

# Normalize by row for better visualization
confusion_norm <- sweep(confusion_matrix, 1,
                       rowSums(confusion_matrix), "/")

heatmap(confusion_norm,
        main = sprintf("Confusion Matrix\n%s vs %s",
                      annotation_column1, annotation_column2),
        xlab = annotation_column2,
        ylab = annotation_column1,
        col = colorRampPalette(c("white", "lightblue", "darkblue"))(100),
        scale = "none",
        margins = c(10, 10))
dev.off()

# Generate agreement bar plot
cat("Generating agreement bar plot...\n")
pdf(file.path(output_dir, "plots", "celltype_agreement.pdf"),
    width = 10, height = 6)

p <- ggplot(celltype_agreement, aes(x = reorder(celltype, agreement_rate),
                                    y = agreement_rate)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = sprintf("%.1f%%", agreement_rate)),
            hjust = -0.2, size = 3) +
  coord_flip() +
  ylim(0, 110) +
  labs(title = "Per-CellType Agreement Rate",
       x = "Cell Type",
       y = "Agreement Rate (%)") +
  theme_minimal()

print(p)
dev.off()

# Generate UMAP comparison if available
if ("umap" %in% names(seurat_obj@reductions)) {
  cat("Generating UMAP comparison...\n")

  # Add annotation2 to metadata temporarily for plotting
  seurat_obj@meta.data$temp_annot2 <- annot2

  pdf(file.path(output_dir, "plots", "umap_comparison.pdf"),
      width = 14, height = 6)

  p1 <- DimPlot(seurat_obj,
                reduction = "umap",
                group.by = annotation_column1,
                label = TRUE,
                label.size = 3) +
    ggtitle(annotation_column1)

  p2 <- DimPlot(seurat_obj,
                reduction = "umap",
                group.by = "temp_annot2",
                label = TRUE,
                label.size = 3) +
    ggtitle(annotation_column2)

  print(p1 | p2)
  dev.off()

  # Clean up temporary column
  seurat_obj@meta.data$temp_annot2 <- NULL
}

# Save validation report
validation_report <- list(
  annotation_column1 = annotation_column1,
  annotation_column2 = annotation_column2,
  total_cells = total_cells,
  exact_matches = exact_matches,
  agreement_rate = agreement_rate,
  discrepancies = nrow(discrepancies_only),
  celltype_agreement = celltype_agreement
)

result <- list(
  status = "success",
  validation_report = validation_report
)

result_json <- file.path(output_dir, "result.json")
write_json(result, result_json, pretty = TRUE)

cat("\n=== Annotation Validation Complete ===\n")
cat(sprintf("Comparing: %s vs %s\n", annotation_column1, annotation_column2))
cat(sprintf("Total cells: %d\n", total_cells))
cat(sprintf("Agreement: %.2f%%\n", agreement_rate))
cat(sprintf("Discrepancies: %d\n", nrow(discrepancies_only)))
