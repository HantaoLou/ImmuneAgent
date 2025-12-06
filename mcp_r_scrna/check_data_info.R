#!/usr/bin/env Rscript
# 检查Seurat对象的基本信息

suppressPackageStartupMessages({
  library(Seurat)
})

# 加载数据
input_rds <- "D:/data/test_data_20251001/Age_Bcells.rds"
cat("加载Seurat对象:", input_rds, "\n")
seurat_obj <- readRDS(input_rds)

# 检查基本信息
cat("\n=== 基本信息 ===\n")
cat("细胞数量:", ncol(seurat_obj), "\n")
cat("基因数量:", nrow(seurat_obj), "\n")

# 检查元数据列
cat("\n=== 元数据列 ===\n")
meta_cols <- colnames(seurat_obj@meta.data)
cat("可用的元数据列:\n")
for(col in meta_cols) {
  cat("  -", col, "\n")
}

# 检查聚类信息
if("seurat_clusters" %in% meta_cols) {
  cat("\n=== seurat_clusters 信息 ===\n")
  clusters <- unique(seurat_obj@meta.data$seurat_clusters)
  cat("聚类数量:", length(clusters), "\n")
  cat("聚类标识:", paste(sort(clusters), collapse=", "), "\n")
  
  # 每个聚类的细胞数量
  cluster_counts <- table(seurat_obj@meta.data$seurat_clusters)
  cat("各聚类细胞数量:\n")
  for(i in names(cluster_counts)) {
    cat("  cluster", i, ":", cluster_counts[i], "cells\n")
  }
}

# 检查其他可能的分组变量
for(col in meta_cols) {
  if(col != "seurat_clusters" && is.factor(seurat_obj@meta.data[[col]]) || 
     is.character(seurat_obj@meta.data[[col]])) {
    unique_vals <- unique(seurat_obj@meta.data[[col]])
    if(length(unique_vals) <= 20 && length(unique_vals) > 1) {
      cat("\n===", col, "信息 ===\n")
      cat("唯一值数量:", length(unique_vals), "\n")
      cat("唯一值:", paste(sort(unique_vals), collapse=", "), "\n")
      
      # 计数
      val_counts <- table(seurat_obj@meta.data[[col]])
      cat("各组细胞数量:\n")
      for(i in names(val_counts)) {
        cat("  ", i, ":", val_counts[i], "cells\n")
      }
    }
  }
}

cat("\n检查完成!\n")