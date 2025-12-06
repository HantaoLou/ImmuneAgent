# 整合fluBcells_BCR_predict_processed.csv和fluBcells.rds文件
# 作者: R代码专家
# 日期: 2024

# 加载必要的库
library(Seurat)
library(dplyr)
library(readr)

# 设置工作目录和文件路径
# csv_file、rds_file和output_path应该作为参数传递给脚本
# 例如: Rscript integrate_bcr_data.R csv_file rds_file output_path
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 3) {
  csv_file <- args[1]
  rds_file <- args[2]
  output_path <- args[3]
} else {
  stop("Usage: Rscript integrate_bcr_data.R <csv_file> <rds_file> <output_path>")
}

# 处理输出路径
if (dir.exists(output_path)) {
  # 如果是目录，使用原文件名
  filename <- basename(rds_file)
  output_file <- file.path(output_path, filename)
} else {
  # 如果是文件路径，直接使用
  output_file <- output_path
}

# 读取CSV文件
cat("读取CSV文件...\n")
tryCatch({
  combine_barcode <- read_csv(csv_file, locale = locale(encoding = "UTF-8"))
  cat("CSV文件包含", nrow(combine_barcode), "行数据\n")
  cat("CSV文件列名:", paste(colnames(combine_barcode), collapse = ", "), "\n")
}, error = function(e) {
  cat("CSV文件读取失败，尝试其他编码...\n")
  combine_barcode <<- read.csv(csv_file, stringsAsFactors = FALSE)
  cat("CSV文件包含", nrow(combine_barcode), "行数据\n")
  cat("CSV文件列名:", paste(colnames(combine_barcode), collapse = ", "), "\n")
})

# 读取RDS文件
cat("读取RDS文件...\n")
cat("文件路径:", rds_file, "\n")
cat("文件是否存在:", file.exists(rds_file), "\n")
tryCatch({
  seurat_obj <- readRDS(rds_file)
  cat("Seurat对象包含", ncol(seurat_obj), "个细胞\n")
}, error = function(e) {
  cat("RDS文件读取失败，错误信息:", e$message, "\n")
  cat("尝试使用不同的方法读取...\n")
  # 尝试使用load函数
  tryCatch({
    load(rds_file)
    # 查找环境中的Seurat对象
    obj_names <- ls()
    cat("加载的对象:", paste(obj_names, collapse = ", "), "\n")
    stop("请手动指定Seurat对象名称")
  }, error = function(e2) {
    stop(paste("无法读取RDS文件:", e$message))
  })
})

# 检查metadata中是否存在combine_barcode字段
if (!"combine_barcode" %in% colnames(seurat_obj@meta.data)) {
  stop("错误: Seurat对象的metadata中未找到combine_barcode字段")
}

# 显示当前metadata的列名
cat("当前metadata包含以下字段:\n")
print(colnames(seurat_obj@meta.data))

# 动态检测CSV文件中的variant相关字段
csv_cols <- colnames(combine_barcode)

# 提取各类字段的数量（包括基础字段和带后缀的字段）
variant_seq_cols <- grep("^variant_seq($|\\.|_)", csv_cols, value = TRUE)
variant_name_cols <- grep("^variant_name($|\\.|_)", csv_cols, value = TRUE)
bind_output_cols <- grep("^bind_output($|\\.|_)", csv_cols, value = TRUE)
bind_predict_cols <- grep("^bind_predict($|\\.|_)", csv_cols, value = TRUE)

# 检测neu相关字段
neu_output_cols <- grep("^neu_output($|\\.|_)", csv_cols, value = TRUE)
neu_predict_cols <- grep("^neu_predict($|\\.|_)", csv_cols, value = TRUE)

# 动态构建要合并的列
merge_cols <- c("Heavy", "Light", 
                variant_seq_cols,
                variant_name_cols,
                bind_output_cols,
                bind_predict_cols,
                neu_output_cols,
                neu_predict_cols)

cat("检测到的字段数量:\n")
cat("  variant_seq:", length(variant_seq_cols), "\n")
cat("  variant_name:", length(variant_name_cols), "\n")
cat("  bind_output:", length(bind_output_cols), "\n")
cat("  bind_predict:", length(bind_predict_cols), "\n")
cat("  neu_output:", length(neu_output_cols), "\n")
cat("  neu_predict:", length(neu_predict_cols), "\n")

# 检查CSV文件中是否包含所有需要的列
missing_cols <- setdiff(merge_cols, colnames(combine_barcode))
if (length(missing_cols) > 0) {
  cat("警告: CSV文件中缺少以下列:\n")
  print(missing_cols)
}

# 只选择存在的列进行合并
available_cols <- intersect(merge_cols, colnames(combine_barcode))
cat("将合并以下", length(available_cols), "列:\n")
print(available_cols)

# 准备合并数据
merge_data <- combine_barcode[, c("combine_barcode", available_cols)]

# 获取当前metadata
current_meta <- seurat_obj@meta.data

# 检查并移除已存在的重复字段，避免产生.x和.y后缀
existing_cols <- intersect(available_cols, colnames(current_meta))
if (length(existing_cols) > 0) {
  cat("检测到已存在的字段，将先移除以避免重复:\n")
  print(existing_cols)
  current_meta <- current_meta[, !colnames(current_meta) %in% existing_cols, drop = FALSE]
}

# 执行左连接，直接使用combine_barcode进行匹配
cat("执行数据整合...\n")
integrated_meta <- current_meta %>%
  left_join(merge_data, by = c("combine_barcode" = "combine_barcode"))

# 检查合并结果并处理缺失列
if (length(available_cols) > 0) {
  for (col in available_cols) {
    if (col %in% colnames(integrated_meta)) {
      # 将NA值替换为"NA"字符串
      integrated_meta[[col]][is.na(integrated_meta[[col]])] <- "NA"
    } else {
      # 如果列不存在，创建一个全为"NA"的列
      cat("警告: 列", col, "在合并后不存在，创建默认值\n")
      integrated_meta[[col]] <- rep("NA", nrow(integrated_meta))
    }
  }
}

# 保持原有的行名
rownames(integrated_meta) <- rownames(current_meta)

# 更新Seurat对象的metadata
seurat_obj@meta.data <- integrated_meta

# 显示整合结果统计
cat("\n整合完成!\n")
cat("原始metadata列数:", ncol(current_meta), "\n")
cat("整合后metadata列数:", ncol(integrated_meta), "\n")
cat("新增列数:", length(available_cols), "\n")

# 检查匹配情况
matched_cells <- sum(!is.na(integrated_meta[[available_cols[1]]]))
cat("成功匹配的细胞数:", matched_cells, "/", nrow(integrated_meta), "\n")
cat("匹配率:", round(matched_cells/nrow(integrated_meta)*100, 2), "%\n")

###############################################################################
#                          添加UMAP降维分析                                  #
###############################################################################

cat("\n=== 开始UMAP降维分析 ===\n")

# 检查是否已经存在降维结果
if ("umap" %in% names(seurat_obj@reductions)) {
  cat("检测到已存在UMAP降维结果，提取坐标...\n")
  
  # 提取UMAP坐标
  umap_coords <- seurat_obj@reductions$umap@cell.embeddings
  
  # 添加到metadata
  seurat_obj@meta.data$UMAP_1 <- umap_coords[, 1]
  seurat_obj@meta.data$UMAP_2 <- umap_coords[, 2]
  
  cat("已从现有降维结果提取UMAP坐标\n")
  
} else {
  cat("未检测到UMAP降维结果，开始执行降维分析...\n")
  
  # 检查是否需要进行标准化和PCA
  if (!"pca" %in% names(seurat_obj@reductions)) {
    cat("执行数据标准化和PCA...\n")
    
    # 寻找高变基因（如果还没有）
    if (length(VariableFeatures(seurat_obj)) == 0) {
      seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)
      cat("已识别", length(VariableFeatures(seurat_obj)), "个高变基因\n")
    }
    
    # 数据标准化
    seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 10000)
    cat("数据标准化完成\n")
    
    # 数据缩放（只对高变基因进行缩放以节省内存）
    seurat_obj <- ScaleData(seurat_obj, features = VariableFeatures(seurat_obj))
    cat("数据缩放完成\n")
    
    # PCA分析
    seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(object = seurat_obj), verbose = FALSE)
    cat("PCA分析完成\n")
  } else {
    cat("检测到已存在PCA结果，跳过PCA步骤\n")
  }
  
  # 动态确定PCA维度数量
  pca_dims <- min(20, ncol(seurat_obj@reductions$pca@cell.embeddings))
  cat("使用PCA维度: 1:", pca_dims, "\n")
  
  # 构建邻接图（用于聚类和UMAP）
  cat("构建邻接图...\n")
  seurat_obj <- FindNeighbors(seurat_obj, dims = 1:pca_dims, verbose = FALSE)
  
  # 进行基础聚类以便后续使用
  seurat_obj <- FindClusters(seurat_obj, resolution = 0.5, verbose = FALSE)
  
  # 执行UMAP降维
  cat("执行UMAP降维...\n")
  seurat_obj <- RunUMAP(seurat_obj, dims = 1:pca_dims, verbose = FALSE)
  
  # 提取UMAP坐标并添加到metadata
  umap_coords <- seurat_obj@reductions$umap@cell.embeddings
  seurat_obj@meta.data$UMAP_1 <- umap_coords[, 1]
  seurat_obj@meta.data$UMAP_2 <- umap_coords[, 2]
  
  cat("UMAP降维完成，坐标已添加到metadata\n")
}

# 显示UMAP坐标统计
cat("\nUMAP坐标统计:\n")
cat("UMAP_1 范围:", round(min(seurat_obj@meta.data$UMAP_1), 3), "到", round(max(seurat_obj@meta.data$UMAP_1), 3), "\n")
cat("UMAP_2 范围:", round(min(seurat_obj@meta.data$UMAP_2), 3), "到", round(max(seurat_obj@meta.data$UMAP_2), 3), "\n")

###############################################################################
#                        基于marker基因的细胞类型注释                        #
###############################################################################

cat("\n=== 开始基于marker基因的细胞类型注释 ===\n")

# 定义B细胞亚型的marker基因
B_cell_markers <- list(
  "Naive" = c("TCL1A", "FCER2", "IL4R", "IGHD", "CD23", "CD38"),
  "Memory" = c("CD27", "TNFRSF13B", "TNFRSF13C", "AIM2", "GPR183"),
  "Germinal_Center" = c("BCL6", "AICDA", "CD10", "MME", "CXCR4", "LMO2"),
  "Plasma" = c("PRDM1", "XBP1", "MZB1", "JCHAIN", "CD138", "SDC1"),
  "Activated" = c("CD69", "CD83", "EGR1", "FOS", "JUN", "NR4A2"),
  "Proliferating" = c("MKI67", "TOP2A", "PCNA", "STMN1", "TUBB"),
  "Atypical" = c("FCRL5", "ITGAX", "TBX21", "FCRL4", "CD274"),
  "Transitional" = c("CD24", "CD38", "FCER2", "IL4R")
)

# 计算每个细胞类型的marker基因评分
calculate_marker_score <- function(seurat_obj, markers) {
  # 获取表达矩阵
  expr_matrix <- GetAssayData(seurat_obj, assay = "RNA", slot = "data")
  
  # 过滤存在的基因
  available_markers <- markers[markers %in% rownames(expr_matrix)]
  
  if(length(available_markers) == 0) {
    return(rep(0, ncol(expr_matrix)))
  }
  
  # 计算平均表达
  if(length(available_markers) == 1) {
    scores <- as.numeric(expr_matrix[available_markers, ])
  } else {
    scores <- colMeans(expr_matrix[available_markers, , drop = FALSE])
  }
  
  return(scores)
}

# 为每个细胞分配细胞类型
assign_cell_types <- function(seurat_obj, marker_list, min_score = 0.1) {
  # 计算每个细胞类型的评分
  scores_matrix <- matrix(0, nrow = ncol(seurat_obj), ncol = length(marker_list))
  colnames(scores_matrix) <- names(marker_list)
  rownames(scores_matrix) <- colnames(seurat_obj)
  
  for(cell_type in names(marker_list)) {
    cat("计算", cell_type, "的marker评分...\n")
    scores_matrix[, cell_type] <- calculate_marker_score(seurat_obj, marker_list[[cell_type]])
  }
  
  # 为每个细胞分配细胞类型
  predicted_types <- apply(scores_matrix, 1, function(x) {
    max_score <- max(x)
    if(max_score < min_score) {
      return("Unknown")
    }
    return(names(x)[which.max(x)])
  })
  
  # 计算置信度
  confidence <- apply(scores_matrix, 1, function(x) {
    sorted_scores <- sort(x, decreasing = TRUE)
    if(sorted_scores[1] == 0) return(0)
    if(length(sorted_scores) == 1) return(1)
    return((sorted_scores[1] - sorted_scores[2]) / sorted_scores[1])
  })
  
  return(list(
    predicted_types = predicted_types,
    confidence = confidence,
    scores = scores_matrix
  ))
}

# 执行细胞类型注释
cat("开始计算marker基因评分...\n")
annotation_results <- assign_cell_types(seurat_obj, B_cell_markers, min_score = 0.05)

# 将结果添加到metadata
seurat_obj@meta.data$CellType <- annotation_results$predicted_types
seurat_obj@meta.data$CellType_Confidence <- annotation_results$confidence

# 显示注释结果
cat("\n细胞类型注释完成!\n")
cat("识别出", length(unique(seurat_obj@meta.data$CellType)), "种细胞类型\n")
cat("细胞类型分布:\n")
print(table(seurat_obj@meta.data$CellType))

cat("\n注释置信度统计:\n")
print(summary(seurat_obj@meta.data$CellType_Confidence))

# 显示低置信度细胞的比例
low_confidence_cells <- sum(seurat_obj@meta.data$CellType_Confidence < 0.3)
cat("低置信度细胞数量 (置信度 < 0.3):", low_confidence_cells, "/", ncol(seurat_obj), 
    "(", round(low_confidence_cells/ncol(seurat_obj)*100, 2), "%)", "\n")

# 如果存在聚类信息，比较聚类和细胞类型注释的一致性
cluster_cols <- grep("RNA_snn_res|seurat_clusters", colnames(seurat_obj@meta.data), value = TRUE)
if (length(cluster_cols) > 0) {
  cluster_col <- cluster_cols[1]
  cat("\n比较聚类结果与细胞类型注释:\n")
  comparison_table <- table(seurat_obj@meta.data[[cluster_col]], seurat_obj@meta.data$CellType)
  print(comparison_table)
}

# 保存整合后的Seurat对象到指定路径
cat("\n保存整合后的数据到:", output_file, "\n")
saveRDS(seurat_obj, output_file)
cat("整合后的RDS文件已保存，BCR数据和UMAP坐标已整合到metadata中\n")

# 显示最终metadata的前几行
cat("\n整合后metadata预览:\n")
final_cols <- c(colnames(current_meta)[1:3], available_cols[1:3], "UMAP_1", "UMAP_2", "CellType")
final_cols <- intersect(final_cols, colnames(seurat_obj@meta.data))
print(head(seurat_obj@meta.data[, final_cols], 3))

cat("\n=== 数据整合和降维分析完成! ===\n")
cat("最终metadata包含", ncol(seurat_obj@meta.data), "个字段\n")
cat("包括BCR预测数据、UMAP坐标和细胞类型信息\n")