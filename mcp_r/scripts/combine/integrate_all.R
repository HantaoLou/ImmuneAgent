#!/usr/bin/env Rscript
# 一键整合BCR数据分析工具
# 功能：自动标准化CSV和RDS，并整合数据
# 作者: R代码专家
# 日期: 2024

# 加载必要的库
suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(readr)
  library(readxl)   # 用于读取Excel文件
  library(optparse)
})

###############################################################################
#                           工具函数定义                                      #
###############################################################################

#' 标准化CSV文件 - 添加combine_barcode字段
#' 
#' @param csv_data DataFrame对象
#' @param combine_fields 需要组合的字段名向量
#' @param separator 字段连接符，默认为"_"
#' @return 添加了combine_barcode字段的DataFrame
add_combine_barcode_to_csv <- function(csv_data, combine_fields, separator = "_") {
  
  # 验证字段是否存在
  available_fields <- colnames(csv_data)
  missing_fields <- setdiff(combine_fields, available_fields)
  
  if (length(missing_fields) > 0) {
    cat("错误: CSV文件中缺少以下字段:\n")
    print(missing_fields)
    cat("可用字段:\n")
    print(available_fields)
    stop("字段验证失败")
  }
  
  cat("CSV字段验证通过，将组合以下字段:", paste(combine_fields, collapse = ", "), "\n")
  
  # 提取需要组合的字段
  combine_data <- csv_data[, combine_fields, drop = FALSE]
  
  # 处理NA值，转换为"NA"字符串
  combine_data[] <- lapply(combine_data, function(x) {
    ifelse(is.na(x), "NA", as.character(x))
  })
  
  # 组合字段生成combine_barcode
  if (length(combine_fields) == 1) {
    csv_data$combine_barcode <- combine_data[[1]]
  } else {
    csv_data$combine_barcode <- apply(combine_data, 1, function(row) {
      paste(row, collapse = separator)
    })
  }
  
  # 统计信息
  total_rows <- nrow(csv_data)
  unique_barcodes <- length(unique(csv_data$combine_barcode))
  duplicate_rate <- (total_rows - unique_barcodes) / total_rows * 100
  
  cat("CSV combine_barcode生成完成\n")
  cat("  总行数:", total_rows, "\n")
  cat("  唯一barcode数:", unique_barcodes, "\n")
  cat("  重复率:", round(duplicate_rate, 2), "%\n")
  cat("  示例值:", head(csv_data$combine_barcode, 3), "\n")
  
  return(csv_data)
}


#' 标准化RDS文件 - 添加combine_barcode字段
#' 
#' @param seurat_obj Seurat对象
#' @param combine_fields 需要组合的字段名向量（支持"rownames"）
#' @param separator 字段连接符，默认为"_"
#' @return 添加了combine_barcode字段的Seurat对象
add_combine_barcode_to_rds <- function(seurat_obj, combine_fields, separator = "_") {
  
  # 获取metadata
  metadata <- seurat_obj@meta.data
  available_fields <- colnames(metadata)
  
  # 验证字段是否存在（排除rownames特殊字段）
  regular_fields <- combine_fields[combine_fields != "rownames"]
  missing_fields <- setdiff(regular_fields, available_fields)
  
  if (length(missing_fields) > 0) {
    cat("错误: RDS文件metadata中缺少以下字段:\n")
    print(missing_fields)
    cat("可用字段:\n")
    print(available_fields)
    stop("字段验证失败")
  }
  
  cat("RDS字段验证通过，将组合以下字段:", paste(combine_fields, collapse = ", "), "\n")
  
  # 提取需要组合的字段数据
  combine_data_list <- list()
  for (field in combine_fields) {
    if (field == "rownames") {
      combine_data_list[[field]] <- rownames(metadata)
    } else {
      combine_data_list[[field]] <- metadata[[field]]
    }
  }
  
  # 创建data.frame
  combine_data <- data.frame(combine_data_list, stringsAsFactors = FALSE)
  
  # 处理NA值
  combine_data[] <- lapply(combine_data, function(x) {
    ifelse(is.na(x), "NA", as.character(x))
  })
  
  # 组合字段
  if (length(combine_fields) == 1) {
    combine_barcode <- combine_data[[1]]
  } else {
    combine_barcode <- apply(combine_data, 1, function(row) {
      paste(row, collapse = separator)
    })
  }
  
  # 添加到metadata
  seurat_obj@meta.data$combine_barcode <- combine_barcode
  
  # 统计信息
  total_cells <- length(combine_barcode)
  unique_barcodes <- length(unique(combine_barcode))
  duplicate_rate <- (total_cells - unique_barcodes) / total_cells * 100
  
  cat("RDS combine_barcode生成完成\n")
  cat("  总细胞数:", total_cells, "\n")
  cat("  唯一barcode数:", unique_barcodes, "\n")
  cat("  重复率:", round(duplicate_rate, 2), "%\n")
  cat("  示例值:", head(combine_barcode, 3), "\n")
  
  return(seurat_obj)
}


#' 整合CSV和RDS数据
#' 
#' @param seurat_obj Seurat对象（必须包含combine_barcode字段）
#' @param csv_data CSV数据（必须包含combine_barcode字段）
#' @return 整合后的Seurat对象
integrate_data <- function(seurat_obj, csv_data) {
  
  # 验证combine_barcode字段存在
  if (!"combine_barcode" %in% colnames(seurat_obj@meta.data)) {
    stop("错误: Seurat对象metadata中未找到combine_barcode字段")
  }
  
  if (!"combine_barcode" %in% colnames(csv_data)) {
    stop("错误: CSV数据中未找到combine_barcode字段")
  }
  
  cat("\n=== 开始整合数据 ===\n")
  
  # 动态检测CSV文件中的字段
  csv_cols <- colnames(csv_data)
  
  # 提取各类预测字段
  variant_seq_cols <- grep("^variant_seq($|\\.|_)", csv_cols, value = TRUE)
  variant_name_cols <- grep("^variant_name($|\\.|_)", csv_cols, value = TRUE)
  bind_output_cols <- grep("^bind_output($|\\.|_)", csv_cols, value = TRUE)
  bind_predict_cols <- grep("^bind_predict($|\\.|_)", csv_cols, value = TRUE)
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
  
  cat("检测到的预测字段数量:\n")
  cat("  variant_seq:", length(variant_seq_cols), "\n")
  cat("  variant_name:", length(variant_name_cols), "\n")
  cat("  bind_output:", length(bind_output_cols), "\n")
  cat("  bind_predict:", length(bind_predict_cols), "\n")
  cat("  neu_output:", length(neu_output_cols), "\n")
  cat("  neu_predict:", length(neu_predict_cols), "\n")
  
  # 只选择存在的列进行合并
  available_cols <- intersect(merge_cols, csv_cols)
  cat("实际合并字段数:", length(available_cols), "\n")
  
  if (length(available_cols) == 0) {
    cat("警告: 未找到任何预测相关字段，将只使用Heavy和Light\n")
    available_cols <- intersect(c("Heavy", "Light"), csv_cols)
  }
  
  # 获取当前metadata
  current_meta <- seurat_obj@meta.data
  
  ###############################################################################
  #                    智能字段版本控制                                         #
  ###############################################################################
  
  cat("\n=== 智能字段处理策略 ===\n")
  
  # 定义保护字段（已存在则跳过，不追加）
  protected_fields <- c("Heavy", "Light")
  
  # 定义可追加字段（支持版本控制，允许多版本共存）
  # 这些是预测相关的字段，可能会多次运行得到不同版本的结果
  versioned_field_patterns <- c(
    "bind_output", "bind_predict",
    "neu_output", "neu_predict",
    "variant_seq", "variant_name"
  )
  
  # 分类处理字段
  fields_to_skip <- c()      # 需要跳过的字段
  fields_to_add <- c()       # 可以直接添加的字段（新字段）
  fields_to_version <- list() # 需要版本控制的字段（原名 -> 新名）
  
  for (field in available_cols) {
    field_exists <- field %in% colnames(current_meta)
    
    if (field %in% protected_fields) {
      # 保护字段逻辑
      if (field_exists) {
        cat("✓ [保护字段]", field, "已存在，跳过不覆盖\n")
        fields_to_skip <- c(fields_to_skip, field)
      } else {
        cat("✓ [保护字段]", field, "不存在，使用原名添加\n")
        fields_to_add <- c(fields_to_add, field)
      }
    } else {
      # 检查是否为可追加字段（预测相关字段）
      is_versioned <- any(sapply(versioned_field_patterns, function(pattern) {
        grepl(paste0("^", pattern, "($|\\.|_)"), field)
      }))
      
      if (is_versioned && field_exists) {
        # 可追加字段且已存在，需要生成新版本名称
        
        # 查找该字段的所有现有版本
        base_name <- field
        existing_versions <- grep(paste0("^", base_name, "(\\.[0-9]+)?$"), 
                                 colnames(current_meta), value = TRUE)
        
        # 提取版本号
        version_numbers <- sapply(existing_versions, function(v) {
          if (v == base_name) return(0)
          match <- regexec("\\.([0-9]+)$", v)
          if (match[[1]][1] == -1) return(0)
          as.integer(regmatches(v, match)[[1]][2])
        })
        
        # 生成新版本号
        next_version <- max(version_numbers) + 1
        new_field_name <- paste0(base_name, ".", next_version)
        
        cat("✓ [版本字段]", field, "已存在，将添加为", new_field_name, "\n")
        fields_to_version[[field]] <- new_field_name
        
      } else if (field_exists) {
        # 其他字段已存在，跳过
        cat("✓ [普通字段]", field, "已存在，跳过\n")
        fields_to_skip <- c(fields_to_skip, field)
      } else {
        # 字段不存在，直接添加
        cat("✓ [新字段]", field, "不存在，使用原名添加\n")
        fields_to_add <- c(fields_to_add, field)
      }
    }
  }
  
  # 统计信息
  cat("\n=== 字段处理汇总 ===\n")
  cat("跳过字段数:", length(fields_to_skip), "\n")
  if (length(fields_to_skip) > 0) {
    cat("  ", paste(fields_to_skip, collapse = ", "), "\n")
  }
  
  cat("直接添加字段数:", length(fields_to_add), "\n")
  if (length(fields_to_add) > 0) {
    cat("  ", paste(fields_to_add, collapse = ", "), "\n")
  }
  
  cat("版本控制字段数:", length(fields_to_version), "\n")
  if (length(fields_to_version) > 0) {
    for (old_name in names(fields_to_version)) {
      cat("  ", old_name, "->", fields_to_version[[old_name]], "\n")
    }
  }
  
  # 执行数据整合
  total_fields_to_merge <- length(fields_to_add) + length(fields_to_version)
  
  if (total_fields_to_merge > 0) {
    cat("\n执行数据整合...\n")
    
    # 准备合并数据
    fields_in_csv <- c(fields_to_add, names(fields_to_version))
    merge_data <- csv_data[, c("combine_barcode", fields_in_csv)]
    
    # 重命名需要版本控制的字段
    if (length(fields_to_version) > 0) {
      for (old_name in names(fields_to_version)) {
        new_name <- fields_to_version[[old_name]]
        colnames(merge_data)[colnames(merge_data) == old_name] <- new_name
      }
    }
    
    # 执行左连接
    integrated_meta <- current_meta %>%
      left_join(merge_data, by = c("combine_barcode" = "combine_barcode"))
    
    # 更新available_cols为实际添加的字段名（包括重命名后的）
    available_cols <- c(fields_to_add, unlist(fields_to_version))
    
  } else {
    cat("\n所有字段均已存在且为保护字段，跳过数据整合\n")
    integrated_meta <- current_meta
    available_cols <- c()
  }
  
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
  
  # 更新Seurat对象
  seurat_obj@meta.data <- integrated_meta
  
  # 统计匹配结果
  if (length(available_cols) > 0) {
    matched_cells <- sum(integrated_meta[[available_cols[1]]] != "NA")
    total_cells <- nrow(integrated_meta)
    match_rate <- matched_cells / total_cells * 100
  } else {
    matched_cells <- 0
    total_cells <- nrow(integrated_meta)
    match_rate <- 0
  }
  
  cat("\n整合完成!\n")
  cat("  原metadata列数:", ncol(current_meta), "\n")
  cat("  整合后列数:", ncol(integrated_meta), "\n")
  cat("  新增列数:", length(available_cols), "\n")
  cat("  成功匹配细胞数:", matched_cells, "/", total_cells, "\n")
  cat("  匹配率:", round(match_rate, 2), "%\n")
  
  return(seurat_obj)
}


#' 添加UMAP降维分析
#' 
#' @param seurat_obj Seurat对象
#' @param run_umap 是否执行UMAP分析
#' @return 添加了UMAP坐标的Seurat对象
add_umap_analysis <- function(seurat_obj, run_umap = TRUE) {
  
  if (!run_umap) {
    cat("跳过UMAP分析\n")
    return(seurat_obj)
  }
  
  cat("\n=== 开始UMAP降维分析 ===\n")
  
  # 检查是否已存在UMAP结果
  if ("umap" %in% names(seurat_obj@reductions)) {
    cat("检测到已存在UMAP降维结果，提取坐标...\n")
    
    umap_coords <- seurat_obj@reductions$umap@cell.embeddings
    seurat_obj@meta.data$UMAP_1 <- umap_coords[, 1]
    seurat_obj@meta.data$UMAP_2 <- umap_coords[, 2]
    
    cat("已从现有降维结果提取UMAP坐标\n")
    
  } else {
    cat("未检测到UMAP降维结果，开始执行降维分析...\n")
    
    # 检查是否需要PCA
    if (!"pca" %in% names(seurat_obj@reductions)) {
      cat("执行数据标准化和PCA...\n")
      
      # 寻找高变基因
      if (length(VariableFeatures(seurat_obj)) == 0) {
        seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)
        cat("已识别", length(VariableFeatures(seurat_obj)), "个高变基因\n")
      }
      
      # 标准化和缩放
      seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 10000)
      seurat_obj <- ScaleData(seurat_obj, features = VariableFeatures(seurat_obj))
      seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(seurat_obj), verbose = FALSE)
      cat("PCA分析完成\n")
    } else {
      cat("检测到已存在PCA结果，跳过PCA步骤\n")
    }
    
    # 确定PCA维度
    pca_dims <- min(20, ncol(seurat_obj@reductions$pca@cell.embeddings))
    cat("使用PCA维度: 1:", pca_dims, "\n")
    
    # 构建邻接图
    cat("构建邻接图...\n")
    seurat_obj <- FindNeighbors(seurat_obj, dims = 1:pca_dims, verbose = FALSE)
    
    # 进行基础聚类以便后续使用
    cat("进行聚类分析...\n")
    seurat_obj <- FindClusters(seurat_obj, resolution = 0.5, verbose = FALSE)
    cat("聚类完成\n")
    
    # 运行UMAP
    cat("运行UMAP降维...\n")
    seurat_obj <- RunUMAP(seurat_obj, dims = 1:pca_dims, verbose = FALSE)
    
    # 提取坐标
    umap_coords <- seurat_obj@reductions$umap@cell.embeddings
    seurat_obj@meta.data$UMAP_1 <- umap_coords[, 1]
    seurat_obj@meta.data$UMAP_2 <- umap_coords[, 2]
    
    cat("UMAP分析完成\n")
  }
  
  # 显示UMAP坐标统计
  cat("\nUMAP坐标统计:\n")
  cat("UMAP_1 范围:", round(min(seurat_obj@meta.data$UMAP_1), 3), "到", 
      round(max(seurat_obj@meta.data$UMAP_1), 3), "\n")
  cat("UMAP_2 范围:", round(min(seurat_obj@meta.data$UMAP_2), 3), "到", 
      round(max(seurat_obj@meta.data$UMAP_2), 3), "\n")
  
  cat("UMAP坐标已添加到metadata (UMAP_1, UMAP_2)\n")
  
  return(seurat_obj)
}


#' 基于marker基因的细胞类型注释
#' 
#' @param seurat_obj Seurat对象
#' @param run_annotation 是否执行细胞类型注释
#' @return 添加了细胞类型注释的Seurat对象
add_cell_type_annotation <- function(seurat_obj, run_annotation = TRUE) {
  
  if (!run_annotation) {
    cat("跳过细胞类型注释\n")
    return(seurat_obj)
  }
  
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
  
  return(seurat_obj)
}


###############################################################################
#                           主程序                                            #
###############################################################################

#' 主函数 - 一键整合BCR数据
main <- function() {
  
  # 命令行参数定义
  option_list <- list(
    make_option(c("--csv"), type="character", default=NULL,
                help="CSV或Excel文件路径 (.csv/.xlsx/.xls) [必需]", metavar="FILE"),
    make_option(c("--rds"), type="character", default=NULL,
                help="RDS文件路径 [必需]", metavar="FILE"),
    make_option(c("--output"), type="character", default=NULL,
                help="输出文件路径（.rds文件或目录） [必需]", metavar="FILE"),
    make_option(c("--csv-fields"), type="character", default=NULL,
                help="CSV组合字段，逗号分隔，如: Batch,barcode [可选，已有combine_barcode时跳过]", metavar="STRING"),
    make_option(c("--rds-fields"), type="character", default=NULL,
                help="RDS组合字段，逗号分隔，如: rownames,orig.ident [可选，已有combine_barcode时跳过]", metavar="STRING"),
    make_option(c("--separator"), type="character", default="_",
                help="字段连接符 [默认: _]", metavar="STRING"),
    make_option(c("--skip-umap"), action="store_true", default=FALSE,
                help="跳过UMAP降维分析 [默认: FALSE]"),
    make_option(c("--skip-annotation"), action="store_true", default=FALSE,
                help="跳过细胞类型注释 [默认: FALSE]"),
    make_option(c("--force-standardize"), action="store_true", default=FALSE,
                help="强制重新生成combine_barcode（即使已存在） [默认: FALSE]")
  )
  
  opt_parser <- OptionParser(option_list=option_list, 
                            description="一键整合BCR数据分析工具\n自动标准化CSV和RDS文件，并整合数据")
  opt <- parse_args(opt_parser)
  
  # 参数验证
  if (is.null(opt$csv) || is.null(opt$rds) || is.null(opt$output)) {
    print_help(opt_parser)
    stop("错误: 必须提供 --csv, --rds 和 --output 参数")
  }
  
  if (!file.exists(opt$csv)) {
    stop("错误: CSV文件不存在: ", opt$csv)
  }
  
  if (!file.exists(opt$rds)) {
    stop("错误: RDS文件不存在: ", opt$rds)
  }
  
  cat("\n")
  cat("===============================================\n")
  cat("  一键整合BCR数据分析工具\n")
  cat("===============================================\n")
  cat("输入文件:\n")
  cat("  CSV: ", opt$csv, "\n")
  cat("  RDS: ", opt$rds, "\n")
  cat("输出文件: ", opt$output, "\n")
  cat("分隔符: ", opt$separator, "\n")
  cat("===============================================\n\n")
  
  # 检查输入文件类型，如果是Excel则转换为CSV
  csv_file <- opt$csv
  file_ext <- tolower(tools::file_ext(csv_file))
  if (file_ext %in% c("xlsx", "xls")) {
    cat("[预处理] 检测到Excel文件，开始转换...\n")
    cat("Excel文件:", csv_file, "\n")
    
    # 读取Excel文件
    excel_data <- read_excel(csv_file)
    cat("Excel包含", nrow(excel_data), "行,", ncol(excel_data), "列\n")
    
    # 生成临时CSV文件名
    csv_temp_file <- sub("\\.(xlsx|xls)$", "_temp.csv", csv_file)
    
    # 写入CSV
    write.csv(excel_data, csv_temp_file, row.names = FALSE, fileEncoding = "UTF-8")
    cat("已转换为临时CSV:", csv_temp_file, "\n")
    
    # 更新csv_file指向临时CSV文件
    csv_file <- csv_temp_file
    cat("将使用转换后的CSV文件继续处理\n\n")
  }
  
  # 步骤1: 读取CSV文件
  cat("[步骤 1/5] 读取CSV/Excel数据...\n")
  cat("文件路径:", csv_file, "\n")
  tryCatch({
    csv_data <- read_csv(csv_file, locale = locale(encoding = "UTF-8"), show_col_types = FALSE)
    cat("数据读取成功:", nrow(csv_data), "行,", ncol(csv_data), "列\n")
  }, error = function(e) {
    cat("UTF-8编码读取失败，尝试默认编码...\n")
    csv_data <<- read.csv(csv_file, stringsAsFactors = FALSE)
    cat("数据读取成功:", nrow(csv_data), "行,", ncol(csv_data), "列\n")
  })
  
  # 步骤2: 读取RDS文件
  cat("\n[步骤 2/5] 读取RDS文件...\n")
  seurat_obj <- readRDS(opt$rds)
  if (!inherits(seurat_obj, "Seurat")) {
    stop("错误: RDS文件不是Seurat对象")
  }
  cat("Seurat对象读取成功:", ncol(seurat_obj), "个细胞\n")
  
  # 步骤3: 标准化CSV（如果需要）
  cat("\n[步骤 3/5] 标准化CSV数据...\n")
  if ("combine_barcode" %in% colnames(csv_data) && !opt$`force-standardize`) {
    cat("CSV文件已包含combine_barcode字段，跳过标准化\n")
  } else {
    if (is.null(opt$`csv-fields`)) {
      stop("错误: CSV文件缺少combine_barcode字段，必须提供 --csv-fields 参数")
    }
    csv_fields <- trimws(strsplit(opt$`csv-fields`, ",")[[1]])
    cat("将使用以下字段生成CSV的combine_barcode:", paste(csv_fields, collapse = ", "), "\n")
    csv_data <- add_combine_barcode_to_csv(csv_data, csv_fields, opt$separator)
  }
  
  # 步骤4: 标准化RDS（如果需要）
  cat("\n[步骤 4/5] 标准化RDS数据...\n")
  if ("combine_barcode" %in% colnames(seurat_obj@meta.data) && !opt$`force-standardize`) {
    cat("RDS文件已包含combine_barcode字段，跳过标准化\n")
  } else {
    if (is.null(opt$`rds-fields`)) {
      stop("错误: RDS文件缺少combine_barcode字段，必须提供 --rds-fields 参数")
    }
    rds_fields <- trimws(strsplit(opt$`rds-fields`, ",")[[1]])
    cat("将使用以下字段生成RDS的combine_barcode:", paste(rds_fields, collapse = ", "), "\n")
    seurat_obj <- add_combine_barcode_to_rds(seurat_obj, rds_fields, opt$separator)
  }
  
  # 步骤5: 整合数据
  cat("\n[步骤 5/5] 整合CSV和RDS数据...\n")
  seurat_obj <- integrate_data(seurat_obj, csv_data)
  
  # 可选: UMAP分析
  if (!opt$`skip-umap`) {
    seurat_obj <- add_umap_analysis(seurat_obj, run_umap = TRUE)
  }
  
  # 可选: 细胞类型注释
  if (!opt$`skip-annotation`) {
    seurat_obj <- add_cell_type_annotation(seurat_obj, run_annotation = TRUE)
  }
  
  # 保存结果
  cat("\n=== 保存结果 ===\n")
  
  # 处理输出路径
  if (dir.exists(opt$output)) {
    filename <- basename(opt$rds)
    output_file <- file.path(opt$output, filename)
  } else {
    output_file <- opt$output
  }
  
  cat("保存整合后的Seurat对象到:", output_file, "\n")
  saveRDS(seurat_obj, output_file)
  
  cat("\n")
  cat("===============================================\n")
  cat("  整合完成！\n")
  cat("===============================================\n")
  cat("输出文件:", output_file, "\n")
  cat("最终metadata包含", ncol(seurat_obj@meta.data), "个字段\n")
  cat("总细胞数:", ncol(seurat_obj), "\n")
  cat("===============================================\n\n")
  
  return(invisible(output_file))
}

# 执行主程序
if (!interactive()) {
  tryCatch({
    main()
  }, error = function(e) {
    cat("\n错误:", e$message, "\n")
    quit(status = 1)
  })
}
