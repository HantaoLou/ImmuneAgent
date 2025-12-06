# 标准化RDS文件工具
# 作者: R语言专家
# 功能: 将指定字段组合成combine_barcode字段并更新到RDS文件

library(Seurat)
library(dplyr)

#' 标准化RDS文件 - 组合字段生成combine_barcode
#' 
#' @param rds_file_path RDS文件路径
#' @param combine_fields 需要组合的字段名
#' @param output_path 输出文件路径，默认覆盖原文件
#' @param separator 字段连接符，默认为"_"
#' @return 更新后的Seurat对象
#' @export
standardize_rds <- function(rds_file_path, combine_fields, output_path = NULL, separator = "_") {
  
  # 参数验证
  if (!file.exists(rds_file_path)) {
    stop("错误: RDS文件不存在: ", rds_file_path)
  }
  
  if (length(combine_fields) == 0) {
    stop("错误: 必须提供至少一个字段名")
  }
  
  # 读取RDS文件
  cat("读取RDS文件:", rds_file_path, "\n")
  seurat_obj <- readRDS(rds_file_path)
  
  # 检查对象类型
  if (!inherits(seurat_obj, "Seurat")) {
    stop("错误: 文件不是Seurat对象")
  }
  
  cat("Seurat对象包含", ncol(seurat_obj), "个细胞\n")
  
  # 获取metadata
  metadata <- seurat_obj@meta.data
  available_fields <- colnames(metadata)
  
  cat("当前metadata包含字段:\n")
  print(available_fields)
  
  # 验证字段是否存在（排除rownames特殊字段）
  regular_fields <- combine_fields[combine_fields != "rownames"]
  missing_fields <- setdiff(regular_fields, available_fields)
  if (length(missing_fields) > 0) {
    cat("错误: 以下字段在RDS文件中不存在:\n")
    print(missing_fields)
    cat("可用字段:\n")
    print(available_fields)
    stop("字段验证失败")
  }
  
  cat("字段验证通过，将组合以下字段:\n")
  print(combine_fields)
  
  # 组合字段生成combine_barcode
  cat("开始生成combine_barcode字段...\n")
  
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
  
  # 处理NA值，将NA转换为"NA"字符串
  combine_data[] <- lapply(combine_data, function(x) {
    ifelse(is.na(x), "NA", as.character(x))
  })
  
  # 使用paste函数组合字段
  if (length(combine_fields) == 1) {
    combine_barcode <- combine_data[[1]]
  } else {
    combine_barcode <- apply(combine_data, 1, function(row) {
      paste(row, collapse = separator)
    })
  }
  
  # 添加combine_barcode字段到metadata
  seurat_obj@meta.data$combine_barcode <- combine_barcode
  
  cat("combine_barcode字段生成完成\n")
  cat("示例combine_barcode值:\n")
  print(head(combine_barcode, 5))
  
  # 统计信息
  unique_barcodes <- length(unique(combine_barcode))
  total_cells <- length(combine_barcode)
  cat("\n统计信息:\n")
  cat("总细胞数:", total_cells, "\n")
  cat("唯一combine_barcode数:", unique_barcodes, "\n")
  cat("重复率:", round((total_cells - unique_barcodes) / total_cells * 100, 2), "%\n")
  
  # 保存文件
  if (is.null(output_path)) {
    output_path <- rds_file_path  # 覆盖原文件
  } else {
    # 检查output_path是否为目录
    if (dir.exists(output_path)) {
      # 从原文件路径提取文件名
      filename <- basename(rds_file_path)
      output_path <- file.path(output_path, filename)
    }
  }
  
  cat("\n保存更新后的RDS文件到:", output_path, "\n")
  saveRDS(seurat_obj, output_path)
  
  cat("标准化完成!\n")
  cat("最终metadata包含", ncol(seurat_obj@meta.data), "个字段\n")
  
  return(output_path)
}

# 命令行参数处理
args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  # 从命令行获取参数
  input_file <- args[1]
  
  # 默认使用行名作为字段
  default_fields <- c("rownames")
  
  # 如果提供了第二个参数，解析为字段列表
  if (length(args) >= 2) {
    # 支持逗号分隔的字段列表
    fields_str <- args[2]
    combine_fields <- trimws(strsplit(fields_str, ",")[[1]])
  } else {
    combine_fields <- default_fields
  }
  
  # 如果提供了第三个参数，作为输出路径
  output_path <- NULL
  if (length(args) >= 3) {
    output_path <- args[3]
  }
  
  # 执行标准化处理
  cat("\n=== 开始执行RDS标准化处理 ===\n")
  cat("输入文件:", input_file, "\n")
  cat("组合字段:", paste(combine_fields, collapse = ", "), "\n")
  if (!is.null(output_path)) {
    cat("输出路径:", output_path, "\n")
  }
  
  tryCatch({
    result <- standardize_rds(input_file, combine_fields, output_path)
    cat("输出文件:", result, "\n")
    cat("\n=== 标准化处理完成 ===\n")
  }, error = function(e) {
    cat("错误:", e$message, "\n")
    quit(status = 1)
  })
  
}