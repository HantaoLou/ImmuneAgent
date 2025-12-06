# Figure3 共享工具函数文件
# 包含所有模块需要的通用函数和数据处理逻辑

################ Figure3 Shared Utilities ###################

# King数据集CellType映射函数（需要在数据加载前定义）
king_celltype_mapping <- function(king_celltype) {
  mapping <- c(
    "Naive" = "Naive",
    "Activated" = "Activated", 
    "preGC" = "Germinal_Center",
    "LZ GC" = "Germinal_Center",
    "GC" = "Germinal_Center",
    "DZ GC" = "Germinal_Center",
    "FCRL2/3high GC" = "Germinal_Center",
    "prePB" = "Plasma",
    "Plasmablast" = "Plasma",
    "MBC" = "Memory",
    "MBC FCRL4+" = "Atypical",  # FCRL4+记忆B细胞映射为非典型B细胞
    "Cycling" = "Proliferating"
  )
  
  # 返回映射后的细胞类型，如果没有找到映射则返回原值
  mapped_type <- mapping[king_celltype]
  ifelse(is.na(mapped_type), king_celltype, mapped_type)
}

# 统一的绑定预测字段检测函数
detect_all_binding_columns <- function(metadata) {
  all_cols <- colnames(metadata)
  detected_cols <- c()
  
  # 定义所有可能的检测模式
  patterns <- c(
    "^bind_output$",           # 精确匹配 bind_output
    "^bind_predict$",          # 精确匹配 bind_predict
    "^bind_output\\.",          # bind_output.开头的列
    "^bind_predict\\.",         # bind_predict.开头的列
    "^output\\.[xy]$",          # output.x 或 output.y
    "^output\\.[0-9]+$",        # output.数字
    "bind_output\\.[0-9]+$",    # bind_output.数字
    "bind_predict\\.[0-9]+$"    # bind_predict.数字
  )
  
  for (pattern in patterns) {
    matching_cols <- grep(pattern, all_cols, value = TRUE)
    if (length(matching_cols) > 0) {
      detected_cols <- c(detected_cols, matching_cols)
      cat("Found columns matching pattern '", pattern, "':", paste(matching_cols, collapse = ", "), "\n")
    }
  }
  
  return(unique(detected_cols))
}

# 数据加载和预处理函数
load_and_preprocess_data <- function(input_rds_file) {
  # 检查输入文件是否存在
  if (!file.exists(input_rds_file)) {
    stop(paste("Input file does not exist:", input_rds_file))
  }
  
  # 加载细胞数据
  cell_obj <- readRDS(input_rds_file)
  
  # 检测King数据集并应用映射
  king_specific_types <- c("preGC", "LZ GC", "DZ GC", "FCRL2/3high GC", "prePB", "MBC FCRL4+")
  if(any(king_specific_types %in% unique(cell_obj@meta.data$CellType))) {
    cat("检测到King数据集，应用CellType映射...\n")
    # 保存原始CellType
    cell_obj@meta.data$Original_CellType <- cell_obj@meta.data$CellType
    # 应用映射
    cell_obj@meta.data$CellType <- sapply(cell_obj@meta.data$CellType, king_celltype_mapping)
    cat("CellType映射完成\n")
    
    # 显示映射结果统计
    cat("映射后的CellType分布:\n")
    print(table(cell_obj@meta.data$CellType))
  }
  
  # 检测并计算结合预测值
  selected_cols <- detect_all_binding_columns(cell_obj@meta.data)
  
  if(length(selected_cols) > 0) {
    cat("Total detected binding columns:", paste(selected_cols, collapse = ", "), "\n")
  } else {
    cat("Warning: No binding prediction columns found\n")
    cell_obj@meta.data["bind_average_values"] <- 0
  }
  
  if(length(selected_cols) > 0) {
    # 计算平均结合值
    bind_matrix <- cell_obj@meta.data[, selected_cols, drop = FALSE]
    bind_matrix <- apply(bind_matrix, 2, function(x) {
      x <- as.character(x)
      x[x == "NA" | is.na(x)] <- "0"
      as.numeric(x)
    })
    
    # 计算平均值
    if(length(selected_cols) == 1) {
      bind_avg <- bind_matrix[,1]
    } else {
      bind_avg <- rowMeans(bind_matrix, na.rm = TRUE)
    }
    
    # 正确赋值到meta.data
    cell_obj@meta.data["bind_average_values"] <- bind_avg
    
    cat("Calculated bind_average_values with range:", 
        paste(range(cell_obj@meta.data$bind_average_values, na.rm = TRUE), collapse = " to "), "\n")
  }
  
  return(cell_obj)
}

# 创建输出目录函数
create_output_directories <- function(base_dir) {
  # 检查基础目录是否存在
  if (!dir.exists(base_dir)) {
    stop(paste("Base directory does not exist:", base_dir))
  }
  
  # 创建输出目录
  output_dir <- file.path(base_dir, "output", "Figure3")
  plots_dir <- file.path(output_dir, "plots")
  files_dir <- file.path(output_dir, "files")
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(files_dir, recursive = TRUE, showWarnings = FALSE)
  
  return(list(
    output_dir = output_dir,
    plots_dir = plots_dir,
    files_dir = files_dir
  ))
}

# 颜色配置
get_color_palette <- function() {
  my36colors <- c('#E5D2DD', '#53A85F', '#F1BB72', '#F3B1A0', '#D6E7A3', '#57C3F3', '#476D87',
                  '#E95C59', '#E59CC4', '#AB3282', '#23452F', '#BD956A', '#8C549C', '#585658',
                  '#9FA3A8', '#E0D4CA', '#5F3D69', '#C5DEBA', '#58A4C3', '#E4C755', '#F7F398',
                  '#AA9A59', '#E63863', '#E39A35', '#C1E6F3', '#6778AE', '#91D0BE', '#B53E2B',
                  '#712820', '#DCC1DD', '#CCE0F5',  '#CCC9E6', '#625D9E', '#68A180', '#3A6963',
                  '#968175')
  return(my36colors)
}

# B细胞marker基因定义
get_bcell_markers <- function() {
  # 使用与integrate_bcr_data.R一致的B细胞marker基因定义
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
  return(B_cell_markers)
}

# 获取脚本目录的通用函数
get_script_dir <- function() {
  # 方法1：尝试使用commandArgs获取脚本路径
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- sub("--file=", "", file_arg[1])
    return(dirname(script_path))
  }
  
  # 方法2：尝试使用sys.frame（在某些环境下可用）
  tryCatch({
    script_path <- sys.frame(1)$ofile
    if (!is.null(script_path)) {
      return(dirname(script_path))
    }
  }, error = function(e) {})
  
  # 方法3：回退到当前工作目录
  return(getwd())
}

# 加载必需的R包
load_required_packages <- function() {
  required_packages <- c("Seurat", "dplyr", "ggplot2", "cowplot", "ggrepel", 
                        "stringr", "monocle3", "RColorBrewer", "Nebulosa", "ggrastr")
  
  for (pkg in required_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      stop(paste("Required package", pkg, "is not installed"))
    }
  }
  
  cat("所有必需的R包已成功加载\n")
}