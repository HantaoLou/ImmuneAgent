# Figure2 共享工具函数文件
# 包含所有模块需要的通用函数和数据处理逻辑

################ Figure2 Shared Utilities ###################

# King数据集CellType映射函数
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

# 检测结合预测列的函数（用于A2模块）
detect_binding_columns <- function(metadata, patterns = c("bind_predict\\.", "output\\.", "bind_output\\.")) {
  all_cols <- colnames(metadata)
  detected_cols <- c()
  
  # 直接检查列名，不使用正则表达式模式
  for (col_name in all_cols) {
    for (pattern in patterns) {
      if (grepl(pattern, col_name, ignore.case = TRUE)) {
        detected_cols <- c(detected_cols, col_name)
        break
      }
    }
  }
  
  return(unique(detected_cols))
}

# 处理结合数据的函数（用于A2模块）- 修复版本
process_binding_data <- function(metadata, binding_cols) {
  if (length(binding_cols) == 0) {
    warning("No binding prediction columns found")
    return(rep(0, nrow(metadata)))
  }
  
  # 处理嵌套数据框的情况
  bind_values <- c()
  
  for (col_name in binding_cols) {
    col_data <- metadata[[col_name]]
    
    # 如果是数据框，提取同名列
    if (is.data.frame(col_data)) {
      if (col_name %in% colnames(col_data)) {
        values <- col_data[[col_name]]
      } else {
        # 如果没有同名列，跳过这个列
        next
      }
    } else {
      # 如果是向量，直接使用
      values <- col_data
    }
    
    # 转换为数值
    values <- as.character(values)
    values[values == "NA" | is.na(values)] <- "0"
    values <- as.numeric(values)
    
    # 添加到结果中
    if (length(bind_values) == 0) {
      bind_values <- values
    } else {
      bind_values <- bind_values + values  # 累加多个预测值
    }
  }
  
  # 如果有多个列，计算平均值
  if (length(binding_cols) > 1) {
    bind_values <- bind_values / length(binding_cols)
  }
  
  return(bind_values)
}

# 创建结合预测图的函数（用于A2模块）
create_binding_prediction_plot <- function(metadata, title_prefix = "Binding Prediction") {
  p <- ggplot() +
    geom_point(data = metadata[metadata$highlight == "normal",], 
               aes(x = UMAP_1, y = UMAP_2, color = bind_average_values), 
               size = 0.8) +
    geom_point(data = metadata[metadata$highlight == "highlight",], 
               alpha = ifelse(is.na(metadata[metadata$highlight == "highlight",]$bind_average_values), 0, 1), 
               aes(x = UMAP_1, y = UMAP_2, color = bind_average_values), 
               size = 0.8) +
    scale_color_gradientn(colors = c("transparent", "coral", "brown4"),
                          values = c(0, 0.5, 1),
                          breaks = c(0, 0.5, 1),
                          labels = c("0", "0.5", "1"),
                          name = paste0(title_prefix, "\nscore")) +
    labs(title = paste("UMAP -", title_prefix),
         x = "UMAP_1", y = "UMAP_2") +
    theme_classic(base_size = 10) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14),
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 8)
    )
  return(p)
}

# 加载和预处理数据的通用函数
load_and_preprocess_data <- function(input_rds_file) {
  # 加载细胞数据
  cell_obj <- readRDS(input_rds_file)
  
  # 检查并应用King数据集映射
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
  } else {
    cat("Warning: No binding prediction columns found\n")
    cell_obj@meta.data["bind_average_values"] <- 0
  }
  
  return(cell_obj)
}

# 定义颜色方案
get_color_palette <- function() {
  my36colors <- c('#E5D2DD', '#53A85F', '#F1BB72', '#F3B1A0', '#D6E7A3', '#57C3F3', '#476D87',
                  '#E95C59', '#E59CC4', '#AB3282', '#23452F', '#BD956A', '#8C549C', '#585658',
                  '#9FA3A8', '#E0D4CA', '#5F3D69', '#C5DEBA', '#58A4C3', '#E4C755', '#F7F398',
                  '#AA9A59', '#E63863', '#E39A35', '#C1E6F3', '#6778AE', '#91D0BE', '#B53E2B',
                  '#712820', '#DCC1DD', '#CCE0F5',  '#CCC9E6', '#625D9E', '#68A180', '#3A6963',
                  '#968175')
  return(my36colors)
}

# 定义B细胞marker基因
get_bcell_markers <- function() {
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

# 初始化基础设置的函数
initialize_figure2_environment <- function(input_rds_file, base_dir, figure_name) {
  # 注意：不清理全局环境，避免删除命令行参数变量
  
  # 标准化路径分隔符（Windows兼容性修复）
  base_dir <- normalizePath(base_dir, winslash = "/", mustWork = FALSE)
  input_rds_file <- normalizePath(input_rds_file, winslash = "/", mustWork = TRUE)
  
  # 检查输入文件是否存在
  if (!file.exists(input_rds_file)) {
    stop(paste("Input file does not exist:", input_rds_file))
  }
  
  # 检查基础目录是否存在，不存在则创建
  if (!dir.exists(base_dir)) {
    cat(paste("Base directory does not exist, creating:", base_dir, "\n"))
    dir.create(base_dir, recursive = TRUE, showWarnings = FALSE)
    
    # 验证创建是否成功
    if (!dir.exists(base_dir)) {
      stop(paste("Failed to create base directory:", base_dir))
    }
    cat(paste("Base directory created successfully:", base_dir, "\n"))
  }
  
  # 创建输出目录（使用标准化路径）
  output_dir <- file.path(base_dir, "output", "Figure2")
  plots_dir <- file.path(output_dir, "plots")
  files_dir <- file.path(output_dir, "files")
  
  # 确保目录路径正确并创建
  output_dir <- normalizePath(output_dir, winslash = "/", mustWork = FALSE)
  plots_dir <- normalizePath(plots_dir, winslash = "/", mustWork = FALSE)
  files_dir <- normalizePath(files_dir, winslash = "/", mustWork = FALSE)
  
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(files_dir, recursive = TRUE, showWarnings = FALSE)
  
  # 验证目录创建成功
  if (!dir.exists(plots_dir)) {
    stop(paste("Failed to create plots directory:", plots_dir))
  }
  if (!dir.exists(files_dir)) {
    stop(paste("Failed to create files directory:", files_dir))
  }
  
  cat("输出目录创建成功:\n")
  cat("  plots_dir:", plots_dir, "\n")
  cat("  files_dir:", files_dir, "\n")
  
  # 加载必需的包
  library(Seurat)
  library(dplyr)
  library(ggplot2)
  library(cowplot)
  library(ggrepel)
  library(stringr)
  library(monocle3)
  library(RColorBrewer)
  library(Nebulosa)
  library(ggrastr)
  
  # 返回路径信息
  return(list(
    output_dir = output_dir,
    plots_dir = plots_dir,
    files_dir = files_dir
  ))
}

cat("Figure2 共享工具函数加载完成\n")