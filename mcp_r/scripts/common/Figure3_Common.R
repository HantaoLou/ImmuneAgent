rm(list = ls())

################ Figure 3  ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure3.R <input_rds_file> <base_dir>")
}

input_rds_file <- args[1]
fdir <- args[2]

# 检查输入文件是否存在
if (!file.exists(input_rds_file)) {
  stop(paste("Input file does not exist:", input_rds_file))
}

# 检查基础目录是否存在
if (!dir.exists(fdir)) {
  stop(paste("Base directory does not exist:", fdir))
}

# 创建输出目录
output_dir <- file.path(fdir, "output", "Figure3")
plots_dir <- file.path(output_dir, "plots")
files_dir <- file.path(output_dir, "files")
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(files_dir, recursive = TRUE, showWarnings = FALSE)

## I. load packages
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

## II. load data
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

# 加载细胞数据
cell_obj <- readRDS(input_rds_file)


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

# 检测结合预测字段
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

## III. load color
my36colors <-c('#E5D2DD', '#53A85F', '#F1BB72', '#F3B1A0', '#D6E7A3', '#57C3F3', '#476D87',
               '#E95C59', '#E59CC4', '#AB3282', '#23452F', '#BD956A', '#8C549C', '#585658',
               '#9FA3A8', '#E0D4CA', '#5F3D69', '#C5DEBA', '#58A4C3', '#E4C755', '#F7F398',
               '#AA9A59', '#E63863', '#E39A35', '#C1E6F3', '#6778AE', '#91D0BE', '#B53E2B',
               '#712820', '#DCC1DD', '#CCE0F5',  '#CCC9E6', '#625D9E', '#68A180', '#3A6963',
               '#968175'
)

## IV. load marker
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

# 为了兼容性，保留原有的markers_a1变量名
markers_a1 <- B_cell_markers


###############################################################################
#'                          Manuscipt: figure3a                              '#
###############################################################################

## Figure 3A; 
## UMAP - 通用结合/中和预测密度图

# 通用化代码：自动检测可用的预测字段
# 使用统一的检测函数检测结合预测字段
bind_cols <- detect_all_binding_columns(cell_obj@meta.data)

# 检测中和预测字段（包含"neut"关键词）
neut_cols <- grep("neut", colnames(cell_obj@meta.data), 
                  value = TRUE, ignore.case = TRUE)

# 检测已计算的平均值字段
average_cols <- grep("bind.*average|average.*bind", colnames(cell_obj@meta.data), 
                     value = TRUE, ignore.case = TRUE)

# 合并所有可用的预测字段
available_prediction_cols <- c(bind_cols, neut_cols, average_cols)
available_prediction_cols <- unique(available_prediction_cols)

if (length(available_prediction_cols) > 0) {
  # 使用第一个可用的预测字段作为过滤条件
  filter_col <- available_prediction_cols[1]
  
  # 过滤掉NA值的细胞
  sub <- cell_obj[, !is.na(cell_obj@meta.data[[filter_col]])]
  
  # 选择所有可用的预测字段创建新数据矩阵
  available_cols_in_sub <- intersect(available_prediction_cols, colnames(sub@meta.data))
  
  if (length(available_cols_in_sub) > 0) {
    new_data <- sub@meta.data[, available_cols_in_sub, drop = FALSE]
    
    # 确保数据是数值类型
    new_data <- apply(new_data, 2, function(x) {
      x <- as.character(x)
      x[x == "NA" | is.na(x)] <- "0"
      as.numeric(x)
    })
    
    # 确保new_data是矩阵格式
    new_data <- as.matrix(new_data)
    new_data <- t(new_data)
    
    # 确保行名和列名都正确设置（处理点号和下划线替换为连字符）
    rownames(new_data) <- gsub("\\.", "-", rownames(new_data))
    rownames(new_data) <- gsub("_", "-", rownames(new_data))
    colnames(new_data) <- colnames(sub)  # 使用sub对象的细胞名称
    
    sub[['prediction']] <- CreateAssayObject(counts = as.matrix(new_data))
    
    # 创建UMAP reduction对象（使用metadata中的UMAP坐标）
    if("UMAP_1" %in% colnames(sub@meta.data) && "UMAP_2" %in% colnames(sub@meta.data)) {
      umap_coords <- sub@meta.data[, c("UMAP_1", "UMAP_2")]
      colnames(umap_coords) <- c("UMAP_1", "UMAP_2")
      sub[['umap']] <- CreateDimReducObject(embeddings = as.matrix(umap_coords), key = "UMAP_")
      cat("UMAP reduction object created successfully\n")
    } else {
      cat("Warning: UMAP coordinates not found in metadata\n")
    }
    
    DefaultAssay(sub) <- 'prediction'
    
    # 优先使用中和预测字段，如果没有则使用结合预测字段
    original_feature <- if (length(neut_cols) > 0 && neut_cols[1] %in% available_cols_in_sub) {
      neut_cols[1]
    } else {
      available_cols_in_sub[1]
    }
    
    # 获取prediction assay中的实际特征名称（处理点号和下划线替换为连字符）
    plot_feature <- gsub("\\.", "-", original_feature)
    plot_feature <- gsub("_", "-", plot_feature)
    
    # 确保特征名称在prediction assay中存在
    if (!plot_feature %in% rownames(sub[['prediction']])) {
      cat("Warning: Feature", plot_feature, "not found in prediction assay\n")
      cat("Available features:", paste(rownames(sub[['prediction']]), collapse = ", "), "\n")
      plot_feature <- rownames(sub[['prediction']])[1]  # 使用第一个可用特征
    }
    
    p <- ggrastr::rasterize(Nebulosa::plot_density(sub, 
                                                   plot_feature,   
                                                   size = 0.2, 
                                                   reduction = 'umap',
                                                   slot = 'data'), 
                            dpi = 300)
    
    # 保存图片
    ggsave(
      file.path(plots_dir, "Figure_3A.pdf"), 
      plot = p,
      width = 5,
      height = 4
    )
    
    cat("Figure 3A generated successfully using feature:", plot_feature, "\n")
    cat("Available prediction columns:", paste(available_cols_in_sub, collapse = ", "), "\n")
  } else {
    cat("Warning: No valid prediction columns found in the dataset\n")
  }
} else {
  cat("Warning: No prediction columns (bind/neut average values) found in the dataset\n")
}

###############################################################################
#'                          Manuscipt: figure3C                              '#
###############################################################################

## Figure 3C; 
## UMAP - B cells derived from flu BCells cohort

meta = cell_obj@meta.data
# 使用metadata中的UMAP坐标
if("UMAP_1" %in% colnames(meta) && "UMAP_2" %in% colnames(meta)) {
  meta$umap_1 <- meta$UMAP_1
  meta$umap_2 <- meta$UMAP_2
} else {
  cat("Warning: UMAP coordinates not found in metadata\n")
}

# 使用CellType作为分组变量
meta$CellType <- as.factor(meta$CellType)

p <- ggplot(meta, aes(x = umap_1, y = umap_2, color = CellType)) +
  geom_point(size = 0.15,
             shape = 16,
             stroke = 0) +
  theme_void()+
  scale_color_manual(values = my36colors, name = '')+
  theme(aspect.ratio = 1,
        legend.position = "")



ggsave(
  file.path(plots_dir, "Figure_3C.pdf"),plot = p,
  width = 2,
  height = 2,
  dpi = 300
)

###############################################################################
#'                          Manuscipt: figure3D                              '#
###############################################################################

## Figure 3D; 
## Dotplot representation of marker expression across different B cell subsets. 

# 清理CellType字段，去除重复的因子水平
cell_obj@meta.data$CellType <- as.character(cell_obj@meta.data$CellType)
# 获取唯一的细胞类型并重新创建因子
unique_celltypes <- unique(cell_obj@meta.data$CellType)
cell_obj@meta.data$CellType <- factor(cell_obj@meta.data$CellType, levels = unique_celltypes)

# 设置细胞类型为Idents
Idents(cell_obj) <- cell_obj@meta.data$CellType

# 将markers_a1列表转换为基因向量
all_markers <- unique(unlist(markers_a1))

# 检查哪些基因在数据中存在
available_genes <- rownames(cell_obj)
valid_markers <- all_markers[all_markers %in% available_genes]

if(length(valid_markers) > 0) {
  cat("使用的标记基因:", paste(valid_markers, collapse = ", "), "\n")
  
  p <- DotPlot(object = cell_obj, features = valid_markers, scale = T) + 
    scale_colour_gradientn(colors=brewer.pal(9, "YlGnBu")) + theme_bw() +
    theme(axis.text.x = element_text(angle = 90)) 
  
  ggsave(file.path(plots_dir, "Figure_3D.pdf"), plot = p, width = 18, height = 6)
  cat("Figure 3D 已成功生成\n")
} else {
  cat("警告：没有找到有效的标记基因，跳过Figure 3D生成\n")
}

###############################################################################
#'                          Manuscipt: figure3G                              '#
###############################################################################

## Figure 3G; 
## UMAP - 通用预测值密度图

# 检测可用的预测字段
# 使用统一的检测函数检测所有绑定预测字段
available_pred_cols <- detect_all_binding_columns(cell_obj@meta.data)

# 同时检测已计算的平均值字段
average_value_cols <- grep("average.*values|values.*average", colnames(cell_obj@meta.data), 
                           value = TRUE, ignore.case = TRUE)

# 合并所有可用的预测字段
available_pred_cols <- unique(c(available_pred_cols, average_value_cols))

if(length(available_pred_cols) > 0) {
  # 使用第一个可用的预测字段
  pred_col <- available_pred_cols[1]
  cat("使用预测字段:", pred_col, "\n")
  
  # 获取UMAP坐标和元数据
  meta_data <- cell_obj@meta.data
  
  # 确保预测列是数值型（处理字符型数据）
  if(pred_col %in% colnames(meta_data)) {
    if(is.character(meta_data[[pred_col]]) || is.factor(meta_data[[pred_col]])) {
      # 将字符型或因子型转换为数值型
      meta_data[[pred_col]] <- as.numeric(as.character(meta_data[[pred_col]]))
      cat("已将", pred_col, "从字符型转换为数值型\n")
    }
    # 处理NA值
    meta_data[[pred_col]][is.na(meta_data[[pred_col]])] <- 0
  }
  
  # 检查UMAP坐标是否存在
  if(all(c("UMAP_1", "UMAP_2") %in% colnames(meta_data))) {
    meta_data$highlight <- "highlight"
    meta_data$highlight[meta_data[[pred_col]] == 0] <- "normal"
    
    p <- ggplot(na.rm = TRUE) +
      geom_point(data = meta_data[meta_data$highlight == "normal",], 
                 aes(x = UMAP_1, y = UMAP_2, color = .data[[pred_col]]), size = 0.4) +
      geom_point(data = meta_data[meta_data$highlight == "highlight",], 
                 alpha = ifelse(is.na(meta_data[meta_data$highlight == "highlight",][[pred_col]]), 0, 1),
                 aes(x = UMAP_1, y = UMAP_2, color = .data[[pred_col]]), size = 0.4) +
      scale_color_gradientn(colors = c("transparent", "coral", "brown4"),
                            values = c(0, 0.5, 1),
                            breaks = c(0, 0.5, 1),
                            labels = c("0", "0.5", "1"),
                            name = "Prediction Score") +
      theme_classic(base_size = 10) +
      theme(
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8)
      )
    
    ggsave(file.path(plots_dir, "Figure_3G.pdf"), plot = p, width = 7, height = 6)
    cat("Figure 3G 已成功生成\n")
  } else {
    cat("警告：未找到UMAP坐标，跳过Figure 3G生成\n")
  }
} else {
  cat("警告：未找到预测字段，跳过Figure 3G生成\n")
}

###########################################################
# 记录运行环境
sink(file.path(plots_dir, "session_info.txt"))  # 将输出重定向到文件
sessionInfo()
sink()  # 关闭重定向
