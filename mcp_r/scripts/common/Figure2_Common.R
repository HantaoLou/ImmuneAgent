rm(list = ls())

################ Figure 2  ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure2.R <input_rds_file> <base_dir>")
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
output_dir <- file.path(fdir, "output", "Figure2")
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
#'                          Manuscipt: figureS2A                            '#
###############################################################################

## Figure S2A; 
## UMAP - The B cell subsets of PBMC derived from Priest, et al

# 检查数据结构
print(paste("Number of cells:", ncol(cell_obj)))
print(paste("Available metadata columns:", paste(colnames(cell_obj@meta.data), collapse=", ")))
print(paste("Unique cell types:", paste(unique(cell_obj@meta.data$CellType), collapse=", ")))

# 创建UMAP可视化图
# 由于fluBcells数据中UMAP坐标存储在metadata中，需要手动创建可视化
if("UMAP_1" %in% colnames(cell_obj@meta.data) && "UMAP_2" %in% colnames(cell_obj@meta.data)) {
  # 提取UMAP坐标和元数据
  umap_data <- data.frame(
    UMAP_1 = cell_obj@meta.data$UMAP_1,
    UMAP_2 = cell_obj@meta.data$UMAP_2,
    CellType = cell_obj@meta.data$CellType
  )
  
  # 生成UMAP图展示B细胞亚群
  p <- ggplot(umap_data, aes(x = UMAP_1, y = UMAP_2, color = CellType)) +
    geom_point(size = 0.5, alpha = 0.7) +
    scale_color_manual(values = my36colors[1:length(unique(umap_data$CellType))]) +
    labs(title = "UMAP - The B cell subsets from flu BCells data",
         x = "UMAP_1", y = "UMAP_2") +
    theme_classic() +
    theme(plot.title = element_text(hjust = 0.5, size = 14),
          legend.title = element_text(size = 12),
          legend.text = element_text(size = 10)) +
    guides(color = guide_legend(override.aes = list(size = 3, alpha = 1)))
  
  # 保存图片
  ggsave(file.path(plots_dir, "Figure_S2A_fluBcells_UMAP.pdf"), p, width = 12, height = 8, create.dir = TRUE)
  
  print("UMAP visualization saved successfully!")
  
} else {
  print("UMAP coordinates not found in metadata")
}

###############################################################################
#'                          Manuscipt: figure2A2                            '#
###############################################################################

## Figure 2A.2; 
## Prediction of binding breadth against flu variants (adapted from SARS-CoV-2)

# Function to detect binding prediction columns
detect_binding_columns <- function(metadata, patterns = c("bind_predict\\.", "output\\.", "bind_output\\.")) {
  all_cols <- colnames(metadata)
  detected_cols <- c()
  
  for (pattern in patterns) {
    matching_cols <- grep(pattern, all_cols, value = TRUE)
    if (length(matching_cols) > 0) {
      detected_cols <- c(detected_cols, matching_cols)
    }
  }
  
  return(unique(detected_cols))
}

# Function to process binding data
process_binding_data <- function(metadata, binding_cols) {
  if (length(binding_cols) == 0) {
    warning("No binding prediction columns found")
    return(rep(0, nrow(metadata)))
  }
  
  # Convert to numeric and handle NA values
  bind_matrix <- metadata[, binding_cols, drop = FALSE]
  bind_matrix <- apply(bind_matrix, 2, function(x) {
    x <- as.character(x)
    x[x == "NA" | is.na(x)] <- "0"
    as.numeric(x)
  })
  
  # Calculate average
  if (length(binding_cols) == 1) {
    return(bind_matrix[,1])
  } else {
    return(rowMeans(bind_matrix, na.rm = TRUE))
  }
}

# Clean duplicate column names in metadata
original_colnames <- colnames(cell_obj@meta.data)
if(any(duplicated(original_colnames))) {
  cat("Warning: Found duplicate column names, cleaning...\n")
  # Make column names unique
  colnames(cell_obj@meta.data) <- make.unique(colnames(cell_obj@meta.data))
  cat("Column names cleaned successfully\n")
}

# Detect and process binding prediction columns
binding_cols <- detect_binding_columns(cell_obj@meta.data)
cat("Detected binding columns:", paste(binding_cols, collapse = ", "), "\n")

if(length(binding_cols) > 0) {
  # 计算每个细胞的平均结合预测值
  cell_obj@meta.data$bind_average_values <- process_binding_data(cell_obj@meta.data, binding_cols)
  
  # 提取UMAP坐标和元数据
  meta_data <- cell_obj@meta.data
  meta_data$row_name <- rownames(meta_data)
  
  # 创建highlight分组
  meta_data$highlight <- "highlight"
  meta_data$highlight[meta_data$bind_average_values == 0] <- "normal"
  
  # Function to create binding prediction plot
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
  
  # 生成结合预测UMAP图
  p <- create_binding_prediction_plot(meta_data, "Prediction of binding breadth against flu variants")
  
  # 保存图片
  ggsave(file.path(plots_dir, "Figure_2A2_flu_bind_prediction.pdf"), p, width = 10, height = 8)
  
  print("Flu binding prediction UMAP saved successfully!")
  
  # 输出统计信息
  cat("\nBinding prediction statistics:\n")
  cat("Number of binding columns used:", length(binding_cols), "\n")
  if (length(binding_cols) > 0) {
    cat("Average binding score range:", round(min(meta_data$bind_average_values, na.rm = TRUE), 3), 
        "to", round(max(meta_data$bind_average_values, na.rm = TRUE), 3), "\n")
    cat("Number of cells with binding score > 0.5:", sum(meta_data$bind_average_values > 0.5, na.rm = TRUE), "\n")
    cat("Percentage of broadly reactive cells:", 
        round(100 * sum(meta_data$bind_average_values > 0.5, na.rm = TRUE) / nrow(meta_data), 2), "%\n")
  }
  
} else {
  print("No binding prediction columns found in metadata")
}

###############################################################################
#'                          Manuscipt: figure2B                             '#
###############################################################################

## Figure 2B.1; 
## Cell type distribution

if("CellType" %in% colnames(cell_obj@meta.data)) {
  # 手动创建DimPlot效果
  umap_data <- data.frame(
    UMAP_1 = cell_obj@meta.data$UMAP_1,
    UMAP_2 = cell_obj@meta.data$UMAP_2,
    CellType = cell_obj@meta.data$CellType
  )
  
  p <- ggplot(umap_data, aes(x = UMAP_1, y = UMAP_2, color = CellType)) +
    geom_point(size = 0.5, alpha = 0.7) +
    scale_color_manual(values = my36colors[1:length(unique(umap_data$CellType))]) +
    labs(title = "UMAP - B cell subtypes",
         x = "UMAP_1", y = "UMAP_2") +
    theme_classic() +
    theme(plot.title = element_text(hjust = 0.5, size = 14),
          legend.title = element_text(size = 12),
          legend.text = element_text(size = 10))
  
  ggsave(file.path(plots_dir, "Figure_2B1_celltype_distribution.pdf"), p, width = 10, height = 8)
  
  print("Cell type distribution UMAP saved successfully!")
}

###############################################################################
#'                          Manuscipt: figure2B2                            '#
###############################################################################

## Figure 2B.2; 
## The percentage broadly reactive B cells distribution by intervals

# 检查是否存在结合预测值和分组字段
if(exists("cell_obj") && "bind_average_values" %in% colnames(cell_obj@meta.data)) {
  
  # 提取 metadata
  metadata <- cell_obj@meta.data
  
  # 过滤掉结合预测值的空值
  metadata_filtered <- metadata %>%
    filter(!is.na(bind_average_values))
  
  # 将结合预测值分成 11 个等差的区间
  metadata_filtered <- metadata_filtered %>%
    mutate(bind_interval = cut(
      bind_average_values,
      breaks = seq(0, 1, by = 0.1),
      include.lowest = TRUE,
      right = FALSE
    ))
  
  # 检查是否存在分组字段（如Status），如果不存在则使用CellType作为主要分组
  group_cols <- c("CellType")
  if("Status" %in% colnames(metadata_filtered)) {
    group_cols <- c("Status", "CellType")
  }
  
  # 按分组统计每个区间的比例
  if(length(group_cols) == 2) {
    # 有Status字段的情况
    bind_summary <- metadata_filtered %>%
      group_by(Status, CellType, bind_interval) %>%
      summarise(
        cell_count = n(),
        .groups = "drop"
      ) %>%
      group_by(Status, CellType) %>%
      mutate(proportion = cell_count / sum(cell_count))
    
    # 绘制分面堆叠柱状图
    p <- ggplot(bind_summary, aes(x = CellType, y = proportion, fill = bind_interval)) +
      geom_bar(stat = "identity", position = "stack") +
      facet_wrap(~ Status) +
      labs(
        title = "Proportion of binding prediction values in each interval",
        x = "CellType",
        y = "Proportion",
        fill = "Bind Interval"
      ) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
  } else {
    # 只有CellType字段的情况
    bind_summary <- metadata_filtered %>%
      group_by(CellType, bind_interval) %>%
      summarise(
        cell_count = n(),
        .groups = "drop"
      ) %>%
      group_by(CellType) %>%
      mutate(proportion = cell_count / sum(cell_count))
    
    # 绘制简单堆叠柱状图
    p <- ggplot(bind_summary, aes(x = CellType, y = proportion, fill = bind_interval)) +
      geom_bar(stat = "identity", position = "stack") +
      labs(
        title = "Proportion of binding prediction values in each interval",
        x = "CellType",
        y = "Proportion",
        fill = "Bind Interval"
      ) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  }
  
  # 保存图片
  ggsave(
    file.path(plots_dir, "Figure_2B2_binding_intervals.pdf"),
    plot = p,
    width = 8,
    height = 6
  )
  
  print("Binding intervals distribution plot saved successfully!")
  
  # 输出统计信息
  cat("\nBinding intervals statistics:\n")
  cat("Total cells with binding data:", nrow(metadata_filtered), "\n")
  cat("Number of cell types:", length(unique(metadata_filtered$CellType)), "\n")
  if("Status" %in% colnames(metadata_filtered)) {
    cat("Number of status groups:", length(unique(metadata_filtered$Status)), "\n")
  }
  
} else {
  print("Binding intervals plot skipped: no binding prediction values found")
}

###############################################################################
#'                    Manuscipt: figureS2C                                   '#
###############################################################################

## Figure S2C; 
## Dot plot of marker genes relating to the cell types

# 检查是否存在基因表达数据和标记基因
if(exists("cell_obj") && exists("markers_a1")) {
  
  # 过滤掉不存在的基因
  # 先将列表转换为向量，然后找出存在的基因
  all_markers <- unlist(markers_a1)
  available_markers <- intersect(all_markers, rownames(cell_obj))
  
  if(length(available_markers) > 0) {
    # 生成点图
    p <- DotPlot(object = cell_obj, features = available_markers, scale = T) + 
      scale_colour_gradientn(colors=brewer.pal(9, "YlGnBu")) + theme_bw() +
      theme(axis.text.x = element_text(angle = 90)) 
    
    ggsave(file.path(plots_dir, "Figure_S2C_marker_genes_dotplot.pdf"), p, width=16, height=8)
    
    print("Marker genes dot plot saved successfully!")
    
    # 输出统计信息
    cat("\nMarker genes dot plot statistics:\n")
    cat("Total marker genes defined:", length(all_markers), "\n")
    cat("Available marker genes in data:", length(available_markers), "\n")
    cat("Missing marker genes:", length(all_markers) - length(available_markers), "\n")
    
    # 输出缺失的基因
    missing_markers <- setdiff(all_markers, rownames(cell_obj))
    if(length(missing_markers) > 0) {
      cat("Missing genes:", paste(missing_markers, collapse=", "), "\n")
    }
    
  } else {
    print("No marker genes found in the dataset")
  }
  
} else {
  print("Marker genes dot plot skipped: cell_obj or markers_a1 not found")
}


###############################################################################
#'                          Manuscipt: figure2C                              '#
###############################################################################

## Figure 2C;  ??? 
## Volcano plot of the differentially expressed genes between broadly reactive AtM B cells and the non-reactive B cells 

# setting 15 ：跟≥5个抗原结合的ITGAX+AtM 细胞 vs 跟=1个抗原结合的MB细胞做DEG
# setting 11 ：跟≥5个抗原结合的ITGAX+AtM 细胞 vs 跟=1个抗原结合的ALL细胞做DEG

# 去除merge不到BCR的细胞
sub <- cell_obj[,!is.na(cell_obj$bind_average_values)]

# 通用的数据分布判断方法
# 自动根据bind_average_values的分布设置阈值
bind_values <- sub$bind_average_values[!is.na(sub$bind_average_values)]
unique_values <- sort(unique(bind_values))

# 方法1：基于分位数的动态阈值设置
q33 <- quantile(bind_values, 0.33, na.rm = TRUE)
q67 <- quantile(bind_values, 0.67, na.rm = TRUE)

# 方法2：基于实际数据分布的智能阈值
if(length(unique_values) <= 15) {
  # 离散值较少时，基于实际值设置
  low_threshold <- unique_values[ceiling(length(unique_values) * 0.33)]
  high_threshold <- unique_values[ceiling(length(unique_values) * 0.67)]
} else {
  # 连续值较多时，使用分位数
  low_threshold <- q33
  high_threshold <- q67
}

cat("Selected thresholds - Low:", low_threshold, ", High:", high_threshold, "\n")

# 应用阈值进行分类
sub$bind_level <- ifelse(sub$bind_average_values >= high_threshold, 2, 0)
sub$bind_level[sub$bind_average_values >= low_threshold & sub$bind_average_values < high_threshold] <- 1

# 输出分类结果统计
cat("Bind level distribution:\n")
cat("Level 0 (Low):", sum(sub$bind_level == 0), "cells\n")
cat("Level 1 (Medium):", sum(sub$bind_level == 1), "cells\n")
cat("Level 2 (High):", sum(sub$bind_level == 2), "cells\n")

# 保存阈值信息用于文件命名
i <- high_threshold

# 
j = 11
sub$DEG_level <- 2
# 使用Atypical细胞类型（对应原来的ITGAX+AtM）
sub$DEG_level[(sub$CellType %in% c('Atypical'))
              & (sub$bind_level == 2)] <- 1
sub$DEG_level[sub$bind_level == 1] <- 0
Idents(sub) <- 'DEG_level'
# 清理Seurat对象的命令历史以避免参数冲突
sub@commands <- list()

# 验证身份组是否存在且有足够细胞
available_idents <- levels(Idents(sub))
if ("0" %in% available_idents && "1" %in% available_idents) {
  cells_0 <- sum(Idents(sub) == "0")
  cells_1 <- sum(Idents(sub) == "1")
  
  if (cells_0 >= 3 && cells_1 >= 3) {
    markers <- FindMarkers(sub, ident.1 = "1", ident.2 = "0",
                           logfc.threshold=0, min.pct=0.2)
    
    # 处理markers结果
    if (exists("markers")) {
      markers$p_val_adj = p.adjust(markers$p_val, method='fdr')
      markers$pct_dif <- markers$pct.1 - markers$pct.2
      write.csv(markers,file.path(files_dir,paste0("20250228-B-bind_level_New-cluster_marker_threshold=",i,"_setting=",j,".csv")))
    }
  } else {
    cat("跳过分析 j=", j, "：组中细胞数量不足 (组0:", cells_0, "个细胞, 组1:", cells_1, "个细胞)\n")
  }
} else {
  cat("跳过分析 j=", j, "：缺少必要的身份组\n")
}


# 
j = 15
sub$DEG_level <- 2
# 使用Atypical细胞类型（对应原来的ITGAX+AtM）
sub$DEG_level[(sub$CellType %in% c('Atypical'))
              & (sub$bind_level == 2)] <- 1
# 使用Memory和Atypical细胞类型作为对照组
sub$DEG_level[(sub$CellType %in% c('Memory', 'Atypical'))
              & (sub$bind_level == 1)] <- 0
Idents(sub) <- 'DEG_level'
# 清理Seurat对象的命令历史以避免参数冲突
sub@commands <- list()

# 验证身份组是否存在且有足够细胞
available_idents <- levels(Idents(sub))
if ("0" %in% available_idents && "1" %in% available_idents) {
  cells_0 <- sum(Idents(sub) == "0")
  cells_1 <- sum(Idents(sub) == "1")
  
  if (cells_0 >= 3 && cells_1 >= 3) {
    markers <- FindMarkers(sub, ident.1 = "1", ident.2 = "0",
                           logfc.threshold=0, min.pct=0.2)
    
    # 处理markers结果
    if (exists("markers")) {
      markers$p_val_adj = p.adjust(markers$p_val, method='fdr')
      markers$pct_dif <- markers$pct.1 - markers$pct.2
      write.csv(markers,file.path(files_dir,paste0("20250228-B-bind_level_New-cluster_marker_threshold=",i,"_setting=",j,".csv")))
    }
  } else {
    cat("跳过分析 j=", j, "：组中细胞数量不足 (组0:", cells_0, "个细胞, 组1:", cells_1, "个细胞)\n")
  }
} else {
  cat("跳过分析 j=", j, "：缺少必要的身份组\n")
}

# 只有当markers存在时才进行后续分析和绘图
if (exists("markers") && !is.null(markers) && nrow(markers) > 0) {
  # 添加 -log10(p_val_adj) 列
  markers$log10pvalue <- -log10(markers$p_val_adj)
  # 标记显著性
  markers$significance <- ifelse(markers$p_val_adj < 0.05 & abs(markers$avg_log2FC) > 1, 
                                 ifelse(markers$avg_log2FC > 1, "Up", "Down"), "Not Significant")
  markers$pct_dif <- markers$pct.1 - markers$pct.2
  x1 = max(abs(markers$avg_log2FC))
  x2 = -x1
  # 绘制火山图
  p <- ggplot(markers, aes(x = avg_log2FC, y = log10pvalue, color = significance)) +
    geom_point(alpha = 0.8, size = 1) +
    scale_color_manual(values = c("Up" = "red", "Down" = "blue", "Not Significant" = "grey")) +
    geom_vline(xintercept = c(-1, 1), linetype = "dashed") +
    #geom_hline(yintercept = -log10(0.05), linetype = "dashed") +
    labs(title = "Volcano Plot", x = "log2 Fold Change", y = "-log10 p-value") +
    theme_minimal()+
    theme(aspect.ratio = 1,
          text = element_text(size = 18),
          panel.grid.major = element_blank(),  # 去掉主要网格线
          panel.grid.minor = element_blank(),  # 去掉次要网格线
          axis.line = element_line(colour = "black")  # 显示坐标轴线
          
          # legend.position = "",
          # axis.title.x = element_blank(),  # 不显示 x 轴标题
          # axis.title.y = element_blank()   # 不显示 y 轴标题
    )+
    xlim(x2,x1)+
    ylim(0,max(markers$log10pvalue)+3)
  
  # 添加基因名标记
  p <- p + geom_text_repel(aes(label = ifelse(significance != "Not Significant", rownames(markers), "")), 
                           size = 3, box.padding = 0.3,max.overlaps=20)
  
  ggsave(file.path(plots_dir,"Figure_2C.pdf"), p, width=8,height=6,dpi = 300)
  cat("火山图已保存到:", file.path(plots_dir,"Figure_2C.pdf"), "\n")
} else {
  cat("没有有效的markers数据，跳过火山图绘制\n")
}
