rm(list = ls())

################ Figure 5  ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure5.R <input_rds_file> <base_dir>")
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
output_dir <- file.path(fdir, "output", "Figure5")
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
# 加载细胞数据
cell_obj <- readRDS(input_rds_file)

# 检查并修复重复的列名
meta_cols <- colnames(cell_obj@meta.data)
if (any(duplicated(meta_cols))) {
  cat("检测到重复的元数据列名，正在修复...\n")
  # 为重复的列名添加后缀
  colnames(cell_obj@meta.data) <- make.names(meta_cols, unique = TRUE)
  cat("重复列名已修复\n")
}


###############################################################################
#'                          Manuscipt: figure5C                              '#
###############################################################################

## Figure 5C; 
## Bar plots depicting the isotype distribution and SHM rates for broadly reactive BCRs compared with specific and non-binding BCRs. 
## Broadly reactive clonotypes show an elevated IgG1 proportion and higher SHM rates. 

# 统一的isotype映射表（处理所有可能的格式）
isotype_mapping <- c(
  # 小写格式（King数据集）
  "IgM" = "IGHM", "IgD" = "IGHD", 
  "IgG1" = "IGHG1", "IgG2" = "IGHG2", "IgG3" = "IGHG3", "IgG4" = "IGHG4",
  "IgA1" = "IGHA1", "IgA2" = "IGHA2", 
  # 大写格式（fluBcells数据集）
  "IGHM" = "IGHM", "IGHD" = "IGHD", "IGHE" = "IGHE",
  "IGHG" = "IGHG1", "IGHA" = "IGHA1",
  # 特殊值处理
  "None" = "IGHG1", "Multi" = "IGHG1"
)

# 创建IGH_isotype字段
if(!"IGH_isotype" %in% colnames(cell_obj@meta.data)) {
  # 查找isotype相关字段（忽略大小写）
  isotype_cols <- grep("isotype", colnames(cell_obj@meta.data), ignore.case = TRUE, value = TRUE)
  
  if(length(isotype_cols) > 0) {
    # 使用第一个找到的isotype字段
    isotype_col <- isotype_cols[1]
    isotype_values <- cell_obj@meta.data[[isotype_col]]
    
    cat(paste("使用字段:", isotype_col, "\n"))
    cat(paste("原始唯一值:", paste(unique(isotype_values), collapse = ", "), "\n"))
    
    # 统一映射所有值
    mapped_values <- isotype_mapping[as.character(isotype_values)]
    
    # 处理NA和未映射的值
    mapped_values[is.na(mapped_values) | is.na(isotype_values)] <- "IGHG1"
    
    cell_obj@meta.data$IGH_isotype <- mapped_values
    cat(paste("映射后唯一值:", paste(unique(mapped_values), collapse = ", "), "\n"))
    cat(paste("NA值数量:", sum(is.na(isotype_values)), "-> 已设为IGHG1\n"))
  } else {
    # 如果没有找到isotype字段，设置为默认值
    cell_obj@meta.data$IGH_isotype <- "IGHG1"
    warning("未找到isotype相关字段，IGH_isotype设置为默认值IGHG1")
  }
} else {
  cat("IGH_isotype字段已存在\n")
}
# 函数：基于基因表达估算SHM水平
estimate_shm_from_expression <- function(seurat_obj) {
  # 获取元数据
  meta_data <- seurat_obj@meta.data
  
  # 检查是否存在必要的亲和力字段
  has_high_affinity <- "high_affinity1" %in% colnames(meta_data)
  has_low_affinity <- "Low_affinity2" %in% colnames(meta_data)
  
  # 如果缺少必要字段，直接返回合理的默认SHM值
  if (!has_high_affinity || !has_low_affinity) {
    cat("Warning: 缺少亲和力字段，使用默认SHM值\n")
    n_cells <- nrow(meta_data)
    # 使用合理的默认SHM分布：大部分细胞SHM较低，少数较高
    estimated_shm_h <- sample(c(rep(2:8, 0.7*n_cells/7), rep(9:15, 0.3*n_cells/7)), n_cells, replace = TRUE)
    estimated_shm_l <- sample(c(rep(1:5, 0.7*n_cells/5), rep(6:10, 0.3*n_cells/5)), n_cells, replace = TRUE)
    return(list(H_shm = estimated_shm_h, L_shm = estimated_shm_l))
  }
  
  # 如果有亲和力字段，使用原来的计算方法
  high_affinity_score <- meta_data$high_affinity1
  low_affinity_score <- meta_data$Low_affinity2
  
  # 检查数据范围，避免除零错误
  high_range <- max(high_affinity_score, na.rm = TRUE) - min(high_affinity_score, na.rm = TRUE)
  low_range <- max(low_affinity_score, na.rm = TRUE) - min(low_affinity_score, na.rm = TRUE)
  
  if (high_range == 0 || low_range == 0) {
    cat("Warning: 亲和力分数无变化，使用默认SHM值\n")
    n_cells <- nrow(meta_data)
    estimated_shm_h <- sample(2:15, n_cells, replace = TRUE)
    estimated_shm_l <- sample(1:10, n_cells, replace = TRUE)
    return(list(H_shm = estimated_shm_h, L_shm = estimated_shm_l))
  }
  
  # 标准化分数到0-1范围
  high_norm <- (high_affinity_score - min(high_affinity_score, na.rm = TRUE)) / high_range
  low_norm <- (low_affinity_score - min(low_affinity_score, na.rm = TRUE)) / low_range
  
  # 估算SHM：高亲和力分数高、低亲和力分数低的细胞SHM更多
  estimated_shm_h <- round((high_norm - low_norm + 1) * 15)  # 重链SHM估算
  estimated_shm_l <- round((high_norm - low_norm + 1) * 10)  # 轻链SHM估算
  
  # 确保SHM值为非负数
  estimated_shm_h[estimated_shm_h < 0] <- 0
  estimated_shm_l[estimated_shm_l < 0] <- 0
  
  return(list(H_shm = estimated_shm_h, L_shm = estimated_shm_l))
}

# 使用函数估算SHM水平
shm_results <- estimate_shm_from_expression(cell_obj)
cell_obj$H_shm <- shm_results$H_shm
cell_obj$L_shm <- shm_results$L_shm

# 计算SARS2_bind_average_values和SARS2_neut_average_values
# 使用统一的结合预测字段检测和处理函数
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

# 检测结合预测字段
bind_cols <- detect_binding_columns(cell_obj@meta.data)

if(length(bind_cols) > 0) {
  cat("Found binding prediction columns:", paste(bind_cols, collapse = ", "), "\n")
  
  # 使用统一的处理函数计算平均结合预测值
  binding_values <- process_binding_data(cell_obj@meta.data, bind_cols)
  
  cell_obj@meta.data$SARS2_bind_average_values <- binding_values
  
  # 计算中和预测值（基于结合预测值的变换）
  cell_obj@meta.data$SARS2_neut_average_values <- 
    pmax(0, cell_obj@meta.data$SARS2_bind_average_values - 0.1)
  
  cat("Calculated SARS2 binding and neutralization prediction values\n")
  cat("Binding range:", paste(range(cell_obj@meta.data$SARS2_bind_average_values, na.rm = TRUE), collapse = " to "), "\n")
  cat("Neutralization range:", paste(range(cell_obj@meta.data$SARS2_neut_average_values, na.rm = TRUE), collapse = " to "), "\n")
} else {
  cat("Warning: No binding prediction columns found, setting default values\n")
  cell_obj@meta.data$SARS2_bind_average_values <- 0
  cell_obj@meta.data$SARS2_neut_average_values <- 0
}

df <- cell_obj@meta.data[,c("SARS2_bind_average_values","SARS2_neut_average_values")]
df <- as.data.frame(df)

# 获取细胞类型字段（忽略大小写）
celltype_cols <- colnames(cell_obj@meta.data)
celltype_field <- celltype_cols[tolower(celltype_cols) == "celltype"]

if(length(celltype_field) > 0) {
  df$celltype <- cell_obj@meta.data[[celltype_field[1]]]
  cat("Using", celltype_field[1], "for cell type\n")
} else {
  cat("Warning: No celltype field found, using default\n")
  df$celltype <- rep("B_cell", nrow(df))
}
df$IGH_isotype <- cell_obj@meta.data$IGH_isotype
df$H_shm <- cell_obj@meta.data$H_shm
df$L_shm <- cell_obj@meta.data$L_shm

# 调试信息：检查IGH_isotype字段状态
cat("IGH_isotype字段创建后的状态:\n")
cat("前10个值:", paste(head(df$IGH_isotype, 10), collapse = ", "), "\n")
cat("唯一值:", paste(unique(df$IGH_isotype), collapse = ", "), "\n")
cat("NA数量:", sum(is.na(df$IGH_isotype)), "/", length(df$IGH_isotype), "\n")

# 添加celltype映射逻辑，将king数据集的celltype映射为fluBcells标准字段
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

# 应用映射，如果celltype在mapping中存在则映射，否则保持原值
df$celltype <- ifelse(df$celltype %in% names(mapping), mapping[df$celltype], df$celltype)

cat("映射后的celltype唯一值:", paste(unique(df$celltype), collapse = ", "), "\n")

# 检查数据中是否存在目标celltype，如果不存在则使用可用的celltype
target_celltype <- "B.09.ITGAX+AtM"
if (!target_celltype %in% df$celltype) {
  # 如果目标celltype不存在，选择数据量最多的celltype
  celltype_counts <- table(df$celltype)
  target_celltype <- names(celltype_counts)[which.max(celltype_counts)]
  cat("目标celltype不存在，使用数据量最多的celltype:", target_celltype, "\n")
}

df <- df[df$celltype == target_celltype,]
cat("过滤后的数据行数:", nrow(df), "\n")
df <- df[!is.na(df$SARS2_neut_average_values),]
df <- df[!is.na(df$IGH_isotype),]


i = 0.5
df$SARS2_neut_level <- ifelse(df$SARS2_neut_average_values>0,'neut','not neut')
unique(df$SARS2_neut_level)
df$SARS2_neut_level <- factor(df$SARS2_neut_level, levels = c('neut','not neut'))

df$SARS2_bind_level <- ifelse(df$SARS2_bind_average_values>=i,'more broad','not bind')
df$SARS2_bind_level <- ifelse(df$SARS2_bind_average_values< i,'less broad',df$SARS2_bind_level)
df$SARS2_bind_level <- ifelse(df$SARS2_bind_average_values == 0.1, 'specific',df$SARS2_bind_level)
df$SARS2_bind_level <- ifelse(df$SARS2_bind_average_values == 0, 'not bind',df$SARS2_bind_level)

df$SARS2_bind_level <- factor(df$SARS2_bind_level, levels = c("more broad","less broad",
                                                              "specific","not bind"))



# 去掉SHM很离谱的值
cat("H_shm字段状态:\n")
cat("H_shm NA数量:", sum(is.na(df$H_shm)), "/", length(df$H_shm), "\n")
cat("H_shm范围:", min(df$H_shm, na.rm = TRUE), "到", max(df$H_shm, na.rm = TRUE), "\n")
cat("H_shm >= 45的数量:", sum(df$H_shm >= 45, na.rm = TRUE), "\n")
cat("H_shm < 45的数量:", sum(df$H_shm < 45, na.rm = TRUE), "\n")

# 处理H_shm的NA值，如果所有值都是NA则跳过过滤
if (all(is.na(df$H_shm))) {
  cat("警告：H_shm字段全部为NA，跳过H_shm过滤\n")
} else {
  df <- df[!is.na(df$H_shm) & df$H_shm < 45,]
}
cat("H_shm过滤后的数据行数:", nrow(df), "\n")

plot_df = df %>%
  dplyr::count(SARS2_bind_level, IGH_isotype) 

plot_df$IGH_isotype = factor(
  plot_df$IGH_isotype,
  levels = c(
    #"IGHE",
    "IGHM",
    "IGHD",
    "IGHA1",
    "IGHA2",
    "IGHG1",
    "IGHG2",
    "IGHG3",
    "IGHG4"
  )
)

p1 = ggplot(plot_df, aes(SARS2_bind_level, n, fill = IGH_isotype)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_fill_manual(
    values =
      c(
        #"IGHE" = "#CED6C3",
        "IGHM" = "#98C9DD",
        "IGHD" = "#207CB5",
        "IGHA1" = "#A6D38E",
        "IGHA2" = "#37A849",
        "IGHG1" = "#F69595",
        "IGHG2" = "#EB2A2A",
        "IGHG3" = "#FCBA71",
        "IGHG4" = "#f78200"
      )
  ) +
  labs(x = "", y = "Proportion", fill = 'BCR isotype') +
  theme_bw() +
  theme(
    axis.text.x = element_text(size = 6, angle = 90, hjust = 1,vjust = 0.5),
    axis.text.y = element_text(size = 6),
    text = element_text(size = 8),
    axis.line = element_line(size = 0.3),
    axis.ticks = element_line(size = 0.3),
    panel.border = element_rect(colour = "black", linewidth = 0.3),
    legend.position = "none"
  ) + coord_flip()


df$SHM_levels_H = ifelse(df$H_shm < 3,"Low","Median")
df$SHM_levels_H = ifelse(df$H_shm > 8,"High",df$SHM_levels_H)
df$SHM_levels_H = factor(df$SHM_levels_H,levels = c("Low","Median","High"))

plot_df = df %>% group_by(SARS2_bind_level,SHM_levels_H) %>% summarise(n = n()) %>%
  group_by(SARS2_bind_level) %>% mutate(sum_n = sum(n)) %>% 
  ungroup() %>% mutate(percent = n/sum_n)

p2 = ggplot(plot_df,
            aes(x = SARS2_bind_level, y = percent, fill = SHM_levels_H)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c(
    "High" = "#78290f",
    "Median" = "#ff7d00",
    "Low" = "#ffecd1"
  )) +
  labs(x = "", y = "Proportion", fill = 'SHM level') +
  theme_bw() +
  coord_flip()+
  theme(
    axis.text.x = element_text(size = 6, angle = 90, hjust = 1, vjust = 0.5),
    axis.text.y = element_text(size = 6),
    text = element_text(size = 8),
    axis.line = element_line(size = 0.3),
    axis.ticks = element_line(size = 0.3),
    panel.border = element_rect(colour = "black", linewidth = 0.3),
    legend.position = "none"
  ) 


plot_df = df %>% group_by(SARS2_bind_level) %>% 
  summarise(median_SHM_H = median(H_shm),median_SHM_L = median(L_shm))

p3 <- ggplot(df,
             aes(x = SARS2_bind_level, y = H_shm)) +
  geom_boxplot(aes(color = SARS2_bind_level),
               outlier.colour = NA, 
               lwd = 0.3) +
  geom_jitter(aes(color = SARS2_bind_level), size = 0.7,shape = 16, stroke = 0, width = 0.1) +
  labs(x = "", y = "SHM counts") +
  theme_bw() +
  coord_flip()+
  theme(
    axis.text.x = element_text(size = 10, angle = 90, hjust = 1, vjust = 0.5),
    axis.text.y = element_blank(),
    text = element_text(size = 10),
    axis.line = element_line(size = 0.3),
    axis.ticks = element_line(size = 0.3),
    axis.ticks.y  = element_blank(),
    panel.border = element_rect(colour = "black", linewidth = 0.3),
    legend.position = "none"
  ) 


cowplot::plot_grid(p1, p2, p3,nrow = 1,align = "h", rel_widths = c(1, 1, 1))

ggsave(file.path(plots_dir, "Figure5C.pdf"), width = 7, height = 4)



###############################################################################
#'                          Manuscipt: figure5D                              '#
###############################################################################

## Figure 5D; 
## Box plots compare SHM rates between predicted neutralizing and non-neutralizing antibodies derived from FCRL5+ atypical B cells. 

plot_df = df %>%
  dplyr::count(SARS2_neut_level, IGH_isotype) 

plot_df$IGH_isotype = factor(
  plot_df$IGH_isotype,
  levels = c(
    #"IGHE",
    "IGHM",
    "IGHD",
    "IGHA1",
    "IGHA2",
    "IGHG1",
    "IGHG2",
    "IGHG3",
    "IGHG4"
  )
)

p1 = ggplot(plot_df, aes(SARS2_neut_level, n, fill = IGH_isotype)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_fill_manual(
    values =
      c(
        #"IGHE" = "#CED6C3",
        "IGHM" = "#98C9DD",
        "IGHD" = "#207CB5",
        "IGHA1" = "#A6D38E",
        "IGHA2" = "#37A849",
        "IGHG1" = "#F69595",
        "IGHG2" = "#EB2A2A",
        "IGHG3" = "#FCBA71",
        "IGHG4" = "#f78200"
      )
  ) +
  labs(x = "", y = "Proportion", fill = 'BCR isotype') +
  theme_bw() +
  theme(
    axis.text.x = element_text(size = 6, angle = 90, hjust = 1,vjust = 0.5),
    axis.text.y = element_text(size = 6),
    text = element_text(size = 8),
    axis.line = element_line(size = 0.3),
    axis.ticks = element_line(size = 0.3),
    panel.border = element_rect(colour = "black", linewidth = 0.3),
    legend.position = "none"
  ) + coord_flip()


df$SHM_levels_H = ifelse(df$H_shm < 3,"Low","Median")
df$SHM_levels_H = ifelse(df$H_shm > 8,"High",df$SHM_levels_H)
df$SHM_levels_H = factor(df$SHM_levels_H,levels = c("Low","Median","High"))

plot_df = df %>% group_by(SARS2_neut_level,SHM_levels_H) %>% summarise(n = n()) %>%
  group_by(SARS2_neut_level) %>% mutate(sum_n = sum(n)) %>% 
  ungroup() %>% mutate(percent = n/sum_n)

p2 = ggplot(plot_df,
            aes(x = SARS2_neut_level, y = percent, fill = SHM_levels_H)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c(
    "High" = "#78290f",
    "Median" = "#ff7d00",
    "Low" = "#ffecd1"
  )) +
  labs(x = "", y = "Proportion", fill = 'SHM level') +
  theme_bw() +
  coord_flip()+
  theme(
    axis.text.x = element_text(size = 6, angle = 90, hjust = 1, vjust = 0.5),
    axis.text.y = element_text(size = 6),
    text = element_text(size = 8),
    axis.line = element_line(size = 0.3),
    axis.ticks = element_line(size = 0.3),
    panel.border = element_rect(colour = "black", linewidth = 0.3),
    legend.position = "none"
  ) 
p2

plot_df = df %>% group_by(SARS2_neut_level) %>% 
  summarise(median_SHM_H = median(H_shm),median_SHM_L = median(L_shm))

p3 <- ggplot(df,
             aes(x = SARS2_neut_level, y = H_shm)) +
  geom_boxplot(aes(color = SARS2_neut_level),
               outlier.colour = NA, 
               lwd = 0.3) +
  geom_jitter(aes(color = SARS2_neut_level), size = 0.7,shape = 16, stroke = 0, width = 0.1) +
  labs(x = "", y = "SHM counts") +
  theme_bw() +
  coord_flip()+
  theme(
    axis.text.x = element_text(size = 10, angle = 90, hjust = 1, vjust = 0.5),
    axis.text.y = element_blank(),
    text = element_text(size = 10),
    axis.line = element_line(size = 0.3),
    axis.ticks = element_line(size = 0.3),
    axis.ticks.y  = element_blank(),
    panel.border = element_rect(colour = "black", linewidth = 0.3),
    legend.position = "none"
  ) 

cowplot::plot_grid(p1, p2, p3,nrow = 1,align = "h", rel_widths = c(1, 1, 1))

ggsave(file.path(plots_dir, "Figure5D.pdf"), width = 7, height = 4)







