# Figure3 F 差异表达基因相关性分析模块 - 独立运行版本
# 分析两个数据集之间差异表达基因的相关性

################ Figure 3F - 独立运行 ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript Figure3_F_Correlation.R <deg_file1> <deg_file2> <base_dir> <dataset1_name> <dataset2_name> [p_value_threshold] [min_common_genes] [highlight_genes]")
}

deg_file1 <- args[1]
deg_file2 <- args[2]
base_dir <- args[3]
dataset1_name <- args[4]
dataset2_name <- if (length(args) >= 5) args[5] else "Dataset2"

# 生信核心参数配置
# 参数1: p值阈值（用于过滤显著基因）
p_value_threshold <- if (length(args) >= 6 && args[6] != "") {
  as.numeric(args[6])
} else {
  0.05  # 默认显著性阈值
}

# 参数2: 最小共同基因数量（确保分析的统计学意义）
min_common_genes <- if (length(args) >= 7 && args[7] != "") {
  as.numeric(args[7])
} else {
  10  # 默认最少需要10个共同基因
}

# 参数3: 高亮基因列表（用逗号分隔）
highlight_genes <- if (length(args) >= 8 && args[8] != "") {
  strsplit(args[8], ",")[[1]]
} else {
  c('ITGAX','FGR','FCRL4','FCRL5','CD68','TNFRSF1B','JCHAIN','MZB1','XBP1','MARCKSL1')
}

# 参数验证
if (p_value_threshold <= 0 || p_value_threshold >= 1) {
  stop("Error: p_value_threshold must be between 0 and 1")
}

if (min_common_genes < 1) {
  stop("Error: min_common_genes must be >= 1")
}

# 输出参数信息
cat("相关性分析参数:\n")
cat("  Dataset 1:", dataset1_name, "\n")
cat("  Dataset 2:", dataset2_name, "\n")
cat("  P-value threshold:", p_value_threshold, "\n")
cat("  Min common genes:", min_common_genes, "\n")
cat("  Highlight genes:", paste(highlight_genes, collapse = ", "), "\n")

# 获取当前脚本所在目录并加载工具函数
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

script_dir <- get_script_dir()
source(file.path(script_dir, "Figure3_Utils.R"))

# 加载必需的R包
load_required_packages()

# 创建输出目录
output_dirs <- create_output_directories(base_dir)
plots_dir <- output_dirs$plots_dir
files_dir <- output_dirs$files_dir

###############################################################################
#'                          Manuscipt: figure3F                              '#
###############################################################################

## Figure 3F; 
## Correlation of differentially expressed genes between two datasets

# 输入文件验证函数
validate_deg_file <- function(file_path, dataset_name) {
  # 检查文件是否存在
  if (!file.exists(file_path)) {
    stop(paste("DEG file does not exist:", file_path))
  }
  
  # 检查文件是否为CSV格式
  if (!grepl("\\.csv$", file_path, ignore.case = TRUE)) {
    stop(paste("File must be CSV format:", file_path))
  }
  
  # 尝试读取文件并检查必需字段
  tryCatch({
    data <- read.csv(file_path, row.names = 1, stringsAsFactors = FALSE)
    
    # 检查必需的列
    required_cols <- c("avg_log2FC", "p_val_adj")
    missing_cols <- setdiff(required_cols, colnames(data))
    
    if (length(missing_cols) > 0) {
      stop(paste("Missing required columns in", dataset_name, ":", paste(missing_cols, collapse = ", ")))
    }
    
    # 检查数据行数
    if (nrow(data) == 0) {
      stop(paste("No data found in", dataset_name, "file"))
    }
    
    cat("✓", dataset_name, "file validation passed:", nrow(data), "genes\n")
    return(data)
    
  }, error = function(e) {
    stop(paste("Error reading", dataset_name, "file:", e$message))
  })
}

# 验证并加载DEG数据
cat("验证输入文件...\n")
data1 <- validate_deg_file(deg_file1, dataset1_name)
data2 <- validate_deg_file(deg_file2, dataset2_name)

# 过滤显著基因
cat("过滤显著基因 (p_val_adj <", p_value_threshold, ")...\n")
data1_sig <- data1[data1$p_val_adj < p_value_threshold, ]
data2_sig <- data2[data2$p_val_adj < p_value_threshold, ]

cat("  ", dataset1_name, "显著基因:", nrow(data1_sig), "\n")
cat("  ", dataset2_name, "显著基因:", nrow(data2_sig), "\n")

# 找到共同的基因
common_genes <- intersect(rownames(data1_sig), rownames(data2_sig))
cat("共同显著基因数量:", length(common_genes), "\n")

# 检查是否有足够的共同基因进行分析
if (length(common_genes) < min_common_genes) {
  stop(paste("Insufficient common genes for analysis. Found:", length(common_genes), 
             ", Required:", min_common_genes))
}

# 合并数据
if (length(common_genes) > 0) {
  # 清理数据集名称，移除特殊字符并转换为小写
  clean_name1 <- tolower(gsub("[^a-zA-Z0-9]", "", dataset1_name))
  clean_name2 <- tolower(gsub("[^a-zA-Z0-9]", "", dataset2_name))
  
  # 提取共同基因的数据
  merged_data <- merge(
    data1_sig[common_genes, c("avg_log2FC", "pct.1", "pct.2", "p_val_adj")],
    data2_sig[common_genes, c("avg_log2FC", "pct.1", "pct.2", "p_val_adj")],
    by = "row.names", 
    all = FALSE,  # 只保留共同基因
    suffixes = c(paste0("_", clean_name1), paste0("_", clean_name2))
  )
  
  # 设置基因名为行名
  rownames(merged_data) <- merged_data$Row.names
  merged_data$Row.names <- NULL
  merged_data$gene <- rownames(merged_data)
  
  cat("成功合并数据，共", nrow(merged_data), "个基因\n")
  
  # 保存合并后的数据
  write.csv(merged_data, file.path(files_dir, paste0("correlation_data_", dataset1_name, "_vs_", dataset2_name, ".csv")))
  
  # 计算相关性
  x_col <- paste0("avg_log2FC_", clean_name1)
  y_col <- paste0("avg_log2FC_", clean_name2)
  
  correlation_test <- cor.test(merged_data[[x_col]], merged_data[[y_col]], method = "pearson")
  
  cat("相关性分析结果:\n")
  cat("  Pearson相关系数:", round(correlation_test$estimate, 4), "\n")
  cat("  P值:", format(correlation_test$p.value, scientific = TRUE), "\n")
  cat("  95%置信区间:", paste(round(correlation_test$conf.int, 4), collapse = " - "), "\n")
  
  # 识别要高亮的基因
  highlight_available <- intersect(highlight_genes, merged_data$gene)
  highlight_missing <- setdiff(highlight_genes, merged_data$gene)
  
  if (length(highlight_missing) > 0) {
    cat("以下高亮基因在数据中未找到:", paste(highlight_missing, collapse = ", "), "\n")
  }
  
  cat("将高亮显示", length(highlight_available), "个基因\n")
  
  # 绘制散点图
  p <- ggplot(merged_data, aes_string(x = x_col, y = y_col)) +
    geom_point(color = "black", size = 0.8, alpha = 0.7) +
    geom_hline(yintercept = 0, color = "blue", linetype = "dashed", alpha = 0.7) +
    geom_vline(xintercept = 0, color = "blue", linetype = "dashed", alpha = 0.7) +
    labs(
      title = paste("Gene Expression Correlation:", dataset1_name, "vs", dataset2_name),
      subtitle = paste("r =", round(correlation_test$estimate, 3), 
                      ", p =", format(correlation_test$p.value, digits = 3),
                      ", n =", length(common_genes), "genes"),
      x = paste(dataset1_name, "log2FC"),
      y = paste(dataset2_name, "log2FC")
    ) +
    theme_minimal() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
      plot.background = element_rect(color = "black", linewidth = 0.5),
      plot.title = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 10),
      axis.title = element_text(size = 11),
      axis.text = element_text(size = 9)
    )
  
  # 添加基因标签（如果有高亮基因）
  if (length(highlight_available) > 0) {
    highlight_data <- merged_data[merged_data$gene %in% highlight_available, ]
    
    p <- p + geom_text_repel(
      data = highlight_data,
      aes(label = gene),
      color = "purple",
      size = 3,
      box.padding = 0.5,
      point.padding = 0.3,
      max.overlaps = 20,
      min.segment.length = 0.1,
      segment.color = "grey50",
      segment.size = 0.3
    )
  }
  
  # 保存图片
  output_filename <- paste0("Figure_3F_", dataset1_name, "_vs_", dataset2_name, ".pdf")
  ggsave(file.path(plots_dir, output_filename), p, width = 6, height = 5, dpi = 300)
  
  # 保存统计结果
  stats_results <- data.frame(
    Dataset1 = dataset1_name,
    Dataset2 = dataset2_name,
    Common_genes = length(common_genes),
    Correlation = correlation_test$estimate,
    P_value = correlation_test$p.value,
    CI_lower = correlation_test$conf.int[1],
    CI_upper = correlation_test$conf.int[2],
    P_threshold = p_value_threshold,
    Highlight_genes_found = length(highlight_available),
    stringsAsFactors = FALSE
  )
  
  write.csv(stats_results, file.path(files_dir, paste0("correlation_stats_", dataset1_name, "_vs_", dataset2_name, ".csv")), row.names = FALSE)
  
  cat("分析完成！\n")
  cat("图片保存至:", file.path(plots_dir, output_filename), "\n")
  cat("数据保存至:", files_dir, "\n")
  
} else {
  stop("No common genes found between datasets")
}

cat("Figure 3F 相关性分析模块执行完成\n")