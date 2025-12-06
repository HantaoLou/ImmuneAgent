# Figure2 A2结合预测可视化模块 - 独立运行版本
# 结合预测可视化

################ Figure 2 A2 - 独立运行 ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure2_A2_Binding.R <input_rds_file> <base_dir> [binding_threshold]")
}

input_rds_file <- args[1]
base_dir <- args[2]

# 广泛反应性阈值参数（可选，默认值为0.5）
binding_threshold <- 0.5  # 默认值
if (length(args) >= 3) {
  binding_threshold <- as.numeric(args[3])
  if (is.na(binding_threshold) || binding_threshold < 0 || binding_threshold > 1) {
    stop("Error: binding_threshold must be a number between 0 and 1")
  }
}

cat("使用广泛反应性阈值:", binding_threshold, "\n")

# 获取当前脚本所在目录并加载工具函数
# 使用更可靠的方法获取脚本路径
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
source(file.path(script_dir, "Figure2_Utils.R"))

# 初始化环境和创建输出目录
paths <- initialize_figure2_environment(input_rds_file, base_dir, "A2")
output_dir <- paths$output_dir
plots_dir <- paths$plots_dir
files_dir <- paths$files_dir

# 加载和预处理数据
cell_obj <- load_and_preprocess_data(input_rds_file)

###############################################################################
#'                          Manuscipt: figure2A2                            '#
###############################################################################

## Figure 2A.2; 
## Prediction of binding breadth against flu variants (adapted from SARS-CoV-2)

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
    cat("Number of cells with binding score >", binding_threshold, ":", 
        sum(meta_data$bind_average_values > binding_threshold, na.rm = TRUE), "\n")
    cat("Percentage of broadly reactive cells (>", binding_threshold, "):", 
        round(100 * sum(meta_data$bind_average_values > binding_threshold, na.rm = TRUE) / nrow(meta_data), 2), "%\n")
  }
  
} else {
  print("No binding prediction columns found in metadata")
}

cat("Figure 2A2结合预测可视化模块执行完成\n")
cat("输出目录:", output_dir, "\n")
cat("图片保存至:", plots_dir, "\n")