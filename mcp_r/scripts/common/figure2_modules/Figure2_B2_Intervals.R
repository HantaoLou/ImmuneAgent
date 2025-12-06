# Figure2 B2区间分布分析模块 - 独立运行版本
# 结合预测值区间分布分析

################ Figure 2 B2 - 独立运行 ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure2_B2_Intervals.R <input_rds_file> <base_dir> [interval_step] [data_min] [data_max]")
}

input_rds_file <- args[1]
base_dir <- args[2]

# 区间分析参数（可选，保持向后兼容的默认值）
interval_step <- 0.1  # 默认步长0.1
data_min <- 0         # 默认最小值0
data_max <- 1         # 默认最大值1

# 解析可选参数
if (length(args) >= 3) {
  interval_step <- as.numeric(args[3])
  if (is.na(interval_step)) {
    stop("Error: interval_step must be a valid number")
  }
}

if (length(args) >= 4) {
  data_min <- as.numeric(args[4])
  if (is.na(data_min)) {
    stop("Error: data_min must be a valid number")
  }
}

if (length(args) >= 5) {
  data_max <- as.numeric(args[5])
  if (is.na(data_max)) {
    stop("Error: data_max must be a valid number")
  }
}

# 参数验证和健壮性检查
if (data_min >= data_max) {
  stop("Error: data_min (", data_min, ") must be less than data_max (", data_max, ")")
}

if (interval_step <= 0) {
  stop("Error: interval_step (", interval_step, ") must be greater than 0")
}

data_range <- data_max - data_min
if (interval_step >= data_range) {
  stop("Error: interval_step (", interval_step, ") must be smaller than data range (", data_range, ")")
}

# 计算区间数量并给出警告
interval_count <- ceiling(data_range / interval_step)
if (interval_count < 2) {
  stop("Error: Parameters result in less than 2 intervals. Please adjust interval_step.")
}
if (interval_count > 50) {
  warning("Warning: This will create ", interval_count, " intervals, which may be too many for meaningful analysis.")
}

# 输出参数信息
cat("区间分析参数:\n")
cat("  数据范围: [", data_min, ", ", data_max, "]\n")
cat("  区间步长: ", interval_step, "\n")
cat("  区间数量: ", interval_count, "\n")

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
paths <- initialize_figure2_environment(input_rds_file, base_dir, "B2")
output_dir <- paths$output_dir
plots_dir <- paths$plots_dir
files_dir <- paths$files_dir

# 加载和预处理数据
cell_obj <- load_and_preprocess_data(input_rds_file)

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
  
  # 将结合预测值分成动态数量的等差区间
  metadata_filtered <- metadata_filtered %>%
    mutate(bind_interval = cut(
      bind_average_values,
      breaks = seq(data_min, data_max, by = interval_step),
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

cat("Figure 2B2区间分布分析模块执行完成\n")
cat("输出目录:", output_dir, "\n")
cat("图片保存至:", plots_dir, "\n")