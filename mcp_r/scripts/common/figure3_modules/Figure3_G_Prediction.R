# Figure3 G 预测值密度图模块 - 独立运行版本
# UMAP - 通用预测值密度图

################ Figure 3G - 独立运行 ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure3_G_Prediction.R <input_rds_file> <base_dir> [prediction_keywords] [prediction_threshold]")
}

input_rds_file <- args[1]
base_dir <- args[2]

# 生信核心参数1: 预测字段检测关键词
prediction_keywords <- if (length(args) >= 3 && args[3] != "") {
  strsplit(args[3], ",")[[1]]
} else {
  c("bind", "predict", "output", "average", "score")  # 默认关键词
}

# 生信核心参数2: 预测值阈值设置
prediction_threshold <- if (length(args) >= 4 && args[4] != "") {
  as.numeric(args[4])
} else {
  0.5  # 默认阈值
}

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

# 加载和预处理数据
cell_obj <- load_and_preprocess_data(input_rds_file)

###############################################################################
#'                          Manuscipt: figure3G                              '#
###############################################################################

## Figure 3G; 
## UMAP - 通用预测值密度图

# 检测可用的预测字段
# 使用参数化关键词检测预测字段
detected_cols <- c()
for (keyword in prediction_keywords) {
  matching_cols <- grep(keyword, colnames(cell_obj@meta.data), 
                       value = TRUE, ignore.case = TRUE)
  detected_cols <- c(detected_cols, matching_cols)
  if (length(matching_cols) > 0) {
    cat("Found columns with keyword '", keyword, "':", paste(matching_cols, collapse = ", "), "\n")
  }
}

# 同时使用原有的检测函数作为补充
bind_cols <- detect_all_binding_columns(cell_obj@meta.data)

# 合并所有检测到的预测字段
available_pred_cols <- unique(c(detected_cols, bind_cols))

cat("检测到的预测字段:", paste(available_pred_cols, collapse = ", "), "\n")

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
    
    # 显示预测值统计
    cat("预测值范围:", paste(range(meta_data[[pred_col]], na.rm = TRUE), collapse = " to "), "\n")
    cat("预测值均值:", mean(meta_data[[pred_col]], na.rm = TRUE), "\n")
    cat("非零预测值数量:", sum(meta_data[[pred_col]] > 0, na.rm = TRUE), "\n")
  }
  
  # 检查UMAP坐标是否存在
  if(all(c("UMAP_1", "UMAP_2") %in% colnames(meta_data))) {
    # 根据预测值阈值设置高亮
    meta_data$highlight <- ifelse(meta_data[[pred_col]] >= prediction_threshold, "highlight", "normal")
    cat("使用预测值阈值:", prediction_threshold, "\n")
    
    # 统计高亮和普通细胞数量
    highlight_count <- sum(meta_data$highlight == "highlight")
    normal_count <- sum(meta_data$highlight == "normal")
    cat("高亮细胞数量:", highlight_count, "\n")
    cat("普通细胞数量:", normal_count, "\n")
    
    # 创建图形
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
        legend.text = element_text(size = 8),
        aspect.ratio = 1
      ) +
      labs(title = paste("Prediction Values:", pred_col),
           x = "UMAP_1",
           y = "UMAP_2")
    
    # 保存图片
    ggsave(file.path(plots_dir, "Figure_3G.pdf"), 
           plot = p, 
           width = 7, 
           height = 6,
           dpi = 300)
    
    cat("Figure 3G 已成功生成\n")
    
    # 保存预测值统计信息
    pred_stats <- data.frame(
      Column = pred_col,
      Min = min(meta_data[[pred_col]], na.rm = TRUE),
      Max = max(meta_data[[pred_col]], na.rm = TRUE),
      Mean = mean(meta_data[[pred_col]], na.rm = TRUE),
      Median = median(meta_data[[pred_col]], na.rm = TRUE),
      NonZero_Count = sum(meta_data[[pred_col]] > 0, na.rm = TRUE),
      Total_Count = nrow(meta_data),
      stringsAsFactors = FALSE
    )
    write.csv(pred_stats, file.path(files_dir, "Figure3G_prediction_stats.csv"), row.names = FALSE)
    
  } else {
    cat("警告：未找到UMAP坐标，跳过Figure 3G生成\n")
    
    # 尝试从Seurat对象的reduction中获取UMAP坐标
    if ("umap" %in% names(cell_obj@reductions)) {
      umap_coords <- Embeddings(cell_obj, reduction = "umap")
      meta_data$UMAP_1 <- umap_coords[, 1]
      meta_data$UMAP_2 <- umap_coords[, 2]
      cat("从Seurat对象的reduction中获取UMAP坐标，重新尝试生成图形\n")
      
      # 重新执行图形生成代码
      # 根据预测值阈值设置高亮
      meta_data$highlight <- ifelse(meta_data[[pred_col]] >= prediction_threshold, "highlight", "normal")
      
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
          legend.text = element_text(size = 8),
          aspect.ratio = 1
        ) +
        labs(title = paste("Prediction Values:", pred_col),
             x = "UMAP_1",
             y = "UMAP_2")
      
      ggsave(file.path(plots_dir, "Figure_3G.pdf"), 
             plot = p, 
             width = 7, 
             height = 6,
             dpi = 300)
      
      cat("Figure 3G 已成功生成（使用Seurat reduction坐标）\n")
    }
  }
} else {
  cat("警告：未找到预测字段，跳过Figure 3G生成\n")
  
  # 保存可用列名用于调试
  available_columns <- data.frame(
    Column = colnames(cell_obj@meta.data),
    stringsAsFactors = FALSE
  )
  write.csv(available_columns, file.path(files_dir, "Figure3G_available_columns.csv"), row.names = FALSE)
}

###########################################################
# 记录运行环境
sink(file.path(plots_dir, "Figure3G_session_info.txt"))  # 将输出重定向到文件
cat("Figure 3G Session Information\n")
cat("=============================\n")
cat("Generated on:", as.character(Sys.time()), "\n")
cat("Input file:", input_rds_file, "\n")
cat("Base directory:", base_dir, "\n")
cat("Prediction keywords:", paste(prediction_keywords, collapse = ", "), "\n")
cat("Prediction threshold:", prediction_threshold, "\n")
cat("Available prediction columns:", paste(available_pred_cols, collapse = ", "), "\n")
if(length(available_pred_cols) > 0) {
  cat("Used prediction column:", available_pred_cols[1], "\n")
}
sessionInfo()
sink()  # 关闭重定向

cat("Figure 3G 模块运行完成！\n")