# Figure3 A 结合/中和预测密度图模块 - 独立运行版本
# UMAP - 通用结合/中和预测密度图

################ Figure 3A - 独立运行 ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure3_A_Density.R <input_rds_file> <base_dir> [prediction_keywords] [na_strategy] [feature_priority]")
}

input_rds_file <- args[1]
base_dir <- args[2]

# 生信核心参数配置
# 参数1: 预测字段检测关键词 (默认值)
prediction_keywords <- if (length(args) >= 3 && args[3] != "") {
  strsplit(args[3], ",")[[1]]
} else {
  c("neut", "bind", "average", "predict", "output")
}

# 参数2: NA值处理策略 (默认排除含NA的细胞)
na_strategy <- if (length(args) >= 4 && args[4] != "") {
  args[4]
} else {
  "exclude_cells"  # 可选: "exclude_cells", "replace_zero", "replace_median"
}

# 参数3: 特征选择优先级 (默认中和优先)
feature_priority <- if (length(args) >= 5 && args[5] != "") {
  args[5]
} else {
  "neutralization_first"  # 可选: "neutralization_first", "binding_first", "highest_value"
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
#'                          Manuscipt: figure3a                              '#
###############################################################################

## Figure 3A; 
## UMAP - 通用结合/中和预测密度图

# 通用化代码：使用参数化关键词检测预测字段
# 基于用户提供的关键词检测预测字段
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
available_prediction_cols <- unique(c(detected_cols, bind_cols))
cat("Total available prediction columns:", paste(available_prediction_cols, collapse = ", "), "\n")

if (length(available_prediction_cols) > 0) {
  # 使用第一个可用的预测字段作为过滤条件
  filter_col <- available_prediction_cols[1]
  
  # 根据NA处理策略处理数据
  if (na_strategy == "exclude_cells") {
    # 排除含NA值的细胞
    sub <- cell_obj[, !is.na(cell_obj@meta.data[[filter_col]])]
    cat("NA处理策略: 排除含NA的细胞，剩余细胞数:", ncol(sub), "\n")
  } else {
    # 保留所有细胞，后续处理NA值
    sub <- cell_obj
    cat("NA处理策略:", na_strategy, "，保留所有细胞数:", ncol(sub), "\n")
  }
  
  # 选择所有可用的预测字段创建新数据矩阵
  available_cols_in_sub <- intersect(available_prediction_cols, colnames(sub@meta.data))
  
  if (length(available_cols_in_sub) > 0) {
    new_data <- sub@meta.data[, available_cols_in_sub, drop = FALSE]
    
    # 根据NA策略确保数据是数值类型
    new_data <- apply(new_data, 2, function(x) {
      x <- as.character(x)
      if (na_strategy == "replace_zero") {
        x[x == "NA" | is.na(x)] <- "0"
      } else if (na_strategy == "replace_median") {
        numeric_x <- as.numeric(x)
        median_val <- median(numeric_x, na.rm = TRUE)
        x[x == "NA" | is.na(x)] <- as.character(median_val)
      } else {
        # exclude_cells策略下，NA值应该已经被过滤掉
        x[x == "NA" | is.na(x)] <- "0"
      }
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
    
    # 根据特征优先级策略选择特征
    original_feature <- NULL
    
    if (feature_priority == "neutralization_first") {
      # 优先选择中和相关字段
      neut_cols <- grep("neut|neutralization|inhibition", available_cols_in_sub, 
                       value = TRUE, ignore.case = TRUE)
      if (length(neut_cols) > 0) {
        original_feature <- neut_cols[1]
        cat("选择中和预测字段:", original_feature, "\n")
      }
    } else if (feature_priority == "binding_first") {
      # 优先选择结合相关字段
      bind_cols_sub <- grep("bind|binding|affinity", available_cols_in_sub, 
                           value = TRUE, ignore.case = TRUE)
      if (length(bind_cols_sub) > 0) {
        original_feature <- bind_cols_sub[1]
        cat("选择结合预测字段:", original_feature, "\n")
      }
    } else if (feature_priority == "highest_value") {
      # 选择平均值最高的字段
      col_means <- sapply(available_cols_in_sub, function(col) {
        mean(as.numeric(sub@meta.data[[col]]), na.rm = TRUE)
      })
      original_feature <- names(which.max(col_means))
      cat("选择平均值最高的字段:", original_feature, "，平均值:", max(col_means), "\n")
    }
    
    # 如果没有找到合适的特征，使用第一个可用的
    if (is.null(original_feature)) {
      original_feature <- available_cols_in_sub[1]
      cat("使用默认第一个字段:", original_feature, "\n")
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

###########################################################
# 记录运行环境
sink(file.path(plots_dir, "Figure3A_session_info.txt"))  # 将输出重定向到文件
cat("Figure 3A Session Information\n")
cat("=============================\n")
cat("Generated on:", as.character(Sys.time()), "\n")
cat("Input file:", input_rds_file, "\n")
cat("Base directory:", base_dir, "\n")
cat("Prediction keywords:", paste(prediction_keywords, collapse = ", "), "\n")
cat("NA handling strategy:", na_strategy, "\n")
cat("Feature priority:", feature_priority, "\n")
cat("Available prediction columns:", paste(available_prediction_cols, collapse = ", "), "\n")
if (exists("original_feature")) {
  cat("Selected feature:", original_feature, "\n")
}
sessionInfo()
sink()  # 关闭重定向

cat("Figure 3A 模块运行完成！\n")