# Figure2 C火山图和DEG分析模块 - 独立运行版本
# 火山图和差异表达基因分析

################ Figure 2 C - 独立运行 ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure2_C_Volcano.R <input_rds_file> <base_dir> [logfc_threshold] [min_pct] [analysis_strategy]")
}

input_rds_file <- args[1]
base_dir <- args[2]

# DEG分析阈值参数（可选，保持向后兼容的默认值）
logfc_threshold <- 0    # 默认值：0（不过滤）
min_pct <- 0.2          # 默认值：0.2（20%细胞表达）
analysis_strategy <- "both"  # 默认值：both（运行两种分析）

# 解析可选参数
if (length(args) >= 3) {
  logfc_threshold <- as.numeric(args[3])
  if (is.na(logfc_threshold)) {
    stop("Error: logfc_threshold must be a valid number")
  }
}

if (length(args) >= 4) {
  min_pct <- as.numeric(args[4])
  if (is.na(min_pct)) {
    stop("Error: min_pct must be a valid number")
  }
}

if (length(args) >= 5) {
  analysis_strategy <- args[5]
  if (!analysis_strategy %in% c("both", "broad", "specific")) {
    stop("Error: analysis_strategy must be 'both', 'broad', or 'specific'")
  }
}

# 参数验证
if (logfc_threshold < 0) {
  stop("Error: logfc_threshold (", logfc_threshold, ") must be >= 0")
}

if (min_pct < 0 || min_pct > 1) {
  stop("Error: min_pct (", min_pct, ") must be between 0 and 1")
}

# 输出参数信息
cat("DEG分析参数:\n")
cat("  logfc_threshold: ", logfc_threshold, "\n")
cat("  min_pct: ", min_pct, "\n")
cat("  analysis_strategy: ", analysis_strategy, "\n")

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
paths <- initialize_figure2_environment(input_rds_file, base_dir, "C")
output_dir <- paths$output_dir
plots_dir <- paths$plots_dir
files_dir <- paths$files_dir

# 加载和预处理数据
cell_obj <- load_and_preprocess_data(input_rds_file)

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

# 根据分析策略执行DEG分析
markers <- NULL  # 初始化markers变量

# 广泛对照分析（原setting 11）
if (analysis_strategy %in% c("both", "broad")) {
  strategy_name <- "broad_control"
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
      cat("执行广泛对照分析 (", strategy_name, ")...\n")
      markers <- FindMarkers(sub, ident.1 = "1", ident.2 = "0",
                             logfc.threshold=logfc_threshold, min.pct=min_pct)
      
      # 处理markers结果
      if (exists("markers") && !is.null(markers)) {
        markers$p_val_adj = p.adjust(markers$p_val, method='fdr')
        markers$pct_dif <- markers$pct.1 - markers$pct.2
        write.csv(markers,file.path(files_dir,paste0("DEG_analysis_threshold=",i,"_strategy=",strategy_name,".csv")))
      }
    } else {
      cat("跳过广泛对照分析：组中细胞数量不足 (组0:", cells_0, "个细胞, 组1:", cells_1, "个细胞)\n")
    }
  } else {
    cat("跳过广泛对照分析：缺少必要的身份组\n")
  }
}

# 特异对照分析（原setting 15）
if (analysis_strategy %in% c("both", "specific")) {
  strategy_name <- "specific_control"
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
      cat("执行特异对照分析 (", strategy_name, ")...\n")
      markers <- FindMarkers(sub, ident.1 = "1", ident.2 = "0",
                             logfc.threshold=logfc_threshold, min.pct=min_pct)
      
      # 处理markers结果
      if (exists("markers") && !is.null(markers)) {
        markers$p_val_adj = p.adjust(markers$p_val, method='fdr')
        markers$pct_dif <- markers$pct.1 - markers$pct.2
        write.csv(markers,file.path(files_dir,paste0("DEG_analysis_threshold=",i,"_strategy=",strategy_name,".csv")))
      }
    } else {
      cat("跳过特异对照分析：组中细胞数量不足 (组0:", cells_0, "个细胞, 组1:", cells_1, "个细胞)\n")
    }
  } else {
    cat("跳过特异对照分析：缺少必要的身份组\n")
  }
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

cat("Figure 2C火山图和DEG分析模块执行完成\n")
cat("输出目录:", output_dir, "\n")
cat("图片保存至:", plots_dir, "\n")
cat("文件保存至:", files_dir, "\n")