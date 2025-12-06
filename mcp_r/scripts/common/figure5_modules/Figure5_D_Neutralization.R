# Figure5 D 中和抗体SHM率比较分析模块 - 独立运行版本
# Box plots compare SHM rates between predicted neutralizing and non-neutralizing antibodies

################ Figure 5D - 独立运行 ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure5_D_Neutralization.R <input_rds_file> <base_dir> [binding_threshold]")
}

input_rds_file <- args[1]
base_dir <- args[2]

# 生信核心参数: 结合阈值（中和抗体分析的基础分类标准）
binding_threshold <- if (length(args) >= 3 && args[3] != "") {
  as.numeric(args[3])
} else {
  0.5  # 默认阈值，与Figure5C保持一致，适合中和抗体分析
}

# 检查输入文件是否存在
if (!file.exists(input_rds_file)) {
  stop(paste("Input file does not exist:", input_rds_file))
}

# 检查基础目录是否存在
if (!dir.exists(base_dir)) {
  stop(paste("Base directory does not exist:", base_dir))
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
source(file.path(script_dir, "Figure5_Utils.R"))

# 加载必需的R包
load_required_packages()

# 创建输出目录
output_dirs <- create_output_directories(base_dir)
plots_dir <- output_dirs$plots_dir
files_dir <- output_dirs$files_dir

# 加载和预处理数据
cell_obj <- load_and_preprocess_data(input_rds_file)

###############################################################################
#'                          Manuscipt: figure5D                              '#
###############################################################################

## Figure 5D; 
## Box plots compare SHM rates between predicted neutralizing and non-neutralizing antibodies derived from FCRL5+ atypical B cells. 

# 创建IGH_isotype字段
cell_obj <- create_igh_isotype_field(cell_obj)

# 使用函数估算SHM水平
shm_results <- estimate_shm_from_expression(cell_obj)
cell_obj$H_shm <- shm_results$H_shm
cell_obj$L_shm <- shm_results$L_shm

# 计算SARS2_bind_average_values和SARS2_neut_average_values
cell_obj <- calculate_sars2_predictions(cell_obj)

# 准备分析数据框
df <- prepare_analysis_dataframe(cell_obj)

# 选择目标细胞类型
df <- select_target_celltype(df)

# 创建结合和中和水平分类（使用参数化的结合阈值）
cat("使用结合阈值:", binding_threshold, "进行中和抗体分类分析\n")
df <- create_binding_neutralization_levels(df, threshold = binding_threshold)

# 过滤异常SHM值
df <- filter_extreme_shm(df)

# 创建SHM水平分类
df <- create_shm_levels(df)

# 生成图形 - 基于中和水平分析
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

# 组合图形并保存
combined_plot <- cowplot::plot_grid(p1, p2, p3, nrow = 1, align = "h", rel_widths = c(1, 1, 1))
ggsave(file.path(plots_dir, "Figure5D.pdf"), plot = combined_plot, width = 7, height = 4)

# 保存数据
write.csv(df, file.path(files_dir, "Figure5D_analysis_data.csv"), row.names = FALSE)

# 保存统计数据
stats_df <- df %>% group_by(SARS2_neut_level) %>% 
  summarise(
    count = n(),
    median_SHM_H = median(H_shm, na.rm = TRUE),
    median_SHM_L = median(L_shm, na.rm = TRUE),
    mean_SHM_H = mean(H_shm, na.rm = TRUE),
    mean_SHM_L = mean(L_shm, na.rm = TRUE),
    .groups = 'drop'
  )
write.csv(stats_df, file.path(files_dir, "Figure5D_statistics.csv"), row.names = FALSE)

cat("Figure 5D generated successfully\n")

###########################################################
# 记录运行环境
sink(file.path(plots_dir, "Figure5D_session_info.txt"))  # 将输出重定向到文件
cat("Figure 5D Session Information\n")
cat("=============================\n")
cat("Generated on:", as.character(Sys.time()), "\n")
cat("Input file:", input_rds_file, "\n")
cat("Base directory:", base_dir, "\n")
cat("Parameters:\n")
cat("  - Binding threshold:", binding_threshold, "\n")
cat("Final data rows:", nrow(df), "\n")
sessionInfo()
sink()  # 关闭重定向

cat("Figure 5D 模块运行完成！\n")