rm(list = ls())

################ Figure 5  ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript Figure5.R <input_rds_file>")
}

input_rds_file <- args[1]

# 检查输入文件是否存在
if (!file.exists(input_rds_file)) {
  stop(paste("Input file does not exist:", input_rds_file))
}

# 获取脚本所在目录作为工作目录
# 从配置文件读取基础目录
library(jsonlite)
config_path <- file.path(dirname(getwd()), "config.json")
if (file.exists(config_path)) {
  config <- fromJSON(config_path)
  fdir <- config$base_dir
} else {
  fdir <- dirname(getwd())  # 回退到默认行为
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
library(readxl)

## II. load data
# 加载King RSV数据文件
seurat_obj <- readRDS(input_rds_file)

## III. 数据预处理和字段映射
# 计算SHM相关指标（与Figure4.R保持一致的计算方式）
# 直接使用metadata列进行计算，避免额外的向量操作

# 计算H_shm：优先使用IGH_MU_FREQ，其次使用v_identity（与Figure4.R一致）
seurat_obj@meta.data$H_shm <- ifelse(
  !is.na(seurat_obj@meta.data$IGH_MU_FREQ),
  round(seurat_obj@meta.data$IGH_MU_FREQ * 100),
  ifelse(
    !is.na(seurat_obj@meta.data$v_identity_IGH.x),
    round(100 - seurat_obj@meta.data$v_identity_IGH.x),  # 修正：去除错误的*0.3缩放因子
    0
  )
)

# 计算L_shm：基于v_identity_IGL.x（与Figure4.R一致）
seurat_obj@meta.data$L_shm <- ifelse(
  !is.na(seurat_obj@meta.data$v_identity_IGL.x),
  round(100 - seurat_obj@meta.data$v_identity_IGL.x),  # 修正：去除错误的*0.2缩放因子
  0
)

# 限制SHM值在合理范围内
seurat_obj@meta.data$H_shm <- pmax(0, pmin(seurat_obj@meta.data$H_shm, 50))
seurat_obj@meta.data$L_shm <- pmax(0, pmin(seurat_obj@meta.data$L_shm, 30))

# 映射ISOTYPE字段（统一格式）
isotype_mapping <- c("IgM" = "IGHM", "IgD" = "IGHD", "IgG1" = "IGHG1", 
                    "IgG2" = "IGHG2", "IgG3" = "IGHG3", "IgG4" = "IGHG4",
                    "IgA1" = "IGHA1", "IgA2" = "IGHA2", "None" = "IGHA1", 
                    "Multi" = "IGHG1")
seurat_obj@meta.data$IGH_isotype <- isotype_mapping[seurat_obj@meta.data$ISOTYPE]
seurat_obj@meta.data$IGH_isotype[is.na(seurat_obj@meta.data$IGH_isotype)] <- "IGHM"


###############################################################################
#'                          Manuscipt: figure5C                              '#
###############################################################################

## Figure 5C; 
## Bar plots depicting the isotype distribution and SHM rates for broadly reactive BCRs compared with specific and non-binding BCRs. 
## Broadly reactive clonotypes show an elevated IgG1 proportion and higher SHM rates. 

# 使用RSV-A预测数据（类似原来的A1A11分析）
df <- seurat_obj@meta.data[,c("predict.x","predict.y")]
df <- as.data.frame(df)
df$celltype <- seurat_obj@meta.data$CellType
df$IGH_isotype <- seurat_obj@meta.data$IGH_isotype
df$H_shm <- seurat_obj@meta.data$H_shm
df$L_shm <- seurat_obj@meta.data$L_shm

##
df <- df[df$celltype == "MBC FCRL4+",]  # 替代原来的B.09.ITGAX+AtM
df <- df[!is.na(df$predict.y),]
df <- df[!is.na(df$IGH_isotype),]

# 设置阈值（基于RSV预测值的分布调整）
i = 0.2  # 调整阈值以适应RSV数据分布
df$predict.y_level <- ifelse(df$predict.y > 0.1, 'neut', 'not neut')
df$predict.y_level <- factor(df$predict.y_level, levels = c('neut','not neut'))

df$predict.x_level <- ifelse(df$predict.x >= i, 'more broad', 'not bind')
df$predict.x_level <- ifelse(df$predict.x < i & df$predict.x > 0.05, 'less broad', df$predict.x_level)
df$predict.x_level <- ifelse(df$predict.x <= 0.05 & df$predict.x > 0.01, 'specific', df$predict.x_level)
df$predict.x_level <- ifelse(df$predict.x <= 0.01, 'not bind', df$predict.x_level)

df$predict.x_level <- factor(df$predict.x_level, levels = c("more broad","less broad",
                                                              "specific","not bind"))

# 去掉SHM很离谱的值
df <- df[df$H_shm< 45,]

plot_df = df %>%
  dplyr::count(predict.x_level, IGH_isotype) 

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

p1 = ggplot(plot_df, aes(predict.x_level, n, fill = IGH_isotype)) +
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
    axis.text.x = element_text(size = 6, angle = 90, hjust = 1, vjust = 0.5),
    axis.text.y = element_text(size = 6),
    text = element_text(size = 8),
    axis.line = element_line(linewidth = 0.3),
    axis.ticks = element_line(linewidth = 0.3),
    panel.border = element_rect(colour = "black", linewidth = 0.3),
    legend.position = "none"
  ) + coord_flip()


df$SHM_levels_H = ifelse(df$H_shm < 3,"Low","Median")
df$SHM_levels_H = ifelse(df$H_shm > 8,"High",df$SHM_levels_H)
df$SHM_levels_H = factor(df$SHM_levels_H,levels = c("Low","Median","High"))

plot_df = df %>% group_by(predict.x_level,SHM_levels_H) %>% summarise(n = n()) %>%
  group_by(predict.x_level) %>% mutate(sum_n = sum(n)) %>% 
  ungroup() %>% mutate(percent = n/sum_n)

p2 = ggplot(plot_df,
            aes(x = predict.x_level, y = percent, fill = SHM_levels_H)) +
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
    axis.line = element_line(linewidth = 0.3),
    axis.ticks = element_line(linewidth = 0.3),
    panel.border = element_rect(colour = "black", linewidth = 0.3),
    legend.position = "none"
  ) 


plot_df = df %>% group_by(predict.x_level) %>% 
  summarise(median_SHM_H = median(H_shm),median_SHM_L = median(L_shm))

p3 <- ggplot(df,
             aes(x = predict.x_level, y = H_shm)) +
  geom_boxplot(aes(color = predict.x_level),
               outlier.colour = NA, 
               lwd = 0.3) +
  geom_jitter(aes(color = predict.x_level), size = 0.7,shape = 16, stroke = 0, width = 0.1) +
  labs(x = "", y = "SHM counts") +
  theme_bw() +
  coord_flip()+
  theme(
    axis.text.x = element_text(size = 10, angle = 90, hjust = 1, vjust = 0.5),
    axis.text.y = element_blank(),
    text = element_text(size = 10),
    axis.line = element_line(linewidth = 0.3),
    axis.ticks = element_line(linewidth = 0.3),
    axis.ticks.y  = element_blank(),
    panel.border = element_rect(colour = "black", linewidth = 0.3),
    legend.position = "none"
  ) 


cowplot::plot_grid(p1, p2, p3, nrow = 1, align = "h", rel_widths = c(1, 1, 1))

ggsave(file.path(plots_dir, "Figure5C.pdf"), width = 7, height = 4)

###############################################################################
#'                          Manuscipt: figure5D                              '#
###############################################################################

## Figure 5D; 
## Box plots compare SHM rates between predicted neutralizing and non-neutralizing antibodies derived from FCRL5+ atypical B cells. 


# A1A11
plot_df = df %>%
  dplyr::count(predict.y_level, IGH_isotype) 

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

p1 = ggplot(plot_df, aes(predict.y_level, n, fill = IGH_isotype)) +
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

plot_df = df %>% group_by(predict.y_level,SHM_levels_H) %>% summarise(n = n()) %>%
  group_by(predict.y_level) %>% mutate(sum_n = sum(n)) %>% 
  ungroup() %>% mutate(percent = n/sum_n)

p2 = ggplot(plot_df,
            aes(x = predict.y_level, y = percent, fill = SHM_levels_H)) +
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

# 计算统计数据（基于RSV预测分组）
plot_df = df %>% group_by(predict.y_level) %>% 
  summarise(median_SHM_H = median(H_shm),median_SHM_L = median(L_shm))

p3 <- ggplot(df,
             aes(x = predict.y_level, y = H_shm)) +
  geom_boxplot(aes(color = predict.y_level),
               outlier.colour = NA, 
               lwd = 0.3) +
  geom_jitter(aes(color = predict.y_level), size = 0.7,shape = 16, stroke = 0, width = 0.1) +
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


###############################################################################
#'                          Manuscipt: figureS7C                            '#
###############################################################################

## Figure S7C; 
## Bar plots depicting the isotype distribution and SHM rates for broadly reactive BCRs compared with specific and non-binding BCRs. 
## Broadly reactive clonotypes show an elevated IgG1 proportion and higher SHM rates. 

# 使用RSV-B预测数据（类似原来的flu分析）  
df2 <- seurat_obj@meta.data[,c("predict.y","predict.x")]
df2 <- as.data.frame(df2)
df2$celltype <- seurat_obj@meta.data$CellType
df2$IGH_isotype <- seurat_obj@meta.data$IGH_isotype
df2$H_shm <- seurat_obj@meta.data$H_shm
df2$L_shm <- seurat_obj@meta.data$L_shm

## 只取MBC FCRL4+
df2 <- df2[df2$celltype == "MBC FCRL4+",]
df2 <- df2[!is.na(df2$predict.x),]
df2 <- df2[!is.na(df2$IGH_isotype),]
df2 <- df2[!is.na(df2$H_shm),]

# 定义中和、结合 level
df2$predict.x_level <- ifelse(df2$predict.x > 0.1, 'neut', 'not neut')
df2$predict.x_level <- factor(df2$predict.x_level, levels = c('neut','not neut'))

i = 0.15  # 调整阈值以适应RSV-B数据分布
df2$predict.y_level <- ifelse(df2$predict.y >= i, 'more broad', 'less broad')
df2$predict.y_level <- ifelse(df2$predict.y <= 0.03 & df2$predict.y > 0.01, 'specific', df2$predict.y_level)
df2$predict.y_level <- ifelse(df2$predict.y <= 0.01, 'not bind', df2$predict.y_level)

df2$predict.y_level <- factor(df2$predict.y_level, levels = c("more broad","less broad",
                                                           "specific","not bind"))

# 去掉SHM很离谱的值
df2 <- df2[df2$H_shm < 45,]



plot_df = df2 %>%
  dplyr::count(predict.y_level, IGH_isotype) 

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

p1 = ggplot(plot_df, aes(predict.y_level, n, fill = IGH_isotype)) +
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
    axis.text.x = element_text(size = 10, angle = 90, hjust = 1,vjust = 0.5),
    axis.text.y = element_text(size = 10),
    text = element_text(size = 10),
    axis.line = element_line(size = 0.3),
    axis.ticks = element_line(size = 0.3),
    panel.border = element_rect(colour = "black", linewidth = 0.3),
    legend.position = "none"
  ) + coord_flip()


df2$SHM_levels_H = ifelse(df2$H_shm < 3,"Low","Median")
df2$SHM_levels_H = ifelse(df2$H_shm > 8,"High",df2$SHM_levels_H)
df2$SHM_levels_H = factor(df2$SHM_levels_H,levels = c("Low","Median","High"))

plot_df = df2 %>% dplyr::group_by(predict.y_level,SHM_levels_H) %>% dplyr::summarise(n = n()) %>%
  dplyr::group_by(predict.y_level) %>% dplyr::mutate(sum_n = sum(n)) %>% 
  ungroup() %>% dplyr::mutate(percent = n/sum_n)

p2 = ggplot(plot_df,
            aes(x = predict.y_level, y = percent, fill = SHM_levels_H)) +
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
    axis.text.x = element_text(size = 10, angle = 90, hjust = 1, vjust = 0.5),
    axis.text.y = element_text(size = 10),
    text = element_text(size = 10),
    axis.line = element_line(size = 0.3),
    axis.ticks = element_line(size = 0.3),
    panel.border = element_rect(colour = "black", linewidth = 0.3),
    legend.position = "none"
  ) 


p3 <- ggplot(df2,
             aes(x = predict.y_level, y = H_shm)) +
  geom_boxplot(aes(color = predict.y_level),
               outlier.colour = NA, 
               lwd = 0.3) +
  geom_jitter(aes(color = predict.y_level), size = 0.7,shape = 16, stroke = 0, width = 0.1) +
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

ggsave(file.path(plots_dir, "FigureS7C.pdf"), width = 7, height = 4)

###############################################################################
#'                          Manuscipt: figureS7D                            '#
###############################################################################

## Figure S7D; 
## The atypical B cell encoded mAbs are grouped into non-neutralizing mAbs and neutralizing mAbs against any of the 8 H1N1 strains. 
## Box plot compared the SHM of the two groups of mAbs. 


plot_df = df2 %>%
  dplyr::count(predict.x_level, IGH_isotype) 

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
#df$Annotation <- 
p1 = ggplot(plot_df, aes(predict.x_level, n, fill = IGH_isotype)) +
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
    axis.text.x = element_text(size = 10, angle = 90, hjust = 1,vjust = 0.5),
    axis.text.y = element_text(size = 10),
    text = element_text(size = 10),
    axis.line = element_line(size = 0.3),
    axis.ticks = element_line(size = 0.3),
    panel.border = element_rect(colour = "black", linewidth = 0.3),
    legend.position = "none"
  ) + coord_flip()

plot_df = df2 %>% dplyr::group_by(predict.x_level,SHM_levels_H) %>% dplyr::summarise(n = n()) %>%
  dplyr::group_by(predict.x_level) %>% dplyr::mutate(sum_n = sum(n)) %>% 
  ungroup() %>% dplyr::mutate(percent = n/sum_n)

p2 = ggplot(plot_df,
            aes(x = predict.x_level, y = percent, fill = SHM_levels_H)) +
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
    axis.text.x = element_text(size = 10, angle = 90, hjust = 1, vjust = 0.5),
    axis.text.y = element_text(size = 10),
    text = element_text(size = 10),
    axis.line = element_line(size = 0.3),
    axis.ticks = element_line(size = 0.3),
    panel.border = element_rect(colour = "black", linewidth = 0.3),
    legend.position = "none"
  ) 

p3 <- ggplot(df2,
             aes(x = predict.x_level, y = H_shm)) +
  geom_boxplot(aes(color = predict.x_level),
               outlier.colour = NA, 
               lwd = 0.3) +
  geom_jitter(aes(color = predict.x_level), size = 0.7,shape = 16, stroke = 0, width = 0.1) +
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

ggsave(file.path(plots_dir, "FigureS7D.pdf"), width = 7, height = 4)

###########################################################
# 在文件末尾添加成功信息
cat("Figure 5 analysis completed successfully!\n")
cat("Output files saved to:", plots_dir, "\n")
cat("CSV files saved to:", files_dir, "\n")

# 记录运行环境
sink(file.path(output_dir, "session_info.txt"))  # 将输出重定向到文件
sessionInfo()
sink()  # 关闭重定向

