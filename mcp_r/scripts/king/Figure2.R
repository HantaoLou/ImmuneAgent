rm(list = ls())

################ Figure 2  ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript Figure2.R <input_rds_file>")
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
# 只加载RSV数据
rsv_obj <- readRDS(input_rds_file)

# 计算RSV结合平均值
rsv_obj$RSV_bind_average_values <- rowMeans(rsv_obj@meta.data[, c("output.x", "output.y")], na.rm = TRUE)

## III. load color
my36colors <-c('#E5D2DD', '#53A85F', '#F1BB72', '#F3B1A0', '#D6E7A3', '#57C3F3', '#476D87',
               '#E95C59', '#E59CC4', '#AB3282', '#23452F', '#BD956A', '#8C549C', '#585658',
               '#9FA3A8', '#E0D4CA', '#5F3D69', '#C5DEBA', '#58A4C3', '#E4C755', '#F7F398',
               '#AA9A59', '#E63863', '#E39A35', '#C1E6F3', '#6778AE', '#91D0BE', '#B53E2B',
               '#712820', '#DCC1DD', '#CCE0F5',  '#CCC9E6', '#625D9E', '#68A180', '#3A6963',
               '#968175'
)

flu_color_panel = c(
  'B.01.TCL1A+Bn' = '#53A85F',### naive
  'B.02.RGS7+Bm' = '#C1E6F3', ##IFN naive
  'B.03.S100A4+Bm' = '#CCC9E6', ### activate naive
  'B.04.TEX14+Bm' =  '#E4C755',
  'B.05.ITGB1+Bm' =  '#D6E7A3',
  'B.05.ITGB1+Bm' = '#57C3F3',
  'B.06.pre-GC' =  '#91D0BE',
  'B.07.Bgc_DZ-like' = '#E95C59',
  'B.08.Bgc_LZ-like' = '#E59CC4',
  'B.09.ITGAX+AtM' = '#E39A35',
  'B.10.CST7+AtM' = '#F3B1A0'
)

## IV. load marker
#### A1-A11 marker genes
markers_a1 = list(
  'B.01.TCL1A+Bn' = c("FCER2", "TCL1A", "IL4R",  "BACH2", "IGHD", "IGHM"), ## "CD72",'YBX3',
  'B.02.NR4A2+Bn'= c('NR4A2', "NR4A1",  'JUN','CD69',"CD83"), #"CREM",'VPS37B',
  'B.03.IFIT3+Bn' = c( "IFIT3","ISG15", "IFI44L", "IFI6"),
  'B.04.DUSP2+Bm' = c("DUSP2",'GPR183',"TNFSF9",'SOCS3','RGS2'), ##'ZNF804A','RGS16'  'CCR7',"CD27", "TNFRSF13B", 'TEX14','PTCH2','TNFSF9'
  'B.05.S100A4_Bm' = c('S100A4',"CD27",'TNFRSF13B','TBC1D9','VOPP1'),  ## 'ARHGAP24',
  'B.06.ITGB1+Bm' = c("S100A10",'PDE4D',"GSTK1","ITGB1",'TFRC'), ## ,"CR2","CD1C" 'COCH',"CRIP1",
  'B.08.CD1C+AtM' = c('LTB','CD24','CXCR5','MARCKS','CD1C'), ##'FCRL4' ,'CR2'
  'B.09.ITGAX+AtM'  = c('ITGAX',"ENC1","ZEB2","FCRL5","FCRL3","SOX5",'FGR','CEMIP2'), ##,'MACROD2'
  'B.10.Plasmablast' = c("JCHAIN","PRDM1","XBP1","MZB1")
  # 'IG-genes' = c("IGHG1", "IGHG2", "IGHG3", "IGHG4","IGHA1", "IGHA2",'IGHE')
)

###############################################################################
#'                          Manuscipt: figureS2A                            '#
###############################################################################

## Figure S2A; 
## UMAP - The B cell subsets of PBMC derived from Priest, et al

# 设置亚群标签为当前分析标识
Idents(rsv_obj) <- "CellType"

# 生成UMAP图展示合并后的亚群 (使用metadata中的UMAP坐标)
meta_data <- rsv_obj@meta.data
meta_data$cell_id <- rownames(meta_data)

p <- ggplot(meta_data, aes(x = UMAP_1, y = UMAP_2, color = CellType)) +
  geom_point(size = 0.5) +
  theme_classic() +
  labs(x = "UMAP_1", y = "UMAP_2") +
  guides(color = guide_legend(override.aes = list(size = 3))) +
  theme(plot.title = element_text(hjust = 0.5))

# print(p2)
ggsave(file.path(plots_dir, "Figure_S2A.pdf"), p, width = 10, height = 8)

###############################################################################
#'                    Manuscipt: figure2A/figureS2B                            '#
###############################################################################

## Figure 2A.1; 
## memory B cells highlighted by RSV binding cells.

# 跳过ADT分析，因为King数据中没有ADT数据
# 原始代码需要tetramer结合数据，这里无法实现
cat("Figure 2A.1 skipped: No ADT data for tetramer binding analysis in King dataset\n")

## Figure 2A.2; 
## Predition of binding breath against RSV variants

DefaultAssay(rsv_obj) <- 'RNA'
# Extract UMAP embeddings and metadata (from metadata columns)
meta_data <- rsv_obj@meta.data


meta_data$highlight <- "highlight"
meta_data$highlight[meta_data$RSV_bind_average_values == 0] <-  "normal"
p <- ggplot() +
  geom_point(data = meta_data[meta_data$highlight == "normal",], aes(x = UMAP_1, y = UMAP_2, color = RSV_bind_average_values), size = 0.8) +
  geom_point(data = meta_data[meta_data$highlight == "highlight",], alpha = ifelse(is.na(meta_data[meta_data$highlight == "highlight",]$RSV_bind_average_values), 0, 1), aes(x = UMAP_1, y = UMAP_2, color = RSV_bind_average_values), size = 0.8) +
  scale_color_gradientn(colors = c("transparent", "coral", "brown4"),
                        # scale_color_gradientn(colors = c("white", "coral", "brown4"), 
                        values = c(0, 0.5, 1),
                        breaks = c(0, 0.5, 1),
                        labels = c("0", "0.5", "1"),
                        name = "bind_CellScore") +
  theme_classic(base_size = 10) +
  theme(
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 8)
  )

ggsave(file.path(plots_dir, "Figure_2A2.pdf"), p, width = 10, height = 8)


## Figure S2B; 
## Skip neutralization prediction (not available in RSV data)

# 跳过中和预测部分，RSV数据中没有对应字段

###############################################################################
#'                    Manuscipt: figure2B                                    '#
###############################################################################

## Figure 2B.1; 
## the origin of sample

# 使用Status字段作为样本分组
meta_data <- rsv_obj@meta.data

p <- ggplot(meta_data, aes(x = UMAP_1, y = UMAP_2, color = Status)) +
  geom_point(size = 0.5) +
  scale_color_manual(values = my36colors[c(2,3,6)]) +
  theme_classic() +
  labs(x = "UMAP_1", y = "UMAP_2")
ggsave(
  file.path(plots_dir, "Figure_2B1.pdf"),plot = p,
  width = 10,
  height = 8
)

## Figure 2B.2; 
## the percentage broadly reactive B cells of mRNA vaccination, moderate or severe COVID

# 提取 metadata
metadata <- rsv_obj@meta.data
# 过滤掉 RSV_bind_average_values 的空值
metadata_filtered <- metadata %>%
  filter(!is.na(RSV_bind_average_values))
# 将 RSV_bind_average_values 分成 11 个等差的区间
metadata_filtered <- metadata_filtered %>%
  mutate(bind_interval = cut(
    RSV_bind_average_values,
    breaks = seq(0, 1, by = 0.1),
    include.lowest = TRUE,
    right = FALSE
  ))
# 按 Status 和 CellType 分组，统计每个区间的比例
bind_summary <- metadata_filtered %>%
  group_by(Status, CellType, bind_interval) %>%
  summarise(
    cell_count = n(),
    .groups = "drop"
  ) %>%
  group_by(Status, CellType) %>%
  mutate(proportion = cell_count / sum(cell_count))
# 绘制堆叠柱状图
p<-ggplot(bind_summary, aes(x = CellType, y = proportion, fill = bind_interval)) +
  geom_bar(stat = "identity", position = "stack") +
  facet_wrap(~ Status) +
  labs(
    title = "Proportion of RSV_bind_average_values in each interval",
    x = "CellType",
    y = "Proportion",
    fill = "Bind Interval"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


ggsave(
  file.path(plots_dir, "Figure_2B2.pdf"),
  plot = p,
  width = 6,
  height = 5
)

###############################################################################
#'                    Manuscipt: figureS2C                                   '#
###############################################################################

## Figure S2C; 
## Dot plot of marker genes relating to the cell types


# 过滤掉不存在的基因
# 先将列表转换为向量，然后找出存在的基因
all_markers <- unlist(markers_a1)
available_markers <- intersect(all_markers, rownames(rsv_obj))

p <- DotPlot(object = rsv_obj, features = available_markers, scale = T) + 
  scale_colour_gradientn(colors=brewer.pal(9, "YlGnBu")) + theme_bw() +
  theme(axis.text.x = element_text(angle = 90)) 

ggsave(file.path(plots_dir, "Figure_S2C.pdf"), p, width=16 ,height=8)


###############################################################################
#'                          Manuscipt: figure2C                              '#
###############################################################################

## Figure 2C;  ??? 
## Volcano plot of the differentially expressed genes between broadly reactive AtM B cells and the non-reactive B cells 

# setting 15 ：跟≥5个抗原结合的ITGAX+AtM 细胞 vs 跟=1个抗原结合的MB细胞做DEG
# setting 11 ：跟≥5个抗原结合的ITGAX+AtM 细胞 vs 跟=1个抗原结合的ALL细胞做DEG

# 去除merge不到BCR的细胞
sub_obj <- rsv_obj[,!is.na(rsv_obj$output.x)]
sub_obj$RSV_bind_average_values <- rowMeans(sub_obj@meta.data[, c("output.x", "output.y")], na.rm = TRUE)

# 基于King数据的实际分布调整阈值：只有0和1，没有0.5
# 使用0.5作为阈值来区分高中低结合
i = 0.5
sub_obj$RSV_bind_level <- ifelse(sub_obj$RSV_bind_average_values>=i,2,0)
sub_obj$RSV_bind_level[sub_obj$RSV_bind_average_values > 0 & sub_obj$RSV_bind_average_values < i] <- 1

# 
j = 11
sub_obj$DEG_level <- 2
sub_obj$DEG_level[(sub_obj$CellType %in% c('MBC FCRL4+'))
              & (sub_obj$RSV_bind_level == 2)] <- 1
sub_obj$DEG_level[sub_obj$RSV_bind_level == 1] <- 0

# 检查是否有足够的细胞进行比较
if(sum(sub_obj$DEG_level == 1) > 0 & sum(sub_obj$DEG_level == 0) > 0) {
  Idents(sub_obj) <- 'DEG_level'
  markers <- FindMarkers(sub_obj, ident.1 = "1", ident.2 = "0",
                         logfc.threshold=0, min.pct=0.2)
} else {
  markers <- data.frame(avg_log2FC = numeric(0), p_val_adj = numeric(0))
}

if(nrow(markers) > 0) {
  markers$p_val_adj = p.adjust(markers$p_val, method='fdr')
  markers$pct_dif <- markers$pct.1 - markers$pct.2
  # markers$feature <- marker_anno_df$feature[match(rownames(markers),marker_anno_df$gene)]
  write.csv(markers,file.path(files_dir, paste0("RSV_bind_level_marker_threshold=",i,"_setting=",j,".csv")))
}


# 
j = 15
sub_obj$DEG_level <- 2
sub_obj$DEG_level[(sub_obj$CellType %in% c('MBC FCRL4+'))
              & (sub_obj$RSV_bind_level == 2)] <- 1
sub_obj$DEG_level[(sub_obj$CellType %in% c('MBC',
                                         'MBC FCRL4+'
))
& (sub_obj$RSV_bind_level == 1)] <- 0

# 检查是否有足够的细胞进行比较
if(sum(sub_obj$DEG_level == 1) > 0 & sum(sub_obj$DEG_level == 0) > 0) {
  Idents(sub_obj) <- 'DEG_level'
  markers <- FindMarkers(sub_obj, ident.1 = "1", ident.2 = "0",
                         logfc.threshold=0, min.pct=0.2)
  markers$p_val_adj = p.adjust(markers$p_val, method='fdr')
  markers$pct_dif <- markers$pct.1 - markers$pct.2
  # markers$feature <- marker_anno_df$feature[match(rownames(markers),marker_anno_df$gene)]
  write.csv(markers,file.path(files_dir, paste0("RSV_bind_level_marker_threshold=",i,"_setting=",j,".csv")))
} else {
  markers <- data.frame(avg_log2FC = numeric(0), p_val_adj = numeric(0))
}

# 生成火山图（只在有差异基因结果时）
if(nrow(markers) > 0) {
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
  # p <- p + theme(panel.grid.major = element_blank(),
  #                panel.grid.minor = element_blank())


  ggsave(file.path(plots_dir, "Figure_2C.pdf"), p, width=8,height=6,dpi = 300)
}

###############################################################################
#'                       Manuscipt: figure2I_RSV                             '#
###############################################################################

## Figure 2I (adapted for RSV subtypes);
## Correlation plot of the differentially expressed genes by broadly reactive AtM B cells for RSV-A versus RSV-B

# 去除merge不到BCR的细胞
sub <- rsv_obj[,!is.na(rsv_obj$output.x)]

# RSV-A分析 (output.x)
# 适应King数据的实际分布：只有0和1
i = 0.5
sub$RSVA_bind_level <- ifelse(sub$output.x>=i,2,0)
sub$RSVA_bind_level[sub$output.x > 0 & sub$output.x < i] <- 1

# 
j = 15
sub$DEG_level <- 2
sub$DEG_level[(sub$CellType %in% c('MBC FCRL4+'))
              & (sub$RSVA_bind_level == 2)] <- 1
# 使用低结合细胞作为对照组（适应King数据没有中等结合的情况）
sub$DEG_level[(sub$CellType %in% c('MBC',
                                   'MBC FCRL4+'
))
& (sub$RSVA_bind_level == 0)] <- 0

# 检查是否有足够的细胞进行比较
if(sum(sub$DEG_level == 1) > 0 & sum(sub$DEG_level == 0) > 0) {
  Idents(sub) <- 'DEG_level'
  markers <- FindMarkers(sub, ident.1 = "1", ident.2 = "0",
                         logfc.threshold=0, min.pct=0.2)
  markers$p_val_adj = p.adjust(markers$p_val, method='fdr')
  markers$pct_dif <- markers$pct.1 - markers$pct.2
  write.csv(markers,file.path(files_dir, paste0("20250227-B-RSVA_bind_level_cluster_marker_threshold=",i,"_setting=",j,".csv")))
} else {
  cat("Warning: Not enough cells for RSV-A comparison with threshold=",i," setting=",j,"\n")
}

# 
j = 11
sub$DEG_level <- 2
sub$DEG_level[(sub$CellType %in% c('MBC FCRL4+'))
              & (sub$RSVA_bind_level == 2)] <- 1
# 使用低结合细胞作为对照组
sub$DEG_level[sub$RSVA_bind_level == 0] <- 0

# 检查是否有足够的细胞进行比较
if(sum(sub$DEG_level == 1) > 0 & sum(sub$DEG_level == 0) > 0) {
  Idents(sub) <- 'DEG_level'
  markers <- FindMarkers(sub, ident.1 = "1", ident.2 = "0",
                         logfc.threshold=0, min.pct=0.2)
  markers$p_val_adj = p.adjust(markers$p_val, method='fdr')
  markers$pct_dif <- markers$pct.1 - markers$pct.2
  write.csv(markers,file.path(files_dir, paste0("20250227-B-RSVA_bind_level_cluster_marker_threshold=",i,"_setting=",j,".csv")))
} else {
  cat("Warning: Not enough cells for RSV-A comparison with threshold=",i," setting=",j,"\n")
}

# RSV-B分析 (output.y)
sub$RSVB_bind_level <- ifelse(sub$output.y>=i,2,0)
sub$RSVB_bind_level[sub$output.y > 0 & sub$output.y < i] <- 1

# 
j = 15
sub$DEG_level <- 2
sub$DEG_level[(sub$CellType %in% c('MBC FCRL4+'))
              & (sub$RSVB_bind_level == 2)] <- 1
# 使用低结合细胞作为对照组
sub$DEG_level[(sub$CellType %in% c('MBC',
                                   'MBC FCRL4+'
))
& (sub$RSVB_bind_level == 0)] <- 0

# 检查是否有足够的细胞进行比较
if(sum(sub$DEG_level == 1) > 0 & sum(sub$DEG_level == 0) > 0) {
  Idents(sub) <- 'DEG_level'
  markers <- FindMarkers(sub, ident.1 = "1", ident.2 = "0",
                         logfc.threshold=0, min.pct=0.2)
  markers$p_val_adj = p.adjust(markers$p_val, method='fdr')
  markers$pct_dif <- markers$pct.1 - markers$pct.2
  write.csv(markers,file.path(files_dir, paste0("20250227-B-RSVB_bind_level_cluster_marker_threshold=",i,"_setting=",j,".csv")))
} else {
  cat("Warning: Not enough cells for RSV-B comparison with threshold=",i," setting=",j,"\n")
}

# 
j = 11
sub$DEG_level <- 2
sub$DEG_level[(sub$CellType %in% c('MBC FCRL4+'))
              & (sub$RSVB_bind_level == 2)] <- 1
# 使用低结合细胞作为对照组
sub$DEG_level[sub$RSVB_bind_level == 0] <- 0

# 检查是否有足够的细胞进行比较
if(sum(sub$DEG_level == 1) > 0 & sum(sub$DEG_level == 0) > 0) {
  Idents(sub) <- 'DEG_level'
  markers <- FindMarkers(sub, ident.1 = "1", ident.2 = "0",
                         logfc.threshold=0, min.pct=0.2)
  markers$p_val_adj = p.adjust(markers$p_val, method='fdr')
  markers$pct_dif <- markers$pct.1 - markers$pct.2
  write.csv(markers,file.path(files_dir, paste0("20250227-B-RSVB_bind_level_cluster_marker_threshold=",i,"_setting=",j,".csv")))
} else {
  cat("Warning: Not enough cells for RSV-B comparison with threshold=",i," setting=",j,"\n")
}

##### correlation figure2I (RSV-A vs RSV-B) #####

# 初始化一个空数据框
degs <- data.frame()
RSVA_i <- 0.5
RSVB_i <- 0.5

# 循环读取不同设置下的数据
for (j in c(11, 15)) {
  # 检查文件是否存在并读取RSV-A DEG 结果
  rsva_file <- file.path(files_dir, paste0("20250227-B-RSVA_bind_level_cluster_marker_threshold=",RSVA_i,"_setting=",j,".csv"))
  rsvb_file <- file.path(files_dir, paste0("20250227-B-RSVB_bind_level_cluster_marker_threshold=",RSVB_i,"_setting=",j,".csv"))
  
  if(file.exists(rsva_file) & file.exists(rsvb_file)) {
    # 读取RSV-A DEG 结果
    data_RSVA <- read.csv(rsva_file, row.names = 1)
    data_RSVA <- data_RSVA %>% filter(p_val_adj < 0.05)
    
    # 读取RSV-B DEG 结果
    data_RSVB <- read.csv(rsvb_file, row.names = 1)
    data_RSVB <- data_RSVB %>% filter(p_val_adj < 0.05)
    
    # 找到共同的基因
    genes <- intersect(rownames(data_RSVA), rownames(data_RSVB))
    cat(j, "num=", length(genes), "\n")
    
    if(length(genes) > 0){
      # 合并数据
      markers <- merge(data_RSVA[genes, c("avg_log2FC", "pct.1", "pct.2", "p_val_adj")],
                       data_RSVB[genes, c("avg_log2FC", "pct.1", "pct.2", "p_val_adj" )],
                       by = "row.names", all = TRUE, suffixes = c("_RSVA", "_RSVB"))
      colnames(markers)[1] <- "gene"
      markers$setting <- j
      
      # 添加到总数据框
      degs <- bind_rows(degs, markers)
    }
  } else {
    cat("Warning: Missing files for correlation analysis with setting=", j, "\n")
  }
  
}

# 遍历特定设置生成图表和分析（图为setting 11）
if(nrow(degs) > 0) {
  for (i in c(11)) {
    df <- degs %>% filter(setting == i) %>% arrange(desc(avg_log2FC_RSVA))
    
    if(nrow(df) > 0) {
      # 提取显著的基因
      # sig <- bind_rows(list(head(df, 20), tail(df, 20)))
      # 标注定制化基因
      genes <- c('ITGAX','FGR','FCRL4','FCRL5','CD68','TNFRSF1B',
                 'JCHAIN','MZB1','XBP1','MARCKSL1')
      sig <- intersect(genes,c(degs$gene))
      print(i)
      print(setdiff(genes,sig))
      # 绘制散点图
      p <- ggplot(df, aes(x = avg_log2FC_RSVA, y = avg_log2FC_RSVB)) +
        geom_point(color = "black", size = 0.5) +
        geom_hline(yintercept = 0, color = "blue", linetype = "dashed") +
        geom_vline(xintercept = 0, color = "blue", linetype = "dashed") +
        # geom_text(data = df[df$gene %in% sig,], aes(label = gene), color = "purple", hjust = -1, vjust = -1, size = 5) +
        geom_text_repel(data = df[df$gene %in% sig,], aes(label = gene), color = "purple", size = 5) +
        labs(title = "Scatter Plot between RSV-A and RSV-B", x = "RSV-A", y = "RSV-B") +
        theme_minimal() +
        theme(
          panel.grid.major = element_blank(),  # 去掉主网格线
          panel.grid.minor = element_blank(),  # 去掉次网格线
          panel.border = element_rect(color = "black", fill = NA, size = 0.5),  # 添加边框
          plot.background = element_rect(color = "black", size = 0.5)  # 添加图表背景边框
        )
      
      
      ggsave(file.path(plots_dir, "Figure_2I_RSV.pdf"), p, width = 4, height = 3.6)
     
      # 计算相关性
      correlation_test <- cor.test(df$avg_log2FC_RSVA, df$avg_log2FC_RSVB, method = "pearson")
      cat("Correlation coefficient between RSV-A and RSV-B:", correlation_test$estimate, "\n")
      cat("P-value:", correlation_test$p.value, "\n")
    } else {
      cat("Warning: No data available for correlation analysis with setting=", i, "\n")
    }
  }
} else {
  cat("Warning: No correlation data available for RSV-A vs RSV-B analysis\n")
}

# 在文件末尾添加成功信息
cat("Figure 2 analysis completed successfully!\n")
cat("Output files saved to:", plots_dir, "\n")
cat("CSV files saved to:", files_dir, "\n")

# 记录运行环境
sink(file.path(output_dir, "session_info_RSV.txt"))
sessionInfo()
sink()

