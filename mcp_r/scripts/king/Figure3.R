rm(list = ls())

################ Figure 3  ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript Figure3.R <input_rds_file>")
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
output_dir <- file.path(fdir, "output", "Figure3")
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
# 统一使用King RSV数据
king_obj <- readRDS(input_rds_file)
flu_obj <- king_obj  # 用于RSV预测分析（原流感分析位置）
a1a11_obj <- king_obj  # 用于细胞类型分析（原A1-A11分析位置）

## III. load color
my36colors <-c('#E5D2DD', '#53A85F', '#F1BB72', '#F3B1A0', '#D6E7A3', '#57C3F3', '#476D87',
               '#E95C59', '#E59CC4', '#AB3282', '#23452F', '#BD956A', '#8C549C', '#585658',
               '#9FA3A8', '#E0D4CA', '#5F3D69', '#C5DEBA', '#58A4C3', '#E4C755', '#F7F398',
               '#AA9A59', '#E63863', '#E39A35', '#C1E6F3', '#6778AE', '#91D0BE', '#B53E2B',
               '#712820', '#DCC1DD', '#CCE0F5',  '#CCC9E6', '#625D9E', '#68A180', '#3A6963',
               '#968175'
)

a1a11_color_panel = c(
  'B.01.TCL1A+Bn' = '#53A85F',### naive
  'B.02.NR4A2+Bn' = '#C1E6F3', ##IFN naive
  'B.03.IFIT3+Bn' = '#CCC9E6', ### activate naive
  'B.04.DUSP2+Bm' =  '#E4C755',
  'B.05.S100A4+Bm' =  '#D6E7A3',
  'B.06.ITGB1+Bm' = '#57C3F3',
  'B.07.IFIT3+Bm' =  '#3A6963',
  'B.08.CD1C+AtM' = '#E95C59',
  'B.09.ITGAX+AtM' = '#E59CC4',
  'B.10.Plasmablast' = '#B53E2B'
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
#'                          Manuscipt: figure3a                              '#
###############################################################################

## Figure 3A; 
## UMAP - Prediction of RSV binding (adapted from H1N1 neutralization)

sub <- flu_obj[,!is.na(flu_obj$output.x)]
# 清理重复列名
sub@meta.data <- sub@meta.data[, !duplicated(colnames(sub@meta.data))]

new_data <- sub@meta.data[,c('output.x','predict.x')]
new_data <- t(new_data)
sub[['prediction']] <- CreateAssayObject(counts = as.matrix(new_data))

# 创建UMAP reduction对象（使用King数据的UMAP坐标）
umap_coords <- sub@meta.data[, c("UMAP_1", "UMAP_2")]
colnames(umap_coords) <- c("UMAP_1", "UMAP_2")
sub[['umap']] <- CreateDimReducObject(embeddings = as.matrix(umap_coords), key = "UMAP_")

DefaultAssay(sub) <- 'prediction'
p <- ggrastr::rasterize(Nebulosa::plot_density(sub, 
                                               "predict.x",   
                                               size=0.2,reduction='umap'), dpi=300)
ggsave(
  file.path(plots_dir, "Figure_3A.pdf"),plot = p,
  width = 5,
  height = 4
)



###############################################################################
#'                          Manuscipt: figure3C                              '#
###############################################################################

## Figure 3C; 
## UMAP - B cells from King tonsil cohort (adapted from ADV vaccines cohort)

meta = a1a11_obj@meta.data
# 清理重复列名
meta <- meta[, !duplicated(colnames(meta))]
# 使用King数据的UMAP坐标
meta$umap_1 <- meta$UMAP_1
meta$umap_2 <- meta$UMAP_2
meta$CellType <- as.factor(meta$CellType)

p <- ggplot(meta, aes(x = umap_1, y = umap_2, color = CellType)) +
  geom_point(size = 0.15,
             shape = 16,
             stroke = 0) +
  theme_void()+
  scale_color_manual(values = my36colors, name = '')+
  theme(aspect.ratio = 1,
        legend.position = "")



ggsave(
  file.path(plots_dir, "Figure_3C.pdf"),plot = p,
  width = 2,
  height = 2,
  dpi = 300
)


###############################################################################
#'                          Manuscipt: figure3D                              '#
###############################################################################

## Figure 3D; 
## Dotplot representation of marker expression across different B cell subsets. 

# 过滤掉不存在的基因
all_markers <- unlist(markers_a1)
available_markers <- intersect(all_markers, rownames(a1a11_obj))

p <- DotPlot(object = a1a11_obj, features = available_markers, scale = T) + 
  scale_colour_gradientn(colors=brewer.pal(9, "YlGnBu")) + theme_bw() +
  theme(axis.text.x = element_text(angle = 90)) 

ggsave(file.path(plots_dir, "Figure_3D.pdf"), plot = p,width = 18,height = 6)


###############################################################################
#'                          Manuscipt: figure3F                              '#
###############################################################################

## Figure 3F; 
## Correlation of differentially expressed genes by predicted broadly reactive AtM B cells from RSV-A vs RSV-B subtypes (adapted from SARS-CoV-2 vs QIV)

##### correlation  figure3F  #####

# 初始化一个空数据框
degs <- data.frame()
rsva_i <- 0.5
rsvb_i <- 0.5

# 循环读取不同设置下的数据(DEG结果文件已在Figure2.R中生成)
for (j in c(15)) {
  # 检查文件是否存在
  figure2_files_dir <- file.path(dirname(fdir), "output", "Figure2", "files")
  rsva_file <- file.path(figure2_files_dir, paste0("20250227-B-RSVA_bind_level_cluster_marker_threshold=",rsva_i,"_setting=",j,".csv"))
  rsvb_file <- file.path(figure2_files_dir, paste0("20250227-B-RSVB_bind_level_cluster_marker_threshold=",rsvb_i,"_setting=",j,".csv"))
  
  if(file.exists(rsva_file) & file.exists(rsvb_file)) {
    # 读取RSV-A数据
    data_rsva <- read.csv(rsva_file, row.names = 1)
    data_rsva <- data_rsva %>% filter(p_val_adj < 0.05)
    
    # 读取RSV-B数据
    data_rsvb <- read.csv(rsvb_file, row.names = 1)
    data_rsvb <- data_rsvb %>% filter(p_val_adj < 0.05)
    
    # 找到共同的基因
    genes <- intersect(rownames(data_rsva), rownames(data_rsvb))
    cat(j, "num=", length(genes), "\n")
    
    if(length(genes) > 0){
      # 合并数据
      markers <- merge(data_rsva[genes, c("avg_log2FC", "pct.1", "pct.2", "p_val_adj")],
                       data_rsvb[genes, c("avg_log2FC", "pct.1", "pct.2", "p_val_adj" )],
                       by = "row.names", all = TRUE, suffixes = c("_rsva", "_rsvb"))
      colnames(markers)[1] <- "gene"
      markers$setting <- j
      
      # 添加到总数据框
      degs <- bind_rows(degs, markers)
    }
  } else {
    cat("Warning: Missing DEG files for RSV correlation analysis with setting=", j, "\n")
  }
  
}

# 遍历特定设置生成图表和分析 
if(nrow(degs) > 0) {
  for (i in c(15)) {
    df <- degs %>% filter(setting == i) %>% arrange(desc(avg_log2FC_rsva))
    
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
      p <- ggplot(df, aes(x = avg_log2FC_rsva, y = avg_log2FC_rsvb)) +
        geom_point(color = "black", size = 0.5) +
        geom_hline(yintercept = 0, color = "blue", linetype = "dashed") +
        geom_vline(xintercept = 0, color = "blue", linetype = "dashed") +
        # geom_text(data = sig, aes(label = gene), color = "red", hjust = 0.5, vjust = -1.5, size = 3) +
        geom_text_repel(data = df[df$gene %in% sig,], aes(label = gene), color = "purple", size = 5) +
        labs(title = "Scatter Plot between RSV-A and RSV-B", x = "RSV-A", y = "RSV-B") +
        theme_minimal() +
        theme(
          panel.grid.major = element_blank(),  # 去掉主网格线
          panel.grid.minor = element_blank(),  # 去掉次网格线
          panel.border = element_rect(color = "black", fill = NA, size = 0.5),  # 添加边框
          plot.background = element_rect(color = "black", size = 0.5)  # 添加图表背景边框
        )
      
      
      # 保存图表
      ggsave(file.path(plots_dir, "Figure_3F.pdf"), p, width = 4, height = 3.6)
      
      # 计算相关性
      correlation_test <- cor.test(df$avg_log2FC_rsva, df$avg_log2FC_rsvb, method = "pearson")
      cat("Correlation coefficient between RSV-A and RSV-B:", correlation_test$estimate, "\n")
      cat("P-value:", correlation_test$p.value, "\n")
    } else {
      cat("Warning: No data available for correlation analysis with setting=", i, "\n")
    }
  }
} else {
  cat("Warning: No correlation data available for RSV-A vs RSV-B analysis\n")
}


###############################################################################
#'                          Manuscipt: figure3G                              '#
###############################################################################

## Figure 3G; 
## UMAP - Prediction of RSV binding (adapted from SARS-2 neutralization)

#### broad binding ####
# Extract UMAP embeddings and metadata
meta_data <- a1a11_obj@meta.data
# 清理重复列名
meta_data <- meta_data[, !duplicated(colnames(meta_data))]
# 使用King数据的UMAP坐标
meta_data$umap_1 <- meta_data$UMAP_1
meta_data$umap_2 <- meta_data$UMAP_2

# 计算RSV平均结合值
meta_data$RSV_bind_average_values <- rowMeans(meta_data[, c("output.x", "output.y")], na.rm = TRUE)

meta_data$highlight <- "highlight"
meta_data$highlight[meta_data$RSV_bind_average_values == 0] <-  "normal"
p <- ggplot(na.rm = TRUE, ) +
  geom_point(data = meta_data[meta_data$highlight == "normal",], aes(x = umap_1, y = umap_2, color = RSV_bind_average_values), size = 0.4) +
  geom_point(data = meta_data[meta_data$highlight == "highlight",], alpha = ifelse(is.na(meta_data[meta_data$highlight == "highlight",]$RSV_bind_average_values), 0, 1), aes(x = umap_1, y = umap_2, color = RSV_bind_average_values), size = 0.4) +
  scale_color_gradientn(colors = c("transparent", "coral", "brown4"),
                        values = c(0, 0.5, 1),
                        breaks = c(0, 0.5, 1),
                        labels = c("0", "0.5", "1"),
                        name = "bind_CellScore") +
  theme_classic(base_size = 10) +
  theme(
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 8)
  )

ggsave(file.path(plots_dir, "Figure_3G.pdf"),plot = p,width = 7,height = 6)

###########################################################
# 在文件末尾添加成功信息
cat("Figure 3 analysis completed successfully!\n")
cat("Output files saved to:", plots_dir, "\n")
cat("CSV files saved to:", files_dir, "\n")

# 记录运行环境
sink(file.path(output_dir, "session_info_RSV.txt"))  # 将输出重定向到文件
sessionInfo()
sink()  # 关闭重定向

