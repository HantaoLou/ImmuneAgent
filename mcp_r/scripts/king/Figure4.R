rm(list = ls())

################ Figure 4  ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript Figure4.R <input_rds_file>")
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
output_dir <- file.path(fdir, "output", "Figure4")
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
library(reshape2)  # For melt function
library(gridExtra) # For grid.arrange function

## II. load data
# Load single RSV RDS file
rsv_obj <- readRDS(input_rds_file)

# Create unified RSV analysis objects for dual pathway analysis
rsv_path1_obj <- rsv_obj
rsv_path1_obj$annotation_final <- rsv_obj$CellType
rsv_path1_obj$main_name <- rsv_obj$barcode

rsv_path2_obj <- rsv_obj  
rsv_path2_obj$annotation_final <- rsv_obj$CellType
rsv_path2_obj$main_name <- rsv_obj$barcode

## III. load cells of path1 or path2 (RSV analysis paths)
# RSV Path1: Naive -> Activated -> preGC -> GC -> prePB (浆母细胞路径)
rsv_path1_trajectory1 <- c("Naive","Activated","preGC","GC","prePB")
rsv_path1_trajectory2 <- c("Naive","Activated","preGC","GC","MBC")

# RSV Path2: Naive -> Activated -> preGC -> GC -> MBC (记忆B细胞路径)
rsv_path2_trajectory1 <- c("Naive","Activated","preGC","GC","prePB")
rsv_path2_trajectory2 <- c("Naive","Activated","preGC","GC","MBC")

# Note: We'll generate cds_subset objects during the analysis instead of loading pre-computed ones

## IV. load SHM counts - create from RSV data using real SHM data
# Create SHM data from sequence identity (SHM rate = 100 - identity)
rsv_path1_shm <- data.frame(
  main_name = rsv_obj$barcode,
  len.H_shm. = 100 - rsv_obj$v_identity_IGH.x,  # 重链SHM率
  len.L_shm. = 100 - rsv_obj$v_identity_IGL.x,  # 轻链SHM率
  stringsAsFactors = FALSE
)

# Create rsv_path2_shm from RSV data (using .y dimension for consistency)
rsv_path2_shm <- data.frame(
  main_name = rsv_obj$barcode,
  len.H_shm. = 100 - rsv_obj$v_identity_IGH.y,  # 重链SHM率
  len.L_shm. = 100 - rsv_obj$v_identity_IGL.y,  # 轻链SHM率
  stringsAsFactors = FALSE
)


## V. load color

rsv_color_panel = c(
  "Naive" = "#53A85F",
  "Activated" = "#C1E6F3",
  "preGC" = "#E4C755",
  "GC" = "#D6E7A3",
  "prePB" = "#E95C59",
  "MBC" = "#E59CC4",
  "DZ GC" = "#57C3F3",
  "Cycling" = "#CCC9E6",
  "LZ GC" = "#3A6963",
  "MBC FCRL4+" = "#B53E2B",
  "FCRL2/3high GC" = "#8C564B"
)

###############################################################################
#'                          Manuscipt: figure4A                              '#
###############################################################################

## Figure 4A; 
## UMAP - the pseudotime trajectory analysis of the single-cell RNA sequencing (scRNA-seq) data from RSV (Path1: predict.x)

sub <- rsv_path1_obj
data_m <- sub@meta.data

###monocle3
data_m <- GetAssayData(sub,assay = "RNA",layer = "counts")
data_m <- data_m[rowSums(data_m>0)>=3,]

cell_metadata <- sub@meta.data
gene_annotation <- data.frame(gene_short_name=rownames(data_m))
rownames(gene_annotation) <- rownames(data_m)

cds <- new_cell_data_set(data_m,
                        cell_metadata = cell_metadata,
                       gene_metadata = gene_annotation)
cds <- preprocess_cds(cds,num_dim = 50)
cds <- reduce_dimension(cds,preprocess_method = "PCA")

cds.embed <- cds@int_colData$reducedDims$UMAP
# Get UMAP coordinates from metadata
umap_coords <- as.matrix(sub@meta.data[, c("UMAP_1", "UMAP_2")])
rownames(umap_coords) <- colnames(sub)
int.embed <- umap_coords[rownames(cds.embed),]
cds@int_colData$reducedDims$UMAP <- int.embed

cds <- cluster_cells(cds,resolution = 0.001,k=40,random_seed=18,verbose=T)
cds@clusters$UMAP$clusters <- sub$annotation_final

cds <- learn_graph(cds, verbose =T,
                   use_partition=T,close_loop=F,learn_graph_control=
                     list(minimal_branch_len=30,rann.k=10)  )

# 自动选择起始细胞（Naive B细胞）
naive_cells <- colnames(cds)[cds@colData$annotation_final == "Naive"]
cds <- order_cells(cds, root_cells = naive_cells[1:min(10, length(naive_cells))])
# 绘图
p <- plot_cells(cds = cds,
                 color_cells_by = "pseudotime",
                 show_trajectory_graph = F,
                 trajectory_graph_color = "white",
                 trajectory_graph_segment_size = 0.5,
                 graph_label_size = 2,
                 cell_size = 1,
                 label_cell_groups = F,
                 label_groups_by_cluster = F,
                 label_branch_points = F,
                 label_roots = F,
                 label_leaves = F) 

ggsave(file.path(plots_dir, "Figure_4A.pdf"),width = 9,height = 8)
save(cds,file = file.path(files_dir, 'RSV_path1_B_monocle_cds.RData'))


###############################################################################
#'                          Manuscipt: figure4B                              '#
###############################################################################

## Figure 4B; 
## UMAP - the pseudotime trajectory analysis of the single-cell RNA sequencing (scRNA-seq) data from RSV (Path2: predict.y)

sub <- rsv_path2_obj[,rsv_path2_obj$annotation_final != 'Plasmablast']
data_m <- sub@meta.data
###monocle3
data_m <- GetAssayData(sub,assay = "RNA",layer = "counts")
data_m <- data_m[rowSums(data_m>0)>=3,]

cell_metadata <- sub@meta.data
gene_annotation <- data.frame(gene_short_name=rownames(data_m))
rownames(gene_annotation) <- rownames(data_m)

cds <- new_cell_data_set(data_m,
                            cell_metadata = cell_metadata,
                            gene_metadata = gene_annotation)
cds <- preprocess_cds(cds,num_dim = 50)

cds <- reduce_dimension(cds,preprocess_method = "PCA")

cds.embed <- cds@int_colData$reducedDims$UMAP
# Get UMAP coordinates from metadata
umap_coords <- as.matrix(sub@meta.data[, c("UMAP_1", "UMAP_2")])
rownames(umap_coords) <- colnames(sub)
int.embed <- umap_coords[rownames(cds.embed),]
cds@int_colData$reducedDims$UMAP <- int.embed

cds <- cluster_cells(cds)
sum(colnames(cds) != colnames(sub))
cds@clusters$UMAP$clusters <- sub$annotation_final
cds <- learn_graph(cds, verbose =T,
                      use_partition=T,close_loop=F,learn_graph_control=
                        list(minimal_branch_len=10,rann.k=15)  )
# 自动选择起始细胞（Naive B细胞）
naive_cells <- colnames(cds)[cds@colData$annotation_final == "Naive"]
cds <- order_cells(cds, root_cells = naive_cells[1:min(10, length(naive_cells))])
# 绘图
p <- plot_cells(cds = cds,
                color_cells_by = "pseudotime",
                show_trajectory_graph = F,
                trajectory_graph_color = "white",
                trajectory_graph_segment_size = 0.5,
                graph_label_size = 2,
                cell_size = 1,
                label_cell_groups = F,
                label_groups_by_cluster = F,
                label_branch_points = F,
                label_roots = F,
                label_leaves = F) 

ggsave(file.path(plots_dir, "Figure_4B.pdf"),width = 9,height = 8)
save(cds,file = file.path(files_dir, 'RSV_path2_B_monocle_cds.RData'))

###############################################################################
#'                          Manuscipt: figure4C                              '#
###############################################################################

## Figure 4C; 
## Barplot displaying the maturation stage of the B cells from RSV analysis


df <- data.frame(pseudotime=pseudotime(cds),
                  celltype=cds@colData$annotation_final)
P <- ggplot(df,aes(y=celltype,x=pseudotime,color=celltype))+geom_boxplot()+
  theme_bw()+scale_color_manual(values = rsv_color_panel)
ggsave(file.path(plots_dir, "Figure_4C.pdf"),width = 9,height = 6)



###############################################################################
#'                     Manuscipt: figure4D/E/F//G                            '#
###############################################################################

## Figure 4D/E/F//G 
## polynomial - RSV analysis with dual paths (predict.x and predict.y) 


## 1. 计算feature score
high_affinity=c("BATF","GARS","GART","LER3","MIF","MYC","SPP1","UCK2",
                "CD320","TIMD2","TNFRSF8",
                "AURKA","BUB1","CCNA2","CCNB1","CCNB2","CCND2","CDC20","CDC25C","KIF22","PLK1",
                "NFIL3","PML")
Low_affinity=c("PPP1R15A","RGS1","CCR6","CD22","CD38","CD72","FCER2A","ICOSL","PACAM1","SIGLECG","TLRL","TNFRSF18",
               "BACH2","EGR3","ELK4","FOXP1","JUN","NR4A1","REL")
exhaustion_genes = c('PDCD1','CD160','FASLG','CD244','LAG3','TNFRSF1B','CCR5','CCL3',
                     'CCL4','CXCL10','CTLA4','LGALS1','LGALS3','PTPN13','RGS16','ISG20',
                     'MX1','IRF4','EOMES','PBX3','NFATC1','NR4A2','CKS2','GAS2',
                     'ENTPD1','CA2',"CD52", "APOE", "PTGDS", "PIM2", "DERL3")  # 移除了PTLP


Bactivated_genes=c("CD69","CD83","IER2","DUSP2","IL6","NR4A2","JUN","CCR7","GPR183")
BCSR_genes=c("APEX1","APEX2","XRCC5","XRCC6","POLD2","AICDA")
CSR_m=c("APEX1","XRCC5","XRCC6","POLD2","POLE3", #CSR machinery
        "NCL","NME2","DDX21",#IgH locus
        "NPM1","SERBP1",#CSR interactors
        "MIR155HG","HSP90AB1",#AICDA/AICDA stability
        "BATF","HIVEP3","BHLHE40","IRF4")#TF

feature <- list(high_affinity=high_affinity,
                Low_affinity=Low_affinity,
                exhaustion_genes=exhaustion_genes,
                Bactivated_genes=Bactivated_genes,
                BCSR_genes=BCSR_genes,
                CSR_m=CSR_m)

rsv_path1_obj <- AddModuleScore(rsv_path1_obj,features = feature,name=c("high_affinity","Low_affinity","exhaustion_genes",
                                                    "Bactivated_genes","BCSR_genes","CSR_m"))

rsv_path2_obj <- AddModuleScore(rsv_path2_obj,features = feature,name=c("high_affinity","Low_affinity","exhaustion_genes",
                                                            "Bactivated_genes","BCSR_genes","CSR_m"))

## 2. 合并shm
# Add SHM data to rsv_path1_obj (using v_identity_IGH.x from RSV data)
rsv_path1_obj$len.H_shm <- rsv_path1_shm[match(rsv_path1_obj$main_name,rsv_path1_shm$main_name),]$len.H_shm.
rsv_path1_obj$len.L_shm <- rsv_path1_shm[match(rsv_path1_obj$main_name,rsv_path1_shm$main_name),]$len.L_shm.

# Add SHM data to rsv_path2_obj (using v_identity_IGH.y from RSV data)
rsv_path2_obj$len.H_shm <- rsv_path2_shm[match(rsv_path2_obj$main_name,rsv_path2_shm$main_name),]$len.H_shm.
rsv_path2_obj$len.L_shm <- rsv_path2_shm[match(rsv_path2_obj$main_name,rsv_path2_shm$main_name),]$len.L_shm.

# Note: Using predict.x and predict.y directly from RSV data
# rsv_path1_obj uses predict.x, rsv_path2_obj uses predict.y


## 3.绘图
col_flg<-colorRampPalette(brewer.pal(8,"Set1"))(8)
###remove outlier
detect_outlier <- function(x) {
  Quantile1 <- quantile(x, probs=.25)
  Quantile3 <- quantile(x, probs=.75)
  IQR = Quantile3-Quantile1
  #x > Quantile3 + (IQR*1.5) | x < Quantile1 - (IQR*1.5)
  x > Quantile3  | x < Quantile1 
}

# create remove outlier function
remove_outlier <- function(dataframe,
                           columns=names(dataframe)) {
  for (col in columns) {
    dataframe <- dataframe[!detect_outlier(dataframe[[col]]), ]
  }
  return(dataframe)
  print("Remove outliers")
}



##  3.1 RSV Path1

load(file.path(files_dir, 'RSV_path1_B_monocle_cds.RData'))
df <- rsv_path1_obj@meta.data[match(cds@colData$main_name,rsv_path1_obj$main_name),
                        c("predict.x","predict.x",
                          "high_affinity1", "Low_affinity2", "exhaustion_genes3", "Bactivated_genes4",
                          "BCSR_genes5", "CSR_m6","len.H_shm","len.L_shm")]
df <- as.data.frame(df)
# Rename columns for consistency
colnames(df)[1:2] <- c("bind_values", "neut_values")
df$pseudotime <- pseudotime(cds)
df$celltype <- cds@colData$annotation_final

pathL <- list(path1 = rsv_path1_trajectory1, path2 = rsv_path1_trajectory2)

totalMD <- NULL
for(li in 1:length(pathL)){
  tempName <- names(pathL[li])
  tempPath <- pathL[[li]]
  tempdf <- df[df$celltype%in%tempPath,]
  # print(nrow(tempdf))
  tempdf <- tempdf[!is.na(tempdf$len.L_shm),]
  # print(nrow(tempdf))
  tempdf <- tempdf[!is.na(tempdf$bind_values),]
  tempdf$path <- names(pathL)[li]
  totalMD <- rbind(totalMD,tempdf)
}


totalMD.clean <- NULL
for(i in 1:length(unique(totalMD$celltype))){
  temp <- totalMD[totalMD$celltype==unique(totalMD$celltype)[i],]
  temp.clean <- remove_outlier(temp,c("pseudotime"))
  totalMD.clean <- rbind(totalMD.clean,temp.clean)
}

totalMD.clean <- totalMD[is.finite(totalMD$pseudotime), ]

totalMD.clean2 <- NULL
for(i in 1:length(unique(totalMD.clean$path))){
  temp.clean <- totalMD.clean[totalMD.clean$path==unique(totalMD.clean$path)[i],]
  temp.clean$pseudotime <- (temp.clean$pseudotime-min(temp.clean$pseudotime))/(max(temp.clean$pseudotime)-min(temp.clean$pseudotime))
  totalMD.clean2 <- rbind(totalMD.clean2,temp.clean)
}

#col_flg
colnames(totalMD.clean2)
totalMD.clean.use <- melt(totalMD.clean2,id.vars = c("celltype", "path",
                                                     "pseudotime")) 

fig <- list()
for(i in 1:length(unique(totalMD.clean.use$variable))){
  temp <- totalMD.clean.use[totalMD.clean.use$variable==unique(totalMD.clean.use$variable)[i],]
  p <- ggplot(temp) +
    #geom_point(aes(x = pseudotime, y = value,color=path)) +
    stat_smooth(aes(x = pseudotime, y = value,color=path),method = "lm", formula = y ~ poly(x, 2),se=F)+
    theme_classic()+scale_color_manual(values=col_flg)+ggtitle(unique(totalMD.clean.use$variable)[i])+
    theme(aspect.ratio = 1,
          text = element_text(size = 18), 
          legend.position = "",
          axis.title.x = element_blank(),  # 不显示 x 轴标题
          axis.title.y = element_blank()   # 不显示 y 轴标题
    )
  fig[[i]] <- p
}
table(totalMD.clean.use$celltype[totalMD.clean.use$path == 'path1'])
table(totalMD.clean.use$celltype[totalMD.clean.use$path == 'path2'])

#rownames(totalMD.clean.use)

length(fig)
fig[['nrow']] <- 2
fig[['ncol']] <- 5
# p <- DimPlot(data)
# ggsave(paste0(fdir,'temp.pdf'), p)

#fig
pdf(file.path(plots_dir, 'Figure4D_E_F_G-RSV_path1.pdf'), width = 25, height = 10)
do.call('grid.arrange', fig)
dev.off()



##  3.2 RSV Path2

load(file.path(files_dir, 'RSV_path2_B_monocle_cds.RData'))
df <- rsv_path2_obj@meta.data[rownames(cds@colData),
                        c("predict.y","predict.y",
                          "high_affinity1", "Low_affinity2", "exhaustion_genes3", "Bactivated_genes4",
                          "BCSR_genes5", "CSR_m6","len.H_shm","len.L_shm")]
df <- as.data.frame(df)
# Rename columns for consistency
colnames(df)[1:2] <- c("bind_values", "neut_values")
df$pseudotime <- pseudotime(cds)
df$celltype <- cds@colData$annotation_final

pathL <- list(path1 = rsv_path2_trajectory1, path2 = rsv_path2_trajectory2)

totalMD <- NULL
for(li in 1:length(pathL)){
  tempName <- names(pathL[li])
  tempPath <- pathL[[li]]
  tempdf <- df[df$celltype%in%tempPath,]
  # print(nrow(tempdf))
  tempdf <- tempdf[!is.na(tempdf$len.L_shm),]
  # print(nrow(tempdf))
  tempdf <- tempdf[!is.na(tempdf$bind_values),]
  tempdf$path <- names(pathL)[li]
  totalMD <- rbind(totalMD,tempdf)
}


totalMD.clean <- NULL
for(i in 1:length(unique(totalMD$celltype))){
  temp <- totalMD[totalMD$celltype==unique(totalMD$celltype)[i],]
  temp.clean <- remove_outlier(temp,c("pseudotime"))
  totalMD.clean <- rbind(totalMD.clean,temp.clean)
}

totalMD.clean <- totalMD[is.finite(totalMD$pseudotime), ]

totalMD.clean2 <- NULL
for(i in 1:length(unique(totalMD.clean$path))){
  temp.clean <- totalMD.clean[totalMD.clean$path==unique(totalMD.clean$path)[i],]
  temp.clean$pseudotime <- (temp.clean$pseudotime-min(temp.clean$pseudotime))/(max(temp.clean$pseudotime)-min(temp.clean$pseudotime))
  totalMD.clean2 <- rbind(totalMD.clean2,temp.clean)
}

#col_flg
colnames(totalMD.clean2)
totalMD.clean.use <- melt(totalMD.clean2,id.vars = c("celltype", "path",
                                                     "pseudotime")) 

fig <- list()
for(i in 1:length(unique(totalMD.clean.use$variable))){
  temp <- totalMD.clean.use[totalMD.clean.use$variable==unique(totalMD.clean.use$variable)[i],]
  p <- ggplot(temp) +
    #geom_point(aes(x = pseudotime, y = value,color=path)) +
    stat_smooth(aes(x = pseudotime, y = value,color=path),method = "lm", formula = y ~ poly(x, 2),se=F)+
    theme_classic()+scale_color_manual(values=col_flg)+ggtitle(unique(totalMD.clean.use$variable)[i])+
    theme(aspect.ratio = 1,
          text = element_text(size = 18), 
          legend.position = "",
          axis.title.x = element_blank(),  # 不显示 x 轴标题
          axis.title.y = element_blank()   # 不显示 y 轴标题
    )
  fig[[i]] <- p
}
table(totalMD.clean.use$celltype[totalMD.clean.use$path == 'path1'])
table(totalMD.clean.use$celltype[totalMD.clean.use$path == 'path2'])

#rownames(totalMD.clean.use)

length(fig)
fig[['nrow']] <- 2
fig[['ncol']] <- 5
# p <- DimPlot(data)
# ggsave(paste0(fdir,'temp.pdf'), p)

#fig
pdf(file.path(plots_dir, 'Figure4D_E_F_G-RSV_path2.pdf'), width = 25, height = 10)
do.call('grid.arrange', fig)
dev.off()


###############################################################################
#'                     Manuscipt: figureS6A/B/C/D                            '#
###############################################################################

## Figure S6A/B/C/D
## (A) B cell activation related transcriptional markers (CD86, ITGB2, ITGB2-AS1, SOX5, TNFRSF1B and FAS) across pseudotime trajectory 2. 
## (B) Atypical B cell related transcriptional markers (CD86, ITGB2, ITGB2-AS1, SOX5, TNFRSF1B and FAS) across pseudotime trajectory 2.  
## (C) Isotype-specific immunoglobulin expression dynamics (IGHA1, IGHD, IGHG1, IGHG2, IGHG3, and IGHM) along the pseudotime trajectory 2
## (D) Expression patterns of transcription factors (AFF3, BACH2, IRF8) along trajectory 2.

# Use the main rsv_path2 cds object instead of separate subset
load(file.path(files_dir, 'RSV_path2_B_monocle_cds.RData'))
rsv_path2_cds_subset_2 <- cds[,colData(cds)$annotation_final %in% rsv_path2_trajectory2]

genes.1 <- c('CD86','ITGB2','ITGB2-AS1','SOX5','TNFRSF1B','FAS')
genes.2 <- c('ITGAX','ZEB2','FCRL3','FCRL4','FCRL5')
genes.3 <- c('IGHA1','IGHD','IGHG1','IGHG2','IGHG3','IGHM')
genes.4 <- c('AFF3','BACH2','IRF8')

# Check gene existence and filter out missing genes
available_genes <- rownames(rsv_path2_cds_subset_2)
genes.1 <- genes.1[genes.1 %in% available_genes]
genes.2 <- genes.2[genes.2 %in% available_genes]
genes.3 <- genes.3[genes.3 %in% available_genes]
genes.4 <- genes.4[genes.4 %in% available_genes]

# Print warning for missing genes
if(length(genes.1) > 0) {
  p <- plot_genes_in_pseudotime(rsv_path2_cds_subset_2[genes.1,],color_cells_by="annotation_final",
                                ncol = 6)
  ggsave(file.path(plots_dir, "FigureS6A.pdf"), plot = p,width = 16, height = 3)
} else {
  cat("Warning: No genes found for FigureS6A\n")
}

if(length(genes.2) > 0) {
  p <- plot_genes_in_pseudotime(rsv_path2_cds_subset_2[genes.2,],color_cells_by="annotation_final",
                                ncol = 6)
  ggsave(file.path(plots_dir, "FigureS6B.pdf"), plot = p,width = 16, height = 3)
} else {
  cat("Warning: No genes found for FigureS6B\n")
}

if(length(genes.3) > 0) {
  p <- plot_genes_in_pseudotime(rsv_path2_cds_subset_2[genes.3,],color_cells_by="annotation_final",
                                ncol = 6)
  ggsave(file.path(plots_dir, "FigureS6C.pdf"), plot = p,width = 16, height = 3)
} else {
  cat("Warning: No immunoglobulin genes found for FigureS6C\n")
}

if(length(genes.4) > 0) {
  p <- plot_genes_in_pseudotime(rsv_path2_cds_subset_2[genes.4,],color_cells_by="annotation_final",
                                ncol = 6)
  ggsave(file.path(plots_dir, "FigureS6D.pdf"), plot = p,width = 9, height = 3)
} else {
  cat("Warning: No genes found for FigureS6D\n")
}




###########################################################
# 在文件末尾添加成功信息
cat("Figure 4 analysis completed successfully!\n")
cat("Output files saved to:", plots_dir, "\n")
cat("Data files saved to:", files_dir, "\n")

# 记录运行环境
sink(file.path(output_dir, "session_info.txt"))  # 将输出重定向到文件
sessionInfo()
sink()  # 关闭重定向

