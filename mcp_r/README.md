# R环境和依赖包安装指南

本文档详细介绍了在Ubuntu系统中安装R环境以及项目所需依赖包的完整流程。

## 目录

- [系统要求](#系统要求)
- [安装R环境](#安装r环境)
- [创建项目环境](#创建项目环境)
- [安装系统依赖](#安装系统依赖)
- [配置镜像源](#配置镜像源)
- [安装R包依赖](#安装r包依赖)
- [验证安装](#验证安装)
- [常见问题](#常见问题)

## 系统要求

- Ubuntu 20.04 LTS 或更高版本
- 至少 4GB 内存
- 至少 2GB 可用磁盘空间
- 网络连接

## 安装R环境

### 1. 更新系统包

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. 安装必要的依赖项

```bash
sudo apt install -y software-properties-common dirmngr gnupg apt-transport-https ca-certificates
```

### 3. 添加R官方仓库

```bash
# 添加GPG密钥
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc

# 添加R仓库源
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
```

### 4. 安装R

```bash
sudo apt update
sudo apt install -y r-base r-base-dev
```

### 5. 验证R安装

```bash
R --version
```

## 创建项目环境

### 1. 创建项目目录

```bash
mkdir ~/r_project
cd ~/r_project
```

### 2. 初始化renv环境

```bash
R
```

在R中执行：

```r
# 安装renv
install.packages("renv")

# 初始化项目环境
renv::init()

# 退出R
q()
```

## 安装系统依赖

### 1. 安装编译工具链

```bash
sudo apt install -y \
    build-essential \
    gfortran \
    g++ \
    gcc \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    r-base-dev
```

### 2. 创建libgfortran符号链接

```bash
# 检查libgfortran状态
ls -la /usr/lib/x86_64-linux-gnu/libgfortran*

# 创建符号链接（如果需要）
sudo ln -sf /usr/lib/x86_64-linux-gnu/libgfortran.so.5 /usr/lib/x86_64-linux-gnu/libgfortran.so

# 更新链接器缓存
sudo ldconfig
```

### 3. 安装图形库依赖

```bash
sudo apt install -y \
    libcairo2-dev \
    libxt-dev \
    libx11-dev \
    libxext-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff5-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libpango1.0-dev \
    libglib2.0-dev
```

### 4. 安装地理空间库依赖（sf包需要）

```bash
sudo apt install -y \
    libudunits2-dev \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libsqlite3-dev
```

### 5. 安装其他常用库

```bash
sudo apt install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgit2-dev \
    libhdf5-dev
```

## 配置镜像源

进入R环境：

```bash
cd ~/r_project
R
```

在R中配置镜像：

```r
# 配置CRAN镜像
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

# 配置Bioconductor镜像
options(BioC_mirror = "https://mirrors.tuna.tsinghua.edu.cn/bioconductor/")

# 保存到项目配置
cat('
# 项目镜像配置
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
options(BioC_mirror = "https://mirrors.tuna.tsinghua.edu.cn/bioconductor/")
', file = ".Rprofile")
```

## 安装R包依赖

### 1. 安装基础CRAN包

```r
# 基础数据处理包
install.packages(c(
  "here",        # 路径管理
  "dplyr",       # 数据处理  
  "ggplot2",     # 可视化
  "readr",       # 数据读取
  "tidyr",       # 数据整理
  "stringr",     # 字符串处理
  "lubridate",   # 日期处理
  "purrr",       # 函数式编程
  "tidyverse"    # 数据科学工具集
))

# 可视化和分析包
install.packages(c(
  "Seurat",      # 单细胞分析
  "cowplot",     # 图形组合
  "ggrepel",     # 标签避免重叠
  "RColorBrewer", # 调色板
  "ggrastr",     # 栅格化处理
  "patchwork",   # 图形拼接
  "readxl",      # Excel文件读取
  "rio",         # 通用数据导入导出
  "reshape2",    # 数据重塑
  "gridExtra"    # 网格布局
))
```

### 2. 安装Bioconductor包

```r
# 安装BiocManager
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

# 安装基础Bioconductor包（按依赖顺序）
# 第一层依赖
BiocManager::install(c(
    "UCSC.utils",
    "S4Arrays", 
    "Biobase"
), update = FALSE, ask = FALSE)

# 第二层依赖
BiocManager::install(c(
    "GenomeInfoDb",
    "SparseArray"
), update = FALSE, ask = FALSE)

# 第三层依赖
BiocManager::install(c(
    "GenomicRanges",
    "DelayedArray"
), update = FALSE, ask = FALSE)

# 第四层依赖
BiocManager::install(c(
    "SummarizedExperiment",
    "SingleCellExperiment"
), update = FALSE, ask = FALSE)

# 应用包
BiocManager::install(c(
    "Nebulosa",
    "harmony"
), update = FALSE, ask = FALSE)
```

### 3. 安装特殊包（从GitHub）

```r
# 安装devtools
if (!requireNamespace("devtools", quietly = TRUE))
    install.packages("devtools")

# 安装monocle3依赖
BiocManager::install(c(
    "batchelor", "BiocParallel", "DelayedMatrixStats", 
    "HDF5Array", "limma"
), update = FALSE, ask = FALSE)

install.packages(c(
    "speedglm", "lmtest", "pbapply", "RcppAnnoy", 
    "RcppHNSW", "sf", "viridis"
))

# 从GitHub安装monocle3
devtools::install_github("cole-trapnell-lab/monocle3")
```

### 4. 保存环境快照

```r
# 保存当前环境状态
renv::snapshot()
```

## 验证安装

### 1. 快速验证

```r
# 检查所有包是否安装
packages <- c("here", "dplyr", "ggplot2", "readr", "tidyr", "stringr", 
              "Seurat", "tidyverse", "cowplot", "ggrepel", "RColorBrewer", 
              "ggrastr", "patchwork", "readxl", "rio", "reshape2", "gridExtra", 
              "monocle3", "Nebulosa", "harmony", "SingleCellExperiment")

missing <- packages[!packages %in% rownames(installed.packages())]
if(length(missing) > 0) {
  print(paste("缺失包:", paste(missing, collapse=", ")))
} else {
  print("所有包都已安装成功！")
}
```

### 2. 测试关键包加载

```r
# 测试关键包
key_packages <- c("ggplot2", "dplyr", "Seurat", "Nebulosa", "monocle3")
for(pkg in key_packages) {
  tryCatch({
    library(pkg, character.only=TRUE)
    cat(pkg, "- 加载成功\n")
  }, error=function(e) cat(pkg, "- 加载失败\n"))
}
```

## 常见问题

### 1. gfortran编译错误

**问题**：`cannot find -lgfortran`

**解决**：
```bash
sudo apt install -y gfortran libgfortran-11-dev
sudo ln -sf /usr/lib/x86_64-linux-gnu/libgfortran.so.5 /usr/lib/x86_64-linux-gnu/libgfortran.so
sudo ldconfig
```

### 2. Cairo包安装失败

**问题**：`Cannot find cairo.h`

**解决**：
```bash
sudo apt install -y libcairo2-dev libxt-dev libx11-dev
```

### 3. sf包安装失败

**问题**：地理空间库依赖问题

**解决**：
```bash
sudo apt install -y libudunits2-dev libgdal-dev libgeos-dev libproj-dev
```

### 4. HDF5Array安装失败

**问题**：HDF5库依赖问题

**解决**：
```bash
sudo apt install -y libhdf5-dev
```

### 5. monocle3安装失败

**问题**：GitHub网络超时

**解决**：
1. 手动下载ZIP文件：https://github.com/cole-trapnell-lab/monocle3/archive/refs/heads/main.zip
2. 解压后使用 `devtools::install_local()` 安装

### 6. 网络下载慢

**解决**：使用国内镜像源
```r
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
options(BioC_mirror = "https://mirrors.tuna.tsinghua.edu.cn/bioconductor/")
```

## 项目结构

安装完成后的项目目录结构：
