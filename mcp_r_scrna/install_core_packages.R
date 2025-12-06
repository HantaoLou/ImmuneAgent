#!/usr/bin/env Rscript
# 安装核心生物信息学包 - 简化版本

cat("开始安装核心生物信息学包...\n")

# 设置CRAN镜像
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# 安装BiocManager
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  cat("安装BiocManager...\n")
  install.packages("BiocManager")
}

# 加载BiocManager
library(BiocManager)

# 核心包列表
core_packages <- c(
  "clusterProfiler",
  "org.Hs.eg.db"
)

# 逐个安装包
for (pkg in core_packages) {
  cat("正在安装:", pkg, "\n")
  
  tryCatch({
    if (!requireNamespace(pkg, quietly = TRUE)) {
      BiocManager::install(pkg, ask = FALSE, update = FALSE)
      cat("✓", pkg, "安装完成\n")
    } else {
      cat("✓", pkg, "已存在\n")
    }
  }, error = function(e) {
    cat("✗", pkg, "安装失败:", e$message, "\n")
  })
}

# 验证安装
cat("\n验证包安装状态:\n")
for (pkg in core_packages) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    cat("✓", pkg, "可用\n")
  } else {
    cat("✗", pkg, "不可用\n")
  }
}

cat("核心包安装完成!\n")