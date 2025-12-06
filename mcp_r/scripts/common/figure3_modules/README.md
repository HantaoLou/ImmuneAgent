# Figure3 模块化代码说明

本目录包含从 `Figure3_Common.R` 拆分出来的独立模块，每个模块对应原始代码中的一个 Manuscript 块。

## 文件结构

```
figure3_modules/
├── Figure3_Utils.R          # 共享工具函数和数据处理逻辑
├── Figure3_A_Density.R      # Figure 3A: UMAP结合/中和预测密度图
├── Figure3_C_CellType.R     # Figure 3C: UMAP B细胞类型分布图
├── Figure3_D_DotPlot.R      # Figure 3D: DotPlot标记基因表达图
├── Figure3_F_Correlation.R  # Figure 3F: 差异表达基因相关性分析
├── Figure3_G_Prediction.R   # Figure 3G: UMAP预测值密度图
├── syntax_check.R           # 语法检查脚本
└── README.md               # 本说明文件
```

## 模块说明

### Figure3_Utils.R
包含所有模块共享的工具函数：
- `king_celltype_mapping()`: King数据集细胞类型映射
- `detect_all_binding_columns()`: 检测结合预测字段
- `load_and_preprocess_data()`: 数据加载和预处理
- `create_output_directories()`: 创建输出目录
- `get_color_palette()`: 获取颜色配置
- `get_bcell_markers()`: 获取B细胞标记基因
- `load_required_packages()`: 加载必需的R包

### Figure3_A_Density.R
生成 Figure 3A：UMAP结合/中和预测密度图
- 自动检测可用的预测字段（结合预测、中和预测、平均值字段）
- 创建Nebulosa密度图显示预测值分布
- 输出文件：`Figure_3A.pdf`

### Figure3_C_CellType.R  
生成 Figure 3C：UMAP B细胞类型分布图
- 显示不同B细胞亚型在UMAP空间中的分布
- 使用36色调色板进行细胞类型着色
- 输出文件：`Figure_3C.pdf`、细胞类型统计CSV

### Figure3_D_DotPlot.R
生成 Figure 3D：DotPlot标记基因表达图
- 显示B细胞亚型特异性标记基因的表达模式
- 自动检测数据中可用的标记基因
- 输出文件：`Figure_3D.pdf`、标记基因信息CSV

### Figure3_F_Correlation.R
生成 Figure 3F：差异表达基因相关性分析
- 分析两个数据集之间差异表达基因的相关性
- 支持参数化的p值阈值和最小共同基因数量
- 自动验证输入文件格式和必需字段
- 生成散点图并标注指定的生物标记基因
- 计算Pearson相关系数和统计显著性
- 输出文件：`Figure_3F_[dataset1]_vs_[dataset2].pdf`、相关性数据CSV、统计结果CSV

### Figure3_G_Prediction.R
生成 Figure 3G：UMAP预测值密度图
- 在UMAP空间中显示预测值的分布
- 使用渐变色显示预测分数
- 输出文件：`Figure_3G.pdf`、预测值统计CSV

## 使用方法

### 单独运行模块
每个模块都可以独立运行，基本语法：
```bash
Rscript <模块文件> <输入RDS文件> <基础目录>
```

### 示例命令
```bash
# 生成Figure 3A
Rscript Figure3_A_Density.R /path/to/data.rds /path/to/output

# 生成Figure 3C  
Rscript Figure3_C_CellType.R /path/to/data.rds /path/to/output

# 生成Figure 3D
Rscript Figure3_D_DotPlot.R /path/to/data.rds /path/to/output

# 生成Figure 3G
Rscript Figure3_G_Prediction.R /path/to/data.rds /path/to/output
```

### 批量运行所有模块
```bash
# 在figure3_modules目录下运行
for script in Figure3_A_Density.R Figure3_C_CellType.R Figure3_D_DotPlot.R Figure3_G_Prediction.R; do
    Rscript $script /path/to/data.rds /path/to/output
done
```

## 输入要求

### 数据文件要求
- **输入文件**：Seurat对象的RDS文件
- **必需字段**：
  - `CellType`: 细胞类型注释
  - `UMAP_1`, `UMAP_2`: UMAP坐标（或Seurat对象中的umap reduction）
  - 预测字段：包含"bind"、"output"、"neut"等关键词的列

### 目录结构
脚本会自动创建以下输出目录结构：
```
<base_dir>/
└── output/
    └── Figure3/
        ├── plots/          # PDF图片文件
        └── files/          # CSV统计文件和session信息
```

## 依赖包

所有模块需要以下R包：
- Seurat
- dplyr  
- ggplot2
- cowplot
- ggrepel
- stringr
- monocle3
- RColorBrewer
- Nebulosa
- ggrastr

## 特性

1. **独立运行**：每个模块都可以单独运行，不依赖其他模块
2. **自动检测**：自动检测数据中可用的预测字段和标记基因
3. **错误处理**：包含完善的错误检查和警告信息
4. **兼容性**：支持King数据集的细胞类型映射
5. **输出完整**：生成图片文件、统计文件和运行环境信息

## 语法检查

运行语法检查脚本验证所有模块：
```bash
Rscript syntax_check.R
```

## 注意事项

1. 确保输入的RDS文件包含完整的Seurat对象
2. 基础目录必须存在且有写入权限
3. 如果数据中缺少某些字段，模块会给出警告但不会停止运行
4. 每个模块都会生成独立的session信息文件用于调试

## 与原始代码的对应关系

| 模块文件 | 原始代码块 | 功能描述 |
|---------|-----------|----------|
| Figure3_A_Density.R | figure3a | UMAP结合/中和预测密度图 |
| Figure3_C_CellType.R | figure3C | UMAP B细胞类型分布图 |
| Figure3_D_DotPlot.R | figure3D | DotPlot标记基因表达图 |
| Figure3_F_Correlation.R | figure3F | 差异表达基因相关性分析 |
| Figure3_G_Prediction.R | figure3G | UMAP预测值密度图 |

所有模块保持了原始代码的逻辑不变，只是增加了独立运行的能力和更好的错误处理。