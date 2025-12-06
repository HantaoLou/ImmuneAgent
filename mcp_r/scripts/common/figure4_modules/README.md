# Figure4 模块化代码说明

本目录包含从 `Figure4_Common.R` 拆分出来的独立模块，每个模块对应原始代码中的一个 Manuscript 块。

## 目录结构

```
figure4_modules/
├── Figure4_Utils.R              # 工具函数集合
├── Figure4_A_Trajectory.R       # Figure 4A - UMAP伪时间轨迹分析
├── Figure4_C_Boxplot.R          # Figure 4C - 伪时间与细胞类型箱线图
├── Figure4_DEFG_Polynomial.R    # Figure 4D/E/F/G - 多项式回归分析
├── Figure4_S6_Supplementary.R   # Figure S6A/B/C/D - 补充图形
└── README.md                    # 本说明文件
```

## 模块功能说明

### 1. Figure4_Utils.R
包含所有模块需要的通用函数：
- `king_celltype_mapping()` - King数据集细胞类型映射
- `load_and_preprocess_data()` - 数据加载和预处理
- `create_output_directories()` - 创建输出目录
- `get_bcell_color_panel()` - 获取B细胞颜色配置
- `get_bcell_markers()` - 获取B细胞标记基因
- `detect_binding_columns()` - 检测结合预测字段
- `process_binding_data()` - 处理结合数据
- `get_feature_gene_sets()` - 获取特征基因集合
- `estimate_shm_from_expression()` - 基于表达估算SHM
- `detect_outlier()` / `remove_outlier()` - 异常值处理

### 2. Figure4_A_Trajectory.R
**功能**: UMAP伪时间轨迹分析
- 使用monocle3进行轨迹分析
- 自动选择根细胞（优先选择Naive B细胞）
- 生成伪时间图
- 保存CDS对象供其他模块使用

**输出**:
- `Figure_4A.pdf` - 伪时间轨迹图
- `flu_B_monocle_cds.RData` - monocle3 CDS对象

### 3. Figure4_C_Boxplot.R
**功能**: 伪时间与细胞类型的箱线图
- 依赖于Figure4_A生成的CDS对象
- 如果CDS对象不存在，会尝试创建简化版本
- 显示不同细胞类型的伪时间分布

**输出**:
- `Figure_4C.pdf` - 箱线图
- `Figure4C_pseudotime_stats.csv` - 伪时间统计数据

### 4. Figure4_DEFG_Polynomial.R
**功能**: 多项式回归分析
- 计算特征基因模块分数
- 估算体细胞超突变(SHM)水平
- 沿轨迹的多项式回归分析
- 生成组合图形

**输出**:
- `Figure4D_E_F_G-flu.pdf` - 组合图形
- `Figure4DEFG_trajectory_data.csv` - 轨迹数据

### 5. Figure4_S6_Supplementary.R
**功能**: 补充图形S6A/B/C/D
- S6A: B细胞激活相关转录标记
- S6B: 非典型B细胞相关转录标记
- S6C: 免疫球蛋白表达动态
- S6D: 转录因子表达模式

**输出**:
- `FigureS6A.pdf` - B细胞激活标记
- `FigureS6B.pdf` - 非典型B细胞标记
- `FigureS6C.pdf` - 免疫球蛋白动态
- `FigureS6D.pdf` - 转录因子模式
- 对应的基因表达数据CSV文件

## 使用方法

### 基本用法
每个模块都可以独立运行：

```bash
# 运行Figure 4A
Rscript Figure4_A_Trajectory.R <input_rds_file> <base_dir>

# 运行Figure 4C
Rscript Figure4_C_Boxplot.R <input_rds_file> <base_dir>

# 运行Figure 4D/E/F/G
Rscript Figure4_DEFG_Polynomial.R <input_rds_file> <base_dir>

# 运行Figure S6
Rscript Figure4_S6_Supplementary.R <input_rds_file> <base_dir>
```

### 参数说明
- `<input_rds_file>`: 输入的RDS文件路径
- `<base_dir>`: 基础输出目录

### 运行顺序建议
1. **首先运行**: `Figure4_A_Trajectory.R` - 生成CDS对象
2. **然后运行**: 其他模块（可并行运行）

### 依赖关系
- `Figure4_C_Boxplot.R` 依赖 `Figure4_A_Trajectory.R` 生成的CDS对象
- `Figure4_DEFG_Polynomial.R` 依赖 `Figure4_A_Trajectory.R` 生成的CDS对象
- `Figure4_S6_Supplementary.R` 依赖 `Figure4_A_Trajectory.R` 生成的CDS对象

## 输入数据要求

### 必需字段
- 细胞类型注释：`CellType`、`combined_cluster` 或 `seurat_clusters`
- UMAP坐标：`UMAP_1`、`UMAP_2` 或 `umap` reduction

### 可选字段
- 结合预测数据：包含 "bind_predict"、"output" 或 "bind_output" 的列
- 序列数据：`ig_seq.x`、`ig_seq.y`（用于SHM计算）

## 依赖包
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
- reshape2 (用于Figure 4D/E/F/G)
- gridExtra (用于Figure 4D/E/F/G)

## 特性
- **独立运行**: 每个模块都可以独立执行
- **错误处理**: 包含完善的错误处理和警告信息
- **自动检测**: 自动检测数据格式和可用字段
- **向后兼容**: 保持与原始Figure4_Common.R相同的逻辑
- **完整输出**: 生成图片和相关数据文件

## 语法验证
所有模块文件都已通过R语法检查：
- ✓ Figure4_Utils.R
- ✓ Figure4_A_Trajectory.R  
- ✓ Figure4_C_Boxplot.R
- ✓ Figure4_DEFG_Polynomial.R
- ✓ Figure4_S6_Supplementary.R

## 与原始代码的对应关系
- **Figure4_A_Trajectory.R** ← Figure4_Common.R 第127-226行 (figure4A)
- **Figure4_C_Boxplot.R** ← Figure4_Common.R 第228-264行 (figure4C)
- **Figure4_DEFG_Polynomial.R** ← Figure4_Common.R 第266-700行 (figure4D/E/F/G)
- **Figure4_S6_Supplementary.R** ← Figure4_Common.R 第702-848行 (figureS6A/B/C/D)