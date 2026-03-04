# T Cell Analysis 使用示例

本文档提供 T 细胞单细胞分析工具的详细使用示例。

## 目录

1. [快速开始](#快速开始)
2. [数据整合](#数据整合)
3. [细胞类型可视化](#细胞类型可视化)
4. [标记基因分析](#标记基因分析)
5. [轨迹分析](#轨迹分析)
6. [TCR 结合可视化](#tcr-结合可视化)
7. [克隆型分析](#克隆型分析)
8. [完整工作流示例](#完整工作流示例)

---

## 快速开始

### 标准 T 细胞分析流程

```python
# Step 1: 整合 TCR 数据（必须首先执行）
{
    "tool": "integrate_tcr_data_complete",
    "arguments": {
        "input_csv": "/data/sessions/session_123/output/nettcr_predictions.csv",
        "input_rds": "/data/sessions/session_123/input/meta.rds",
        "output_path": "/data/sessions/session_123/output/integrated_1.rds"
    }
}

# Step 2: 细胞类型可视化
{
    "tool": "tcell_celltype_visualization",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_1.rds",
        "base_dir": "/data/sessions/session_123/output/celltype"
    }
}

# Step 3: 克隆型分析
{
    "tool": "tcr_clonotype_analysis",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_1.rds",
        "base_dir": "/data/sessions/session_123/output/clonotype"
    }
}
```

---

## 数据整合

### integrate_tcr_data_complete

#### 基本用法

```json
{
    "tool": "integrate_tcr_data_complete",
    "arguments": {
        "input_csv": "/data/sessions/session_123/output/nettcr_predictions.csv",
        "input_rds": "/data/sessions/session_123/input/meta.rds",
        "output_path": "/data/sessions/session_123/output/integrated_1.rds"
    }
}
```

#### 链式整合（多次整合）

```json
// 第一次整合：NetTCR 预测结果
{
    "tool": "integrate_tcr_data_complete",
    "arguments": {
        "input_csv": "/data/sessions/session_123/output/nettcr_predictions.csv",
        "input_rds": "/data/sessions/session_123/input/meta.rds",
        "output_path": "/data/sessions/session_123/output/integrated_1.rds"
    }
}

// 第二次整合：IgBLAST AIRR 结果
{
    "tool": "integrate_tcr_data_complete",
    "arguments": {
        "input_csv": "/data/sessions/session_123/output/airr_results.tsv",
        "input_rds": "/data/sessions/session_123/output/integrated_1.rds",
        "output_path": "/data/sessions/session_123/output/integrated_2.rds"
    }
}
```

#### 自定义 barcode 匹配

```json
{
    "tool": "integrate_tcr_data_complete",
    "arguments": {
        "input_csv": "/data/sessions/session_123/output/predictions.csv",
        "input_rds": "/data/sessions/session_123/input/meta.rds",
        "output_path": "/data/sessions/session_123/output/integrated.rds",
        "csv_fields": "sequence_id,barcode",
        "rds_fields": "cell_id,barcode"
    }
}
```

#### 跳过可选步骤

```json
{
    "tool": "integrate_tcr_data_complete",
    "arguments": {
        "input_csv": "/data/sessions/session_123/output/predictions.csv",
        "input_rds": "/data/sessions/session_123/input/meta.rds",
        "output_path": "/data/sessions/session_123/output/integrated.rds",
        "skip_umap": false,
        "skip_annotation": false
    }
}
```

#### 输出结果

```json
{
    "rds_path": "/data/sessions/session_123/output/integrated_1.rds",
    "metadata_columns": [
        "peptide",
        "score",
        "percentile_rank",
        "is_binder",
        "A1", "A2", "A3",
        "B1", "B2", "B3"
    ],
    "integrated_cells": 8500,
    "total_cells": 10000,
    "integration_rate": "85%"
}
```

---

## 细胞类型可视化

### tcell_celltype_visualization

#### 基本用法

```json
{
    "tool": "tcell_celltype_visualization",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_1.rds",
        "base_dir": "/data/sessions/session_123/output/celltype"
    }
}
```

#### 免疫细胞级别注释

```json
{
    "tool": "tcell_celltype_visualization",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_1.rds",
        "base_dir": "/data/sessions/session_123/output/celltype",
        "annotation_level": "immune"
    }
}
```

#### 输出文件

1. **celltype_umap.png** - 细胞类型 UMAP 图
2. **celltype_pie.png** - 细胞类型饼图
3. **celltype_bar.png** - 细胞类型柱状图
4. **celltype_stats.csv** - 细胞类型统计

#### 统计结果示例

```csv
cell_type,count,percentage
Naive_CD8,2500,25.0%
Effector_CD8,1800,18.0%
Memory_CD8,1200,12.0%
Exhausted_CD8,500,5.0%
Naive_CD4,2000,20.0%
Th1,800,8.0%
Th2,500,5.0%
Treg,400,4.0%
Other,300,3.0%
```

---

## 标记基因分析

### tcell_marker_dotplot_analysis

#### 显示所有标记基因

```json
{
    "tool": "tcell_marker_dotplot_analysis",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_1.rds",
        "base_dir": "/data/sessions/session_123/output/markers",
        "marker_set": "all"
    }
}
```

#### 仅显示 CD8 标记

```json
{
    "tool": "tcell_marker_dotplot_analysis",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_1.rds",
        "base_dir": "/data/sessions/session_123/output/markers",
        "marker_set": "cd8"
    }
}
```

#### 仅显示 CD4 标记

```json
{
    "tool": "tcell_marker_dotplot_analysis",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_1.rds",
        "base_dir": "/data/sessions/session_123/output/markers",
        "marker_set": "cd4"
    }
}
```

#### 标记基因列表

**CD8+ T 细胞标记：**
| 亚型 | 标记基因 |
|------|---------|
| Naive | CCR7, SELL, TCF7, LEF1 |
| Effector | GZMB, PRF1, GNLY, NKG7 |
| Memory | GZMK, CXCR3, KLRG1 |
| Exhausted | PDCD1, HAVCR2, LAG3, TIGIT |

**CD4+ T 细胞标记：**
| 亚型 | 标记基因 |
|------|---------|
| Naive | CCR7, SELL, TCF7, LEF1 |
| Th1 | TBX21, IFNG, CXCR3 |
| Th2 | GATA3, IL4, IL5, IL13 |
| Th17 | RORC, IL17A, IL17F, CCR6 |
| Treg | FOXP3, IL2RA, CTLA4 |
| Tfh | BCL6, CXCR5, PD-1 |

---

## 轨迹分析

### tcell_trajectory_analysis

#### 基本用法

```json
{
    "tool": "tcell_trajectory_analysis",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_1.rds",
        "base_dir": "/data/sessions/session_123/output/trajectory"
    }
}
```

#### 自定义参数

```json
{
    "tool": "tcell_trajectory_analysis",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_1.rds",
        "base_dir": "/data/sessions/session_123/output/trajectory",
        "num_dim": 30,
        "cluster_resolution": 0.005,
        "min_gene_cells": 5
    }
}
```

#### 输出文件

1. **trajectory_umap.png** - 轨迹 UMAP 图（带伪时间着色）
2. **pseudotime_distribution.png** - 伪时间分布图
3. **trajectory_genes.csv** - 轨迹相关基因

#### 轨迹基因示例

```csv
gene_id,moran_test_score,mean_expression,pseudotime_correlation
GZMB,0.85,2.5,0.78
PRF1,0.82,2.3,0.75
PDCD1,0.78,1.8,0.72
HAVCR2,0.75,1.5,0.68
TCF7,-0.65,2.0,-0.70
CCR7,-0.62,1.8,-0.68
```

---

## TCR 结合可视化

### tcr_binding_visualization

#### 基本用法

```json
{
    "tool": "tcr_binding_visualization",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_1.rds",
        "base_dir": "/data/sessions/session_123/output/binding"
    }
}
```

#### 前置条件

此工具需要先整合 NetTCR 预测结果：

```json
// Step 1: 运行 NetTCR 预测
{
    "tool": "predict_tcr_binding_complete",
    "arguments": {
        "test_file": "/data/sessions/session_123/input/tcr_data.csv",
        "output_dir": "/data/sessions/session_123/output/nettcr"
    }
}

// Step 2: 整合预测结果
{
    "tool": "integrate_tcr_data_complete",
    "arguments": {
        "input_csv": "/data/sessions/session_123/output/nettcr/predictions.csv",
        "input_rds": "/data/sessions/session_123/input/meta.rds",
        "output_path": "/data/sessions/session_123/output/integrated_1.rds"
    }
}

// Step 3: 可视化结合预测
{
    "tool": "tcr_binding_visualization",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_1.rds",
        "base_dir": "/data/sessions/session_123/output/binding"
    }
}
```

#### 输出文件

1. **binding_umap.png** - 结合预测 UMAP 图
2. **binding_density.png** - Nebulosa 密度图
3. **binding_stats.csv** - 结合统计

#### 结合统计示例

```csv
metric,value
total_cells,10000
cells_with_prediction,8500
predicted_binders,2500
binder_percentage,29.4%
avg_binding_score,0.45
median_rank,3.5%
```

---

## 克隆型分析

### tcr_clonotype_analysis

#### 基本用法

```json
{
    "tool": "tcr_clonotype_analysis",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_1.rds",
        "base_dir": "/data/sessions/session_123/output/clonotype"
    }
}
```

#### 按条件分组

```json
{
    "tool": "tcr_clonotype_analysis",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_1.rds",
        "base_dir": "/data/sessions/session_123/output/clonotype",
        "group_by": "condition",
        "top_n": 30
    }
}
```

#### 输出文件

1. **clonotype_frequency.png** - 克隆型频率图
2. **clonotype_diversity.png** - 多样性指数图
3. **top_clonotypes.csv** - Top 克隆型列表
4. **diversity_stats.csv** - 多样性统计

#### Top 克隆型示例

```csv
rank,clonotype_id,frequency,count,cdr3a,cdr3b
1,CLONE_001,5.2%,442,CAVRDSNYQLIWGDYKLTF,CASSLAPGATNEKLFF
2,CLONE_002,3.8%,323,CAVRDSNYQLIWGDYKLTF,CASSIEAGGTSGELFF
3,CLONE_003,2.5%,212,CAVNFGGGKLITGTQYF,CASSPGAGGTSGELFF
4,CLONE_004,1.8%,153,CAVRDSNYQLIWGDYKLTF,CASSQDRDRETQYF
5,CLONE_005,1.5%,128,CAVNFGGGKLITGTQYF,CASSLAPGATNEKLFF
```

#### 多样性统计示例

```csv
sample,shannon_entropy,simpson_index,clonality,unique_clonotypes
treated,4.52,0.98,0.32,1250
control,5.21,0.99,0.21,2100
```

---

## 完整工作流示例

### 场景 1: 抗原特异性 T 细胞分析

```python
# 完整的抗原特异性 T 细胞分析流程

# Step 1: TCR-肽段结合预测
{
    "tool": "predict_tcr_binding_complete",
    "arguments": {
        "test_file": "/data/sessions/session_123/input/tcr_sequences.csv",
        "output_dir": "/data/sessions/session_123/output/nettcr",
        "rank_threshold": 2.0
    }
}

# Step 2: 整合预测结果
{
    "tool": "integrate_tcr_data_complete",
    "arguments": {
        "input_csv": "/data/sessions/session_123/output/nettcr/predictions.csv",
        "input_rds": "/data/sessions/session_123/input/meta.rds",
        "output_path": "/data/sessions/session_123/output/integrated_1.rds"
    }
}

# Step 3: 细胞类型分析
{
    "tool": "tcell_celltype_visualization",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_1.rds",
        "base_dir": "/data/sessions/session_123/output/celltype"
    }
}

# Step 4: 结合可视化
{
    "tool": "tcr_binding_visualization",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_1.rds",
        "base_dir": "/data/sessions/session_123/output/binding"
    }
}

# Step 5: 克隆型分析
{
    "tool": "tcr_clonotype_analysis",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_1.rds",
        "base_dir": "/data/sessions/session_123/output/clonotype",
        "group_by": "is_binder"
    }
}
```

### 场景 2: T 细胞分化研究

```python
# Step 1: 数据整合
{
    "tool": "integrate_tcr_data_complete",
    "arguments": {
        "input_csv": "/data/sessions/session_123/output/tcr_annotations.csv",
        "input_rds": "/data/sessions/session_123/input/tcell_data.rds",
        "output_path": "/data/sessions/session_123/output/integrated.rds"
    }
}

# Step 2: 细胞类型注释
{
    "tool": "tcell_celltype_visualization",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated.rds",
        "base_dir": "/data/sessions/session_123/output/celltype"
    }
}

# Step 3: 标记基因验证
{
    "tool": "tcell_marker_dotplot_analysis",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated.rds",
        "base_dir": "/data/sessions/session_123/output/markers",
        "marker_set": "all"
    }
}

# Step 4: 轨迹分析
{
    "tool": "tcell_trajectory_analysis",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated.rds",
        "base_dir": "/data/sessions/session_123/output/trajectory",
        "num_dim": 50,
        "cluster_resolution": 0.001
    }
}
```

### 场景 3: 多肽段比较分析

```python
# 对多个肽段的 TCR 进行比较分析

# Step 1: 预测多个肽段
{
    "tool": "predict_tcr_binding_complete",
    "arguments": {
        "test_file": "/data/sessions/session_123/input/multi_peptide_tcr.csv",
        "output_dir": "/data/sessions/session_123/output/nettcr_multi"
    }
}

# Step 2: 整合结果
{
    "tool": "integrate_tcr_data_complete",
    "arguments": {
        "input_csv": "/data/sessions/session_123/output/nettcr_multi/predictions.csv",
        "input_rds": "/data/sessions/session_123/input/meta.rds",
        "output_path": "/data/sessions/session_123/output/integrated_multi.rds"
    }
}

# Step 3: 按肽段分组进行克隆型分析
{
    "tool": "tcr_clonotype_analysis",
    "arguments": {
        "input_file": "/data/sessions/session_123/output/integrated_multi.rds",
        "base_dir": "/data/sessions/session_123/output/clonotype_by_peptide",
        "group_by": "peptide",
        "top_n": 20
    }
}
```

---

## 与 NetTCR 联合使用

### 推荐工作流

```
┌─────────────────────────────────────────────────────────────┐
│                     T 细胞分析完整流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. TCR 序列数据                                             │
│     ↓                                                       │
│  2. NetTCR 预测 (predict_tcr_binding_complete)               │
│     ↓                                                       │
│  3. 数据整合 (integrate_tcr_data_complete)                   │
│     ↓                                                       │
│  ├──────────────┬──────────────┬──────────────┬───────────│
│  ↓              ↓              ↓              ↓           │
│  细胞类型      标记基因       轨迹分析      结合可视化       │
│  可视化        点图                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 数据流转示例

```python
# 输入
tcr_sequences.csv  →  TCR 序列数据
meta.rds           →  Seurat 单细胞对象

# NetTCR 输出
predictions.csv    →  结合预测结果

# 整合后
integrated_1.rds   →  包含预测结果的 Seurat 对象

# 分析输出
celltype/          →  细胞类型可视化
markers/           →  标记基因点图
trajectory/        →  轨迹分析结果
binding/           →  结合预测可视化
clonotype/         →  克隆型分析
```

---

## 常见问题

### Q1: 整合后没有新增元数据列

**原因：** barcode 匹配失败

**解决方案：** 检查 `csv_fields` 和 `rds_fields` 参数，确保字段名称正确

### Q2: 轨迹分析运行时间过长

**原因：** 细胞数量过多

**解决方案：** 先进行细胞亚群筛选，减少分析的细胞数量

### Q3: 结合可视化没有结果

**原因：** 输入数据中没有结合预测列

**解决方案：** 确保先运行 `integrate_tcr_data_complete` 整合 NetTCR 结果

---

更多详细信息请参考 [best_practices.md](./best_practices.md)。

