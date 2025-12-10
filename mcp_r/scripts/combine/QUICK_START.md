# 🚀 快速上手指南

## 10秒速览

```bash
# 一键整合BCR数据
Rscript integrate_all.R \
  --csv your_bcr_data.csv \
  --rds your_seurat_object.rds \
  --csv-fields "Batch,barcode" \
  --rds-fields "rownames" \
  --output result.rds
```

---

## 📁 目录文件说明

```
combine/
├── 🆕 integrate_all.R           ← 一键整合工具（新增）
├── 📖 README_integrate_all.md   ← 详细使用文档
├── 📊 TOOLS_COMPARISON.md       ← 新旧工具对比
├── 🧪 test_integrate_all.sh     ← 测试脚本
│
├── standardize_csv.py           ← 原工具1: CSV标准化
├── standardize_rds.R            ← 原工具2: RDS标准化
└── integrate_bcr_data.R         ← 原工具3: 数据整合
```

---

## ⚡ 使用方式

### 方式A: 一键完成（推荐） ⭐

```bash
Rscript integrate_all.R \
  --csv data.csv \
  --rds data.rds \
  --csv-fields "Batch,barcode" \
  --rds-fields "rownames" \
  --output result.rds
```

### 方式B: 分步操作（高级）

```bash
# 步骤1: 标准化CSV
python standardize_csv.py

# 步骤2: 标准化RDS
Rscript standardize_rds.R data.rds "rownames" data_std.rds

# 步骤3: 整合数据
Rscript integrate_bcr_data.R data.csv data_std.rds result.rds
```

---

## 🔍 如何选择？

| 场景 | 推荐工具 |
|------|---------|
| 快速完成任务 | ✅ `integrate_all.R` |
| 需要中间文件 | ⚠️ 原三步工具 |
| 只标准化不整合 | ⚠️ `standardize_csv.py` 或 `standardize_rds.R` |
| 数据已标准化 | 两者皆可 |

---

## 📚 文档索引

| 文档 | 说明 | 适合对象 |
|------|------|---------|
| **README_integrate_all.md** | 完整使用文档，包含所有参数说明和示例 | 所有用户 |
| **TOOLS_COMPARISON.md** | 新旧工具对比分析 | 原工具用户 |
| **QUICK_START.md** | 本文档，快速上手 | 新用户 |

---

## ✅ 验证安装

```bash
# 运行测试脚本
bash test_integrate_all.sh

# 或查看帮助
Rscript integrate_all.R --help
```

---

## 💡 常用命令模板

### 1. 基础整合

```bash
Rscript integrate_all.R \
  --csv YOUR_CSV_FILE \
  --rds YOUR_RDS_FILE \
  --csv-fields "FIELD1,FIELD2" \
  --rds-fields "rownames" \
  --output OUTPUT_FILE
```

### 2. 跳过UMAP（加快速度）

```bash
Rscript integrate_all.R \
  --csv data.csv \
  --rds data.rds \
  --csv-fields "Batch,barcode" \
  --rds-fields "rownames" \
  --output result.rds \
  --skip-umap
```

### 3. 已标准化文件

```bash
# 自动检测，无需指定字段
Rscript integrate_all.R \
  --csv standardized.csv \
  --rds standardized.rds \
  --output result.rds
```

### 4. 自定义分隔符

```bash
Rscript integrate_all.R \
  --csv data.csv \
  --rds data.rds \
  --csv-fields "Batch,barcode" \
  --rds-fields "rownames" \
  --separator "-" \
  --output result.rds
```

---

## ❓ 遇到问题？

### 1. 检查字段名

```bash
# 查看CSV字段
head -n 1 your_file.csv

# 查看RDS字段
Rscript -e 'colnames(readRDS("your_file.rds")@meta.data)'
```

### 2. 验证是否已标准化

```bash
# CSV
grep "combine_barcode" your_file.csv | head -n 1

# RDS
Rscript -e '"combine_barcode" %in% colnames(readRDS("file.rds")@meta.data)'
```

### 3. 查看详细日志

```bash
Rscript integrate_all.R \
  --csv data.csv \
  --rds data.rds \
  --csv-fields "Batch,barcode" \
  --rds-fields "rownames" \
  --output result.rds \
  2>&1 | tee integration.log
```

---

## 🎯 核心特性

- ✅ **完全兼容**: 不影响原有工具
- ✅ **智能检测**: 自动判断是否需要标准化
- ✅ **灵活控制**: 支持跳过UMAP、自定义分隔符等
- ✅ **详细日志**: 每步操作都有进度提示
- ✅ **错误处理**: 完善的参数验证和错误提示

---

## 📞 支持

需要帮助？查看：
- 📖 完整文档: `README_integrate_all.md`
- 📊 工具对比: `TOOLS_COMPARISON.md`
- 🧪 运行测试: `bash test_integrate_all.sh`
