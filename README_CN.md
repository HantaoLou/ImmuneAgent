# 抗体生成与免疫分析系统 - 中文文档

## 目录

1. [快速开始](#快速开始)
2. [环境安装及部署](#环境安装及部署)
3. [KB 知识库系统部署](#kb-知识库系统部署)
4. [IgBLAST 服务部署](#igblast-服务部署)
5. [代码运行及测试](#代码运行及测试)

---

## 快速开始

如果你已准备好所有组件，可以按照以下步骤快速启动：

1. **安装依赖环境**（见[环境安装及部署](#环境安装及部署)）
2. **部署 KB 知识库**（见[KB 知识库系统部署](#kb-知识库系统部署)）
3. **部署 IgBLAST 服务**（见[IgBLAST 服务部署](#igblast-服务部署)）
4. **配置并运行代码**（见[代码运行及测试](#代码运行及测试)）

---

## 环境安装及部署

### 1. 安装 uv

**Windows:**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Linux/Mac:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**验证安装:**
```bash
uv --version
```

### 2. 安装项目依赖

```bash
cd agent
uv sync
```

---

## KB 知识库系统部署

### 快速开始

如果你已有 kb 目录，按以下步骤快速部署：

```bash
# 1. 修改 docker-compose.yaml 中的数据目录路径（第17行）
# 将 /data_new/wyl/qdrant_data 改为你的实际路径

# 2. 进入项目目录并启动 Qdrant
cd /path/to/your/kb
docker compose up -d

# 3. 设置 Qdrant 连接为本地（重要！）
export QDRANT_HOST=localhost

# 4. 安装 uv 并配置 PATH
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 5. 安装依赖
uv sync

# 6. 安装 Ollama 并拉取模型
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &  # 后台启动
ollama pull nomic-embed-text

# 7. 验证部署
uv run kb list-collections
```

### 详细部署步骤

#### 1. Docker 和 Docker Compose 安装

```bash
# 更新系统包
sudo apt update
sudo apt upgrade -y

# 安装必要的依赖
sudo apt install -y ca-certificates curl gnupg lsb-release

# 添加 Docker 官方 GPG 密钥
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 添加 Docker 软件源
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 安装 Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 将当前用户添加到 docker 组
sudo usermod -aG docker $USER
newgrp docker

# 启动 Docker 服务
sudo systemctl start docker
sudo systemctl enable docker

# 验证安装
docker --version
docker compose version
```

#### 2. Qdrant 向量数据库部署

**配置文件修改：**

在 `kb/docker-compose.yaml` 中，**必须修改**第17行数据目录路径：

```yaml
volumes:
  - /data_new/wyl/qdrant_data:/qdrant/storage
```

改为你的实际路径，例如：
```yaml
volumes:
  - /你的存储路径/qdrant_data:/qdrant/storage
```

**启动 Qdrant:**
```bash
cd /path/to/your/kb
docker compose up -d
docker compose ps
```

**验证服务:**
```bash
curl http://localhost:6333/healthz
curl http://localhost:6333/collections
```

#### 3. KB 应用部署

**安装依赖:**
```bash
cd /path/to/your/kb
uv sync
```

**配置连接参数:**
```bash
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export QDRANT_GRPC_PORT=6334
```

**安装 Ollama 和模型:**
```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 启动服务
ollama serve &

# 拉取模型
ollama pull nomic-embed-text     # embedding模型（必需）
ollama pull gemma3:4b            # 默认摘要模型（推荐）
ollama pull qwq:latest           # 高级推理模型（可选）
```

#### 4. 使用向量库

```bash
# 加载文档
uv run kb load-doc --path ./library --collection_name collection2

# 查询文档
uv run kb query --query "gearbind" --collection_name immune

# 查看集合
uv run kb list-collections

# 删除集合
uv run kb drop-collection --collection_name collection_name
```

---

## IgBLAST 服务部署

### 项目目录结构

```
mcp_Igblast/
├── config/
│   └── config.py                    ⚠️ 需要修改
├── igblast_changeO/
│   ├── igblast/                     ✓ 数据库文件（直接复制）
│   │   ├── database/                IgBLAST参考数据库
│   │   └── optional_file/           辅助文件
│   ├── MakeDb.py                    ⚠️ 需要修改shebang
│   ├── AssignGenes.py               ⚠️ 需要修改shebang
│   ├── CreateGermlines.py           ⚠️ 需要修改shebang
│   ├── DefineClones.py              ⚠️ 需要修改shebang
│   └── input/                       示例输入文件
├── docs/
│   └── requirements.txt             Python依赖列表
├── output/                          输出目录（自动创建）
└── igblast_mcp_server.py            ⚠️ 需要修改
```

### 依赖环境要求

**系统依赖：**
- Python: 3.8+（推荐 3.12.11）
- IgBLAST: 1.17.0+（推荐 1.22.0）
- ChangeO: 1.2.0+（推荐 1.3.4）

**Python 包：**
- fastmcp: >= 0.2.0
- pandas: >= 2.0.0

### 迁移步骤

#### 步骤 1: 复制目录到新服务器

```bash
rsync -avz /data_new/workspace/antibody_gen/mcp_Igblast/ \
    user@新服务器IP:/新路径/mcp_Igblast/
```

#### 步骤 2: 安装环境和依赖

```bash
# 创建conda环境
conda create -n antibody_venv python=3.12
conda activate antibody_venv

# 安装生物信息学工具
conda install -c bioconda igblast changeo

# 安装Python包
pip install fastmcp pandas

# 验证安装
igblastn -version
MakeDb.py -h
```

#### 步骤 3: 修改配置文件

**方法 A: 使用自动化脚本（推荐）**

```bash
cd /新路径/mcp_Igblast

OLD_PATH="/data_new/workspace/antibody_gen/mcp_Igblast"
NEW_PATH="/新路径/mcp_Igblast"

# 批量替换路径
sed -i "s|$OLD_PATH|$NEW_PATH|g" config/config.py
sed -i "s|$OLD_PATH|$NEW_PATH|g" igblast_mcp_server.py

# 自动获取并更新conda MakeDb.py路径
MAKEDB_PATH=$(which MakeDb.py)
sed -i "s|_conda_makedb = Path(\".*MakeDb.py\")|_conda_makedb = Path(\"$MAKEDB_PATH\")|" igblast_mcp_server.py
```

**方法 B: 手动修改**

1. **编辑 `config/config.py`**：修改第10行和第26行的路径
2. **编辑 `igblast_mcp_server.py`**：修改第333-334行的 MakeDb.py 路径

#### 步骤 4: 修改 ChangeO 脚本

```bash
cd /新路径/mcp_Igblast/igblast_changeO

# 批量修改4个文件的第一行
find . -maxdepth 1 -name "*.py" -type f -exec sed -i '1s|^#!.*|#!/usr/bin/env python|' {} \;

# 验证修改
head -1 MakeDb.py
```

#### 步骤 5: 验证配置

```bash
cd /新路径/mcp_Igblast

# 验证配置文件
python -c "from config.config import IGBLAST_BASE, OUTPUT_DIR; print('✓ 配置导入成功')"

# 验证服务器模块
python -c "import igblast_mcp_server; print('✓ 服务器模块导入成功')"

# 验证数据库文件
ls igblast_changeO/igblast/database/ | head -5
```

#### 步骤 6: 启动服务器

```bash
cd /新路径/mcp_Igblast
conda activate antibody_venv
python igblast_mcp_server.py
```

**预期输出:**
```
INFO - IgBLAST V(D)J Analysis Server started
INFO - Listening on http://0.0.0.0:8110
```

---

## 代码运行及测试

### 1. 修改文件保存目录

打开 `agent/usecases/immunity/common/utils.py` 文件，修改第27行的目录，改为你本地的目录。

### 2. 配置模型及 API Key

打开 `agent/usecases/immunity/config` 目录，根据需要将 `immunity_config_qwen` 或 `immunity_config_gpt4.1` 的配置项粘贴到 `immunity_config.py` 文件中，并修改 API Key。

### 3. 配置可调用的 MCP 服务

打开 `immunity_config.py` 文件，修改配置项中的：

```python
"mcp_config": {
    "service_ids": [
        "igblast",
        "metabcr",
        "lineage_analysis",
        "af3",
        "integrateBcrData",
        "bioinformatics"
    ]
}
```

在 `service_ids` 中添加已启动的 MCP 服务名。推荐至少启动以下六个服务：
- `igblast`
- `metabcr`
- `lineage_analysis` (原 `flu`)
- `af3`
- `integrateBcrData`
- `bioinformatics`

### 4. 运行代码

```bash
cd agent/usecases/immunity
python start_improved_workflow.py --query "your question" --file_url "your init csv file"
```

**示例：**
```bash
python start_improved_workflow.py \
    --query "please design a computational method to identifiy broadly neutralizing antibodies against H5N1." \
    --file_url "https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/flu-simple.xlsx"
```

### 5. 测试数据

为了方便测试，这里提供了一些样例数据：

- **初始CSV 文件：**
  ```
  https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/flu-all_data_with_predict_and_feature.csv
  ```

- **抗原CSV 文件：**
  ```
  https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/Copy%20of%20flu_bind_variant_seq.xlsx
  ```

- **RDS 文件：**
  ```
  https://immunity-test.oss-cn-beijing.aliyuncs.com/artifacts/5b1bcb5c-1079-4d2c-a276-759152acbf54/20240923_flu_B_annotation.rds
  ```

- **实验数据 1：**
  ```
  https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/flu_simple%28origin_flu-binding_neutralizations%29.xlsx
  ```

- **实验数据 2：**
  ```
  https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/flu_second_simple.xlsx
  ```

### 6. 交互注意事项

- **整体流程耗时较长**，可能需要几天的运行时间

- **metabcr 工具**：如果需要补充抗原文件，抗原文件需至少包含 `variant_seq` 列

- **integrate_bcr_data_complet 工具**：需要指定参数：
  ```json
  {
    "args": {
      "csv_fields": "main_name",
      "rds_fields": "main_name"
    }
  }
  ```
  用于指明 CSV 与 RDS 的映射关系

- **工具调用前可修改参数**：系统会在每次工具调用前请求用户确认，你可以修改参数后再确认执行

---

## 服务管理

### 启动所有服务

```bash
# 启动 Qdrant
cd /path/to/your/kb
docker compose up -d

# 启动 Ollama
ollama serve &

# 启动 IgBLAST 服务
cd /path/to/mcp_Igblast
conda activate antibody_venv
python igblast_mcp_server.py
```

### 停止服务

```bash
# 停止 Qdrant
docker compose down

# 停止 Ollama
pkill ollama

# 停止 IgBLAST 服务
# 使用 Ctrl+C 或关闭终端
```

### 查看日志

```bash
# Qdrant 日志
docker compose logs -f qdrant

# 查看最近 100 行
docker compose logs --tail=100 qdrant
```

---

## 故障排查

### 常见问题

1. **uv 命令未找到**
   - 确保已正确安装 uv 并添加到 PATH
   - Windows: 重启 PowerShell 或重新打开终端
   - Linux/Mac: 执行 `source ~/.bashrc` 或 `source ~/.zshrc`

2. **Qdrant 连接失败**
   - 检查 Qdrant 服务是否正常运行：`docker compose ps`
   - 确认环境变量 `QDRANT_HOST` 设置正确
   - 检查防火墙设置

3. **IgBLAST 服务无法启动**
   - 确认 conda 环境已激活
   - 检查配置文件路径是否正确
   - 验证依赖工具是否已安装：`igblastn -version`

4. **模型 API Key 错误**
   - 检查 `immunity_config.py` 中的 API Key 配置
   - 确认 API Key 有效且有足够的配额

---

## 更多信息

- KB 知识库详细文档：参见 `kb/README.md`
- IgBLAST 部署详细文档：参见 `mcp_Igblast/docs/`
- 项目架构说明：参见 `doc/`

