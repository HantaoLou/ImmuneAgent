# Antibody Generation and Immune Analysis System - English Documentation

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Installation and Deployment](#environment-installation-and-deployment)
3. [KB Knowledge Base System Deployment](#kb-knowledge-base-system-deployment)
4. [IgBLAST Service Deployment](#igblast-service-deployment)
5. [Code Execution and Testing](#code-execution-and-testing)

---

## Quick Start

If you have all components ready, follow these steps to quickly start:

1. **Install Dependencies** (see [Environment Installation and Deployment](#environment-installation-and-deployment))
2. **Deploy KB Knowledge Base** (see [KB Knowledge Base System Deployment](#kb-knowledge-base-system-deployment))
3. **Deploy IgBLAST Service** (see [IgBLAST Service Deployment](#igblast-service-deployment))
4. **Configure and Run Code** (see [Code Execution and Testing](#code-execution-and-testing))

---

## Environment Installation and Deployment

### 1. Install uv

**Windows:**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Linux/Mac:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Verify installation:**
```bash
uv --version
```

### 2. Install Project Dependencies

```bash
cd agent
uv sync
```

---

## KB Knowledge Base System Deployment

### Quick Start

If you already have the kb directory, follow these steps for quick deployment:

```bash
# 1. Modify the data directory path in docker-compose.yaml (line 17)
# Change /data_new/wyl/qdrant_data to your actual path

# 2. Enter the project directory and start Qdrant
cd /path/to/your/kb
docker compose up -d

# 3. Set Qdrant connection to local (Important!)
export QDRANT_HOST=localhost

# 4. Install uv and configure PATH
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 5. Install dependencies
uv sync

# 6. Install Ollama and pull models
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &  # Run in background
ollama pull nomic-embed-text

# 7. Verify deployment
uv run kb list-collections
```

### Detailed Deployment Steps

#### 1. Docker and Docker Compose Installation

```bash
# Update system packages
sudo apt update
sudo apt upgrade -y

# Install necessary dependencies
sudo apt install -y ca-certificates curl gnupg lsb-release

# Add Docker official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add current user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Verify installation
docker --version
docker compose version
```

#### 2. Qdrant Vector Database Deployment

**Configuration File Modification:**

In `kb/docker-compose.yaml`, **must modify** line 17 data directory path:

```yaml
volumes:
  - /data_new/wyl/qdrant_data:/qdrant/storage
```

Change to your actual path, for example:
```yaml
volumes:
  - /your/storage/path/qdrant_data:/qdrant/storage
```

**Start Qdrant:**
```bash
cd /path/to/your/kb
docker compose up -d
docker compose ps
```

**Verify Service:**
```bash
curl http://localhost:6333/healthz
curl http://localhost:6333/collections
```

#### 3. KB Application Deployment

**Install Dependencies:**
```bash
cd /path/to/your/kb
uv sync
```

**Configure Connection Parameters:**
```bash
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export QDRANT_GRPC_PORT=6334
```

**Install Ollama and Models:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start service
ollama serve &

# Pull models
ollama pull nomic-embed-text     # embedding model (required)
ollama pull gemma3:4b            # default summarization model (recommended)
ollama pull qwq:latest           # advanced reasoning model (optional)
```

#### 4. Using the Vector Store

```bash
# Load documents
uv run kb load-doc --path ./library --collection_name collection2

# Query documents
uv run kb query --query "gearbind" --collection_name immune

# View collections
uv run kb list-collections

# Delete collection
uv run kb drop-collection --collection_name collection_name
```

---

## IgBLAST Service Deployment

### Project Directory Structure

```
mcp_Igblast/
├── config/
│   └── config.py                    ⚠️ Requires modification
├── igblast_changeO/
│   ├── igblast/                     ✓ Database files (copy directly)
│   │   ├── database/                IgBLAST reference database
│   │   └── optional_file/           Auxiliary files
│   ├── MakeDb.py                    ⚠️ Requires shebang modification
│   ├── AssignGenes.py               ⚠️ Requires shebang modification
│   ├── CreateGermlines.py           ⚠️ Requires shebang modification
│   ├── DefineClones.py              ⚠️ Requires shebang modification
│   └── input/                       Example input files
├── docs/
│   └── requirements.txt             Python dependencies list
├── output/                          Output directory (auto-created)
└── igblast_mcp_server.py            ⚠️ Requires modification
```

### Dependency Requirements

**System Dependencies:**
- Python: 3.8+ (recommended 3.12.11)
- IgBLAST: 1.17.0+ (recommended 1.22.0)
- ChangeO: 1.2.0+ (recommended 1.3.4)

**Python Packages:**
- fastmcp: >= 0.2.0
- pandas: >= 2.0.0

### Migration Steps

#### Step 1: Copy Directory to New Server

```bash
rsync -avz /data_new/workspace/antibody_gen/mcp_Igblast/ \
    user@new_server_IP:/new/path/mcp_Igblast/
```

#### Step 2: Install Environment and Dependencies

```bash
# Create conda environment
conda create -n antibody_venv python=3.12
conda activate antibody_venv

# Install bioinformatics tools
conda install -c bioconda igblast changeo

# Install Python packages
pip install fastmcp pandas

# Verify installation
igblastn -version
MakeDb.py -h
```

#### Step 3: Modify Configuration Files

**Method A: Using Automated Script (Recommended)**

```bash
cd /new/path/mcp_Igblast

OLD_PATH="/data_new/workspace/antibody_gen/mcp_Igblast"
NEW_PATH="/new/path/mcp_Igblast"

# Batch replace paths
sed -i "s|$OLD_PATH|$NEW_PATH|g" config/config.py
sed -i "s|$OLD_PATH|$NEW_PATH|g" igblast_mcp_server.py

# Auto-detect and update conda MakeDb.py path
MAKEDB_PATH=$(which MakeDb.py)
sed -i "s|_conda_makedb = Path(\".*MakeDb.py\")|_conda_makedb = Path(\"$MAKEDB_PATH\")|" igblast_mcp_server.py
```

**Method B: Manual Modification**

1. **Edit `config/config.py`**: Modify lines 10 and 26 paths
2. **Edit `igblast_mcp_server.py`**: Modify lines 333-334 MakeDb.py paths

#### Step 4: Modify ChangeO Scripts

```bash
cd /new/path/mcp_Igblast/igblast_changeO

# Batch modify line 1 of 4 files
find . -maxdepth 1 -name "*.py" -type f -exec sed -i '1s|^#!.*|#!/usr/bin/env python|' {} \;

# Verify modification
head -1 MakeDb.py
```

#### Step 5: Verify Configuration

```bash
cd /new/path/mcp_Igblast

# Verify config file
python -c "from config.config import IGBLAST_BASE, OUTPUT_DIR; print('✓ Config imported successfully')"

# Verify server module
python -c "import igblast_mcp_server; print('✓ Server module imported successfully')"

# Verify database files
ls igblast_changeO/igblast/database/ | head -5
```

#### Step 6: Start Server

```bash
cd /new/path/mcp_Igblast
conda activate antibody_venv
python igblast_mcp_server.py
```

**Expected output:**
```
INFO - IgBLAST V(D)J Analysis Server started
INFO - Listening on http://0.0.0.0:8110
```

---

## Code Execution and Testing

### 1. Modify File Save Directory

Open `agent/usecases/immunity/common/utils.py` file and modify line 27 to your local directory.

### 2. Configure Model and API Key

Open `agent/usecases/immunity/config` directory, copy configuration items from `immunity_config_qwen` or `immunity_config_gpt4.1` to `immunity_config.py` file, and modify the API Key.

### 3. Configure Callable MCP Services

Open `immunity_config.py` file and modify the configuration:

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

Add started MCP service names to `service_ids`. It is recommended to start at least the following six services:
- `igblast`
- `metabcr`
- `lineage_analysis` (formerly `flu`)
- `af3`
- `integrateBcrData`
- `bioinformatics`

### 4. Run Code

```bash
cd agent/usecases/immunity
python start_improved_workflow.py --query "your question" --file_url "your init csv file"
```

**Example:**
```bash
python start_improved_workflow.py \
    --query "please design a computational method to identifiy broadly neutralizing antibodies against H5N1." \
    --file_url "https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/flu-simple.xlsx"
```

### 5. Test Data

For testing convenience, sample data is provided:

- **Initial CSV File:**
  ```
  https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/flu-all_data_with_predict_and_feature.csv
  ```

- **Antigen CSV File:**
  ```
  https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/Copy%20of%20flu_bind_variant_seq.xlsx
  ```

- **RDS File:**
  ```
  https://immunity-test.oss-cn-beijing.aliyuncs.com/artifacts/5b1bcb5c-1079-4d2c-a276-759152acbf54/20240923_flu_B_annotation.rds
  ```

- **Experimental Data 1:**
  ```
  https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/flu_simple%28origin_flu-binding_neutralizations%29.xlsx
  ```

- **Experimental Data 2:**
  ```
  https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/flu_second_simple.xlsx
  ```

### 6. Interactive Notes

- **Overall process takes a long time**, may require several days of runtime

- **metabcr tool**: If additional antigen files are needed, the antigen file must include at least a `variant_seq` column

- **integrate_bcr_data_complet tool**: Need to specify parameters:
  ```json
  {
    "args": {
      "csv_fields": "main_name",
      "rds_fields": "main_name"
    }
  }
  ```
  Used to specify the mapping relationship between CSV and RDS

- **Tool parameters can be modified before calling**: The system will request user confirmation before each tool call, and you can modify parameters before confirming execution

---

## Service Management

### Start All Services

```bash
# Start Qdrant
cd /path/to/your/kb
docker compose up -d

# Start Ollama
ollama serve &

# Start IgBLAST service
cd /path/to/mcp_Igblast
conda activate antibody_venv
python igblast_mcp_server.py
```

### Stop Services

```bash
# Stop Qdrant
docker compose down

# Stop Ollama
pkill ollama

# Stop IgBLAST service
# Use Ctrl+C or close terminal
```

### View Logs

```bash
# Qdrant logs
docker compose logs -f qdrant

# View last 100 lines
docker compose logs --tail=100 qdrant
```

---

## Troubleshooting

### Common Issues

1. **uv command not found**
   - Ensure uv is correctly installed and added to PATH
   - Windows: Restart PowerShell or reopen terminal
   - Linux/Mac: Execute `source ~/.bashrc` or `source ~/.zshrc`

2. **Qdrant connection failed**
   - Check if Qdrant service is running: `docker compose ps`
   - Confirm environment variable `QDRANT_HOST` is set correctly
   - Check firewall settings

3. **IgBLAST service cannot start**
   - Ensure conda environment is activated
   - Check if configuration file paths are correct
   - Verify dependency tools are installed: `igblastn -version`

4. **Model API Key error**
   - Check API Key configuration in `immunity_config.py`
   - Confirm API Key is valid and has sufficient quota

---

## More Information

- KB Knowledge Base Detailed Documentation: See `kb/README.md`
- IgBLAST Deployment Detailed Documentation: See `mcp_Igblast/docs/`
- Project Architecture: See `doc/`

