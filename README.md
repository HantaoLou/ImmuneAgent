# Antibody Generation and Immune Analysis System

A comprehensive computational platform for B cell function analysis and antibody structural analysis, integrating knowledge base retrieval, IgBLAST V(D)J analysis, MetaBCR prediction, lineage analysis, and AlphaFold3 structure prediction.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [System Requirements](#system-requirements)
3. [Quick Start Guide](#quick-start-guide)
4. [Deployment Checklist](#deployment-checklist)
5. [Environment Setup](#environment-setup)
6. [Service Deployment](#service-deployment)
   - [KB Knowledge Base System](#kb-knowledge-base-system)
   - [IgBLAST Service](#igblast-service)
   - [MetaBCR Service](#metabcr-service)
   - [File Utils Service](#file-utils-service)
   - [AlphaFold3 Service](#alphafold3-service)
   - [Lineage Analysis Service](#lineage-analysis-service)
   - [Bioinformatics Service](#bioinformatics-service)
   - [Data Integration Service](#data-integration-service-integratebcrdata)
7. [Configuration and Execution](#configuration-and-execution)
8. [Service Management](#service-management)
9. [Troubleshooting](#troubleshooting)
10. [Additional Resources](#additional-resources)

---

## System Overview

The Antibody Generation and Immune Analysis System provides comprehensive computational tools for B cell function analysis and structural analysis of antibodies. The system integrates multiple components including knowledge base retrieval, IgBLAST V(D)J analysis, MetaBCR prediction, lineage analysis, and AlphaFold3 structure prediction.

![System Overview](doc/assets/Figure1-nbt.ai.pdf)

**Figure 1: System Architecture and Workflow**

This diagram illustrates the core components and workflow of the antibody generation and analysis system:

- **B Cell Function Analysis**: Comprehensive analysis of B cell receptor (BCR) repertoires, including V(D)J gene annotation, clonal lineage tracing, and functional characterization
- **Structural Analysis**: Protein structure prediction and analysis using AlphaFold3, enabling 3D modeling of antibody-antigen interactions
- **Integrated Pipeline**: Seamless integration of multiple MCP (Model Context Protocol) services for end-to-end antibody design and evaluation

The system supports the complete workflow from raw sequencing data to antibody design recommendations, combining bioinformatics tools, machine learning models, and knowledge base retrieval for intelligent decision-making.

---

## System Requirements

### Prerequisites

Before starting the deployment, ensure you have:

- **Operating System**: Linux (Ubuntu 20.04+ recommended) or macOS, Windows with WSL2
- **Python**: 3.8+ (3.12.11 recommended)
- **Docker**: 20.10+ and Docker Compose 2.0+
- **Conda/Miniconda**: For managing bioinformatics tool environments
- **Memory**: Minimum 16GB RAM (32GB+ recommended for full functionality)
- **Storage**: At least 50GB free space for databases and models
- **Network**: Internet connection for downloading models and dependencies

### Large File Downloads

> **Important**: The following large model files must be downloaded separately due to their size. These files are required for MetaBCR service functionality.

**Required Files:**

1. **ProtBERT Model** (`pytorch_model.bin`)
   - **Path**: `mcp_metabcr/External/prot_bert/pytorch_model.bin`
   - **Download URL**: https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/pytorch_model.bin
   - **Download command**:
     ```bash
     cd mcp_metabcr/External/prot_bert
     wget https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/pytorch_model.bin
     # Or use curl:
     curl -O https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/pytorch_model.bin
     ```

2. **RSV-NEU Model** (`fold4.pth.filepart`)
   - **Path**: `mcp_metabcr/Models/0224-rsv-neu/fold4.pth.filepart`
   - **Download URL**: https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/fold4.pth.filepart
   - **Download command**:
     ```bash
     cd mcp_metabcr/Models/0224-rsv-neu
     wget https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/fold4.pth.filepart
     # Or use curl:
     curl -O https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/fold4.pth.filepart
     ```

**Note**: Ensure these directories exist before downloading. Create them if necessary:
```bash
mkdir -p mcp_metabcr/External/prot_bert
mkdir -p mcp_metabcr/Models/0224-rsv-neu
```

### Required Services

The system requires the following services to be running:

1. **Qdrant Vector Database** - For knowledge base storage
2. **Ollama** - For embedding and language models
3. **IgBLAST Service** - For V(D)J analysis
4. **MCP Services** - Multiple Model Context Protocol services (metabcr, lineage_analysis, af3, etc.)

---

## Quick Start Guide

> **Note**: This guide assumes you have all prerequisites installed. For first-time setup, please follow the detailed deployment sections below.

### Step 1: Install Base Dependencies

```bash
# Install uv (Python package manager)
# Windows (PowerShell):
irm https://astral.sh/uv/install.ps1 | iex

# Linux/Mac:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
cd agent
uv sync
```

### Step 2: Deploy KB Knowledge Base

```bash
cd kb
# Modify docker-compose.yaml data directory path (line 17)
docker compose up -d
export QDRANT_HOST=localhost
uv sync
ollama pull nomic-embed-text
```

### Step 3: Deploy IgBLAST Service

```bash
cd mcp_Igblast
conda create -n antibody_venv python=3.12
conda activate antibody_venv
conda install -c bioconda igblast changeo
pip install fastmcp pandas
# Configure paths in config/config.py and igblast_mcp_server.py
python igblast_mcp_server.py
```

### Step 4: Configure and Run

```bash
cd agent/usecases/immunity
# Edit immunity_config.py with your API keys and MCP service IDs
python start_improved_workflow.py --query "your question" --file_url "your data file"
```

---

## Deployment Checklist

Use this checklist to ensure all components are properly deployed:

### Environment Setup
- [ ] `uv` installed and accessible in PATH
- [ ] Project dependencies installed (`cd agent && uv sync`)
- [ ] Python 3.8+ available

### KB Knowledge Base
- [ ] Docker and Docker Compose installed
- [ ] Qdrant data directory configured in `kb/docker-compose.yaml`
- [ ] Qdrant service running (`docker compose ps`)
- [ ] `QDRANT_HOST=localhost` environment variable set
- [ ] KB application dependencies installed (`cd kb && uv sync`)
- [ ] Ollama installed and running
- [ ] Required models downloaded (`nomic-embed-text` at minimum)

### IgBLAST Service
- [ ] Conda environment created (`antibody_venv`)
- [ ] IgBLAST and ChangeO tools installed
- [ ] Configuration files updated with correct paths
- [ ] ChangeO scripts modified (shebang lines)
- [ ] Service starts successfully on port 8110

### MetaBCR Service
- [ ] MetaBCR directory structure exists
- [ ] ProtBERT model downloaded: `mcp_metabcr/External/prot_bert/pytorch_model.bin`
- [ ] RSV-NEU model downloaded: `mcp_metabcr/Models/0224-rsv-neu/fold4.pth.filepart`
- [ ] Model files verified (check file sizes)

### File Utils Service
- [ ] File Utils MCP service dependencies installed
- [ ] Service starts successfully

### AlphaFold3 Service
- [ ] AlphaFold3 service dependencies installed
- [ ] Service starts successfully

### Lineage Analysis Service
- [ ] Lineage Analysis service dependencies installed
- [ ] Service starts successfully

### Bioinformatics Service
- [ ] Bioinformatics service dependencies installed
- [ ] R environment configured (if using R scripts)
- [ ] Service starts successfully

### Data Integration Service
- [ ] Data Integration service dependencies installed
- [ ] R environment configured (if using R scripts)
- [ ] Service starts successfully

### Application Configuration
- [ ] File save directory configured in `agent/usecases/immunity/common/utils.py`
- [ ] API keys configured in `immunity_config.py`
- [ ] MCP service IDs configured in `immunity_config.py`
- [ ] All required MCP services started

### File Utils Service
- [ ] File Utils MCP service dependencies installed
- [ ] Service starts successfully

### Verification
- [ ] Qdrant health check: `curl http://localhost:6333/healthz`
- [ ] KB collections accessible: `uv run kb list-collections`
- [ ] IgBLAST service responding on port 8110
- [ ] Test workflow execution successful

---

## Environment Setup

### 1. Install uv Package Manager

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Linux/Mac:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
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

This will install all Python dependencies required by the main application.

---

## Service Deployment

### KB Knowledge Base System

The KB (Knowledge Base) system provides vector storage and retrieval capabilities using Qdrant and Ollama.

#### Quick Deployment (If KB directory exists)

```bash
# 1. Modify data directory in kb/docker-compose.yaml (line 17)
# Change /data_new/wyl/qdrant_data to your actual path

# 2. Start Qdrant
cd kb
docker compose up -d

# 3. Set environment variables
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export QDRANT_GRPC_PORT=6334

# 4. Install dependencies
uv sync

# 5. Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull nomic-embed-text

# 6. Verify deployment
uv run kb list-collections
```

#### Detailed Deployment Steps

##### Step 1: Install Docker and Docker Compose

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Docker dependencies
sudo apt install -y ca-certificates curl gnupg lsb-release

# Add Docker GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Verify installation
docker --version
docker compose version
```

##### Step 2: Deploy Qdrant Vector Database

**Configure storage path** in `kb/docker-compose.yaml` (line 17):

```yaml
volumes:
  - /your/storage/path/qdrant_data:/qdrant/storage
```

**Start Qdrant:**
```bash
cd kb
docker compose up -d
docker compose ps
```

**Verify service:**
```bash
curl http://localhost:6333/healthz
curl http://localhost:6333/collections
```

##### Step 3: Deploy KB Application

```bash
cd kb
uv sync
export QDRANT_HOST=localhost
```

##### Step 4: Install Ollama and Models

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start service
ollama serve &

# Pull required models
ollama pull nomic-embed-text     # Required: embedding model
ollama pull gemma3:4b            # Recommended: summarization model
ollama pull qwq:latest           # Optional: advanced reasoning model
```

##### Step 5: Using the Vector Store

```bash
# Load documents into collection
uv run kb load-doc --path ./library --collection_name collection2

# Query documents
uv run kb query --query "gearbind" --collection_name immune

# List all collections
uv run kb list-collections

# Delete a collection
uv run kb drop-collection --collection_name collection_name
```

#### KB Service Management

**Start KB services:**
```bash
cd kb
docker compose up -d          # Start Qdrant
ollama serve &                # Start Ollama (if not running)
```

**Stop KB services:**
```bash
cd kb
docker compose down           # Stop Qdrant
pkill ollama                  # Stop Ollama
```

**View logs:**
```bash
docker compose logs -f qdrant
docker compose logs --tail=100 qdrant
```

---

### IgBLAST Service

The IgBLAST service provides V(D)J gene annotation and BCR analysis capabilities using IgBLAST and ChangeO tools.

#### Features

The IgBLAST service provides the following tools:

**V(D)J Analysis:**
- `analyze_vdj_batch` - Perform V(D)J recombination analysis using IgBLAST + ChangeO
  - Automatically splits large files into batches for efficient processing
  - Returns AIRR (Adaptive Immune Receptor Repertoire) format results
  - Supports FASTA input files with antibody sequences
  - Handles heavy and light chain analysis
  - Generates comprehensive V(D)J gene annotations

**CDR3 Extraction:**
- `extract_cdr3_from_airr` - Extract CDR3 information from AIRR format results
  - Supports multiple input formats: AIRR records array, local CSV/JSON files, HTTP/HTTPS URLs
  - Extracts junction sequences (CDR3 regions)
  - Processes large files with streaming progress updates
  - Handles various AIRR format variants

All tools support:
- Local file paths and HTTP/HTTPS URLs
- Large file processing with automatic batching
- Streaming progress updates via SSE
- Comprehensive error handling and validation

#### Project Structure

```
mcp_Igblast/
├── config/
│   └── config.py                    ⚠️ Requires path modification
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
│   └── requirements.txt             Python dependencies
├── output/                          Output directory (auto-created)
└── igblast_mcp_server.py            ⚠️ Requires path modification
```

#### Dependency Requirements

**System Dependencies:**
- Python: 3.8+ (3.12.11 recommended)
- IgBLAST: 1.17.0+ (1.22.0 recommended)
- ChangeO: 1.2.0+ (1.3.4 recommended)

**Python Packages:**
- fastmcp: >= 0.2.0
- pandas: >= 2.0.0

#### Deployment Steps

##### Step 1: Create Conda Environment

```bash
conda create -n antibody_venv python=3.12
conda activate antibody_venv
```

##### Step 2: Install Bioinformatics Tools

```bash
# Install IgBLAST and ChangeO via conda
conda install -c bioconda igblast changeo

# Install Python packages
pip install fastmcp pandas

# Verify installation
igblastn -version
MakeDb.py -h
```

##### Step 3: Configure Paths

**Method A: Automated Script (Recommended)**

```bash
cd mcp_Igblast

OLD_PATH="/data_new/workspace/antibody_gen/mcp_Igblast"
NEW_PATH="$(pwd)"

# Batch replace paths
sed -i "s|$OLD_PATH|$NEW_PATH|g" config/config.py
sed -i "s|$OLD_PATH|$NEW_PATH|g" igblast_mcp_server.py

# Auto-detect and update MakeDb.py path
MAKEDB_PATH=$(which MakeDb.py)
sed -i "s|_conda_makedb = Path(\".*MakeDb.py\")|_conda_makedb = Path(\"$MAKEDB_PATH\")|" igblast_mcp_server.py
```

**Method B: Manual Modification**

1. Edit `config/config.py`: Update paths on lines 10 and 26
2. Edit `igblast_mcp_server.py`: Update MakeDb.py paths on lines 333-334

##### Step 4: Modify ChangeO Scripts

```bash
cd mcp_Igblast/igblast_changeO

# Batch modify shebang lines
find . -maxdepth 1 -name "*.py" -type f -exec sed -i '1s|^#!.*|#!/usr/bin/env python|' {} \;

# Verify modification
head -1 MakeDb.py
```

##### Step 5: Verify Configuration

```bash
cd mcp_Igblast

# Verify config import
python -c "from config.config import IGBLAST_BASE, OUTPUT_DIR; print('✓ Config OK')"

# Verify server module
python -c "import igblast_mcp_server; print('✓ Server module OK')"

# Verify database files
ls igblast_changeO/igblast/database/ | head -5
```

##### Step 6: Start Server

```bash
cd mcp_Igblast
conda activate antibody_venv
python igblast_mcp_server.py
```

**Expected output:**
```
INFO - IgBLAST V(D)J Analysis Server started
INFO - Listening on http://0.0.0.0:8110
```

#### IgBLAST Service Management

**Start service:**
```bash
cd mcp_Igblast
conda activate antibody_venv
python igblast_mcp_server.py
```

**Stop service:**
- Press `Ctrl+C` in the terminal running the server

**Run in background (Linux/Mac):**
```bash
nohup python igblast_mcp_server.py > igblast.log 2>&1 &
```

---

### MetaBCR Service

The MetaBCR service provides BCR binding prediction capabilities using deep learning models, including CNN, GNN, and BERT-based architectures.

#### Features

The MetaBCR service provides the following tools:

**Antibody-Antigen Binding Prediction:**
- `metabcr` - Predict antibody-antigen binding affinity using deep learning
  - Supports multiple model architectures (CNN, GNN, BERT-based)
  - Configurable for various tasks and datasets
  - Handles antibody and antigen sequence inputs
  - Returns binding prediction scores
  - Supports batch processing for multiple antibody-antigen pairs
  - Streaming progress updates for long-running predictions

All tools support:
- Local file paths and HTTP/HTTPS URLs
- Large dataset processing with streaming progress
- Multiple model configurations
- Comprehensive error handling

#### Prerequisites

Before deploying MetaBCR service, you must download the required large model files:

**1. Download ProtBERT Model**

```bash
# Create directory if it doesn't exist
mkdir -p mcp_metabcr/External/prot_bert

# Download the model file
cd mcp_metabcr/External/prot_bert
wget https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/pytorch_model.bin

# Or using curl:
curl -O https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/pytorch_model.bin

# Verify download (check file size)
ls -lh pytorch_model.bin
```

**2. Download RSV-NEU Model**

```bash
# Create directory if it doesn't exist
mkdir -p mcp_metabcr/Models/0224-rsv-neu

# Download the model file
cd mcp_metabcr/Models/0224-rsv-neu
wget https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/fold4.pth.filepart

# Or using curl:
curl -O https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/fold4.pth.filepart

# Verify download (check file size)
ls -lh fold4.pth.filepart
```

> **Important**: These model files are large and must be downloaded separately. Ensure you have sufficient disk space and a stable internet connection before downloading.

#### Deployment Steps

1. **Install Dependencies**

   ```bash
   cd mcp_metabcr
   pip install -r requirements.txt
   ```

2. **Verify Model Files**

   ```bash
   # Check ProtBERT model
   ls -lh External/prot_bert/pytorch_model.bin
   
   # Check RSV-NEU model
   ls -lh Models/0224-rsv-neu/fold4.pth.filepart
   ```

3. **Start MetaBCR Service**

   ```bash
   cd mcp_metabcr
   python metabcr_mcp_server.py
   ```

#### MetaBCR Service Management

**Start service:**
```bash
cd mcp_metabcr
python metabcr_mcp_server.py
```

**Stop service:**
- Press `Ctrl+C` in the terminal running the server

**Run in background (Linux/Mac):**
```bash
cd mcp_metabcr
nohup python metabcr_mcp_server.py > metabcr.log 2>&1 &
```

---

### File Utils Service

The File Utils service provides file download and format conversion capabilities, supporting various file format transformations commonly needed in bioinformatics workflows.

#### Features

The File Utils service provides the following tools:

**File Download:**
- `download_url` - Download files from HTTP/HTTPS URLs with progress tracking

**Format Conversion:**
- `convert_csv_to_fasta` - Convert CSV files to FASTA format
- `convert_excel_to_fasta` - Convert Excel files (XLSX/XLS) to FASTA format
- `convert_xlsx_to_fasta` - Convert XLSX files to FASTA format
- `convert_xls_to_fasta` - Convert XLS files to FASTA format
- `convert_xlsx_to_csv` - Convert XLSX files to CSV format
- `convert_xls_to_csv` - Convert XLS files to CSV format
- `convert_csv_to_xlsx` - Convert CSV files to XLSX format

**File Operations:**
- `merge_csv_cartesian` - Merge two CSV files using Cartesian product (all combinations)
- `merge_csv_by_key` - Merge two CSV files by key column or row index
- `create_csv` - Create blank CSV files with optional headers

All tools support:
- Local file paths and HTTP/HTTPS URLs
- Large file processing with streaming progress updates
- Automatic format detection and error handling

#### Deployment Steps

1. **Install Dependencies**

   ```bash
   cd mcp_file_utils
   pip install fastmcp pandas openpyxl xlrd requests
   ```

   **Required packages:**
   - `fastmcp` - MCP server framework
   - `pandas` - Data manipulation and CSV/Excel handling
   - `openpyxl` - Excel XLSX file support
   - `xlrd` - Excel XLS file support (legacy format)
   - `requests` - HTTP file downloads

2. **Start File Utils Service**

   ```bash
   cd mcp_file_utils
   python file_utils_mcp_server.py
   ```

   **Expected output:**
   ```
   INFO - File Utils MCP Server started
   INFO - Listening on http://0.0.0.0:PORT
   ```

#### File Utils Service Management

**Start service:**
```bash
cd mcp_file_utils
python file_utils_mcp_server.py
```

**Stop service:**
- Press `Ctrl+C` in the terminal running the server

**Run in background (Linux/Mac):**
```bash
cd mcp_file_utils
nohup python file_utils_mcp_server.py > file_utils.log 2>&1 &
```

#### Usage Examples

**Download a file:**
```python
# The service will be called via MCP protocol
# Example: Download a CSV file from URL
download_url(url="https://example.com/data.csv", output_path="./data.csv")
```

**Convert CSV to FASTA:**
```python
# Convert CSV file to FASTA format
convert_csv_to_fasta(
    input_file="antibodies.csv",
    output_file="antibodies.fasta",
    sequence_column="sequence"  # Auto-detected if not specified
)
```

**Merge CSV files:**
```python
# Merge two CSV files by key
merge_csv_by_key(
    input_file1="antibodies.csv",
    input_file2="antigens.csv",
    output_file="merged.csv",
    key_column="id"  # Optional: merge by key column
)
```

---

### AlphaFold3 Service

The AlphaFold3 service provides protein structure prediction capabilities for antibody sequences.

#### Features

The AlphaFold3 service provides the following tools:

**Structure Prediction:**
- `alphafold3` - Predict 3D structure of antibody sequences using AlphaFold3
  - Reads Excel files containing antibody sequences (heavy and light chains)
  - Uses state-of-the-art deep learning model for protein structure prediction
  - Generates PDB format structure files
  - Supports batch processing of multiple antibodies
  - Streaming progress updates for structure prediction

All tools support:
- Excel file input (XLSX format)
- Heavy and light chain sequence inputs
- PDB format output for visualization
- Large batch processing with progress tracking

#### Deployment Steps

1. **Install Dependencies**

   ```bash
   cd mcp_af3
   pip install -r requirements.txt
   ```

   **Required packages:**
   - `fastmcp` - MCP server framework
   - `pandas` - Excel file handling
   - `openpyxl` - Excel file support
   - AlphaFold3 dependencies (as specified in requirements.txt)

2. **Start AlphaFold3 Service**

   ```bash
   cd mcp_af3
   python af3_mcp_server.py
   ```

   **Expected output:**
   ```
   INFO - AlphaFold3 MCP Server started
   INFO - Listening on http://0.0.0.0:PORT
   ```

#### AlphaFold3 Service Management

**Start service:**
```bash
cd mcp_af3
python af3_mcp_server.py
```

**Stop service:**
- Press `Ctrl+C` in the terminal running the server

**Run in background (Linux/Mac):**
```bash
cd mcp_af3
nohup python af3_mcp_server.py > af3.log 2>&1 &
```

---

### Lineage Analysis Service

The Lineage Analysis service provides BCR repertoire analysis and data integration capabilities for influenza and other viral studies.

#### Features

The Lineage Analysis service provides the following tools:

**Data Extraction:**
- `extract_seurat_umap_metadata` - Extract UMAP coordinates and cellular metadata from Seurat RDS files
  - Extracts UMAP coordinates (dimensionality-reduced cell positions)
  - Retrieves cell type annotation information
  - Extracts expression values for genes of interest
  - Supports single-cell RNA-seq data processing

**Data Integration:**
- `integrate_scbcr_bulk_bcr_data` - Integrate single-cell BCR and bulk BCR sequencing data
  - Loads single-cell RNA-seq BCR data
  - Parses FASTQ files from bulk BCR sequencing
  - Merges single-cell and bulk BCR sequence data
  - Appends UMAP coordinates and cell type annotations
  - Standardizes timepoint and cell type information

- `integrate_binding_neutralization_experiments` - Integrate antibody binding and neutralization experimental measurements
  - Loads two batches of antibody functional assay data
  - Applies thresholds to convert continuous measurements into binary labels
  - Processes binding and neutralization data for multiple influenza strains
  - Standardizes antibody nomenclature and batch information
  - Merges replicate measurements with conflict resolution

- `integrate_predictions_with_experimental_data` - Integrate machine learning prediction results with laboratory measurements
  - Loads ensemble prediction results from multiple folds
  - Processes binding and neutralization predictions
  - Merges prediction and experimental data into BCR feature dataset
  - Adds single-cell/bulk data type labels
  - Retains all BCR sequence features, UMAP coordinates, and cell types

All tools support:
- Seurat RDS file format
- CSV and Excel file formats
- FASTQ file parsing
- Large dataset processing with streaming progress
- Comprehensive data validation and error handling

#### Deployment Steps

1. **Install Dependencies**

   ```bash
   cd mcp_lineage_analysis
   pip install -r requirements.txt
   ```

2. **Start Lineage Analysis Service**

   ```bash
   cd mcp_lineage_analysis
   python flu_bcr_repertoire_analysis_server.py
   ```

   Or use the provided startup script:
   ```bash
   bash start_server.sh
   ```

#### Lineage Analysis Service Management

**Start service:**
```bash
cd mcp_lineage_analysis
python flu_bcr_repertoire_analysis_server.py
```

**Stop service:**
- Press `Ctrl+C` in the terminal running the server

**Run in background (Linux/Mac):**
```bash
cd mcp_lineage_analysis
nohup python flu_bcr_repertoire_analysis_server.py > lineage_analysis.log 2>&1 &
```

---

### Bioinformatics Service

The Bioinformatics service provides comprehensive single-cell B cell analysis and visualization capabilities.

#### Features

The Bioinformatics service provides the following tools:

**Visualization Analysis:**
- `antigen_binding_prediction_visualization` - Single-cell B cell antigen binding prediction visualization
  - Automatically detects multiple binding prediction column formats
  - Numerical conversion and NA value handling
  - Broad reactivity threshold classification and statistical analysis
  - Binding prediction value distribution visualization
  - Cell type-specific binding pattern analysis
  - Exports statistical results to CSV files

- `bcell_celltype_distribution_analysis` - Single-cell B cell subtype distribution visualization
  - King dataset cell type mapping and standardized annotation
  - B cell subtype classification statistics (Naive, Memory, GC, Plasma, etc.)
  - Cell type proportion distribution calculation and visualization
  - Multi-color palette cell type coloring scheme
  - Cell type distribution pie charts and bar charts
  - Exports cell type statistical data to CSV files

- `binding_prediction_interval_distribution_analysis` - Antigen binding prediction value interval distribution analysis
  - Customizable interval step and data range
  - Generates interval distribution histograms
  - Calculates number of cells and percentage in each interval
  - Cumulative distribution function (CDF) calculation and visualization
  - Quantile analysis and outlier detection
  - Exports interval statistics to CSV files

- `umap_dimensionality_reduction_visualization` - Single-cell B cell UMAP reduction and cell type visualization
  - UMAP coordinate extraction and two-dimensional space mapping
  - B cell type in UMAP space distribution visualization
  - Cell type specific color encoding and figure legend
  - High quality UMAP plots suitable for publication
  - Cell density distribution and cluster boundary visualization
  - Supports King dataset's cell type mapping
  - Exports UMAP coordinates and cell type information to CSV

- `bcell_marker_gene_dotplot_analysis` - B cell type specific gene expression dotplot analysis
  - B cell type specific gene expression set definition and detection
  - Gene expression level and expression ratio's double visualization
  - Dotplot size represents expression ratio, color represents expression strength
  - Expression threshold filtering for biological significance
  - Multiple B cell type specific gene expression comparison
  - Auto-detects available gene markers
  - Exports gene expression statistics to CSV files

- `antigen_binding_neutralization_density_visualization` - Antigen binding and neutralization prediction density plot visualization
  - Automatically detects multiple prediction field formats
  - Flexible NA value handling strategies
  - Feature selection priority configuration
  - Nebulosa density plot generation in UMAP space
  - Gradient color mapping visualization
  - Supports King dataset cell type mapping
  - Exports prediction value statistics and UMAP coordinate data

All tools support:
- Seurat RDS file format
- Single-cell RNA-seq data processing
- UMAP coordinate extraction and visualization
- Cell type annotation and mapping
- Publication-quality figure generation
- Comprehensive statistical analysis

#### Deployment Steps

1. **Install Dependencies**

   ```bash
   cd mcp_r
   # Install Python dependencies
   pip install fastmcp pandas rpy2
   
   # Install R dependencies (if needed)
   # R script dependencies are managed via renv
   ```

2. **Start Bioinformatics Service**

   ```bash
   cd mcp_r
   python bioinformatics_mcp_server.py
   ```

#### Bioinformatics Service Management

**Start service:**
```bash
cd mcp_r
python bioinformatics_mcp_server.py
```

**Stop service:**
- Press `Ctrl+C` in the terminal running the server

**Run in background (Linux/Mac):**
```bash
cd mcp_r
nohup python bioinformatics_mcp_server.py > bioinformatics.log 2>&1 &
```

---

### Data Integration Service (integrateBcrData)

The Data Integration service provides comprehensive BCR data integration pipeline with UMAP, clustering, and cell type annotation.

#### Features

The Data Integration service provides the following tools:

**Complete BCR Data Integration:**
- `integrate_bcr_data_complete` - Complete BCR data integration pipeline
  - Automatic detection and conversion of Excel files
  - Intelligent field version control (protects Heavy/Light chains, versions prediction fields)
  - UMAP dimensionality reduction and visualization
  - FindClusters for cell clustering analysis
  - Marker gene-based cell type annotation with confidence scoring
  - Excel (.xlsx, .xls) to CSV conversion
  - Field standardization and barcode matching
  - Smart version control for repeated integrations
  - Complete B-cell subset annotation (Naive, Memory, Plasma, GC, etc.)

**Use Cases:**
- One-click BCR prediction data integration
- Complete single-cell B-cell analysis pipeline
- Version-controlled data updates
- Reproducible B-cell immunology analysis

All tools support:
- Excel and CSV file formats
- Seurat RDS file format
- UMAP dimensionality reduction
- Cell clustering and annotation
- Version control for data updates
- Large dataset processing with streaming progress

#### Deployment Steps

1. **Install Dependencies**

   ```bash
   cd mcp_r
   # Install Python dependencies
   pip install fastmcp pandas rpy2
   
   # Install R dependencies (if needed)
   # R script dependencies are managed via renv
   ```

2. **Start Data Integration Service**

   ```bash
   cd mcp_r
   python data_mcp_server.py
   ```

#### Data Integration Service Management

**Start service:**
```bash
cd mcp_r
python data_mcp_server.py
```

**Stop service:**
- Press `Ctrl+C` in the terminal running the server

**Run in background (Linux/Mac):**
```bash
cd mcp_r
nohup python data_mcp_server.py > data_integration.log 2>&1 &
```

---

## Configuration and Execution

### 1. Configure File Save Directory

Edit `agent/usecases/immunity/common/utils.py` and modify line 27 to set your local output directory:

```python
OUTPUT_DIR = "/path/to/your/output/directory"
```

### 2. Configure Model and API Keys

1. Navigate to `agent/usecases/immunity/config`
2. Copy configuration from `immunity_config_qwen` or `immunity_config_gpt4.1` to `immunity_config.py`
3. Update API keys in the configuration file

### 3. Configure MCP Services

Edit `immunity_config.py` and configure the MCP service IDs:

```python
"mcp_config": {
    "service_ids": [
        "igblast",           # V(D)J analysis
        "metabcr",           # BCR prediction
        "lineage_analysis",  # Clonal lineage analysis
        "af3",               # AlphaFold3 structure prediction
        "integrateBcrData",  # BCR data integration
        "bioinformatics",    # Bioinformatics tools
        "file_utils",        # File download and format conversion tools
    ]
}
```

**Required Services:**
- `igblast` - V(D)J gene annotation
- `metabcr` - MetaBCR binding prediction
- `lineage_analysis` - BCR lineage tracing
- `af3` - Protein structure prediction
- `integrateBcrData` - Data integration
- `bioinformatics` - General bioinformatics tools
- `file_utils` - File download and format conversion (CSV, Excel, FASTA)

### 4. Run the Workflow

```bash
cd agent/usecases/immunity
python start_improved_workflow.py \
    --query "your research question" \
    --file_url "path/to/your/data/file.csv"
```

**Example:**
```bash
python start_improved_workflow.py \
    --query "please design a computational method to identify broadly neutralizing antibodies against H5N1." \
    --file_url "https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/flu-simple.xlsx"
```

### 5. Test Data

Sample data files are available for testing:

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

### 6. Important Notes

- **Runtime**: The complete workflow may take several days to finish
- **metabcr tool**: Antigen files must include at least a `variant_seq` column
- **integrate_bcr_data_complet tool**: Specify mapping parameters:
  ```json
  {
    "args": {
      "csv_fields": "main_name",
      "rds_fields": "main_name"
    }
  }
  ```
- **Interactive confirmation**: Tool parameters can be modified before execution - the system will request confirmation before each tool call

---

## Service Management

### Start All Services

```bash
# Terminal 1: Start Qdrant
cd kb
docker compose up -d

# Terminal 2: Start Ollama
ollama serve

# Terminal 3: Start IgBLAST service
cd mcp_Igblast
conda activate antibody_venv
python igblast_mcp_server.py

# Terminal 4: Start File Utils service
cd mcp_file_utils
python file_utils_mcp_server.py

# Terminal 5: Start AlphaFold3 service
cd mcp_af3
python af3_mcp_server.py

# Terminal 6: Start Lineage Analysis service
cd mcp_lineage_analysis
python flu_bcr_repertoire_analysis_server.py

# Terminal 7: Start Bioinformatics service
cd mcp_r
python bioinformatics_mcp_server.py

# Terminal 8: Start Data Integration service
cd mcp_r
python data_mcp_server.py

# Terminal 9: Start other MCP services as needed
# (metabcr, etc.)
```

### Stop All Services

```bash
# Stop Qdrant
cd kb
docker compose down

# Stop Ollama
pkill ollama

# Stop IgBLAST service
# Press Ctrl+C in the terminal running the service

# Stop File Utils service
# Press Ctrl+C in the terminal running the service

# Stop AlphaFold3 service
# Press Ctrl+C in the terminal running the service

# Stop Lineage Analysis service
# Press Ctrl+C in the terminal running the service

# Stop Bioinformatics service
# Press Ctrl+C in the terminal running the service

# Stop Data Integration service
# Press Ctrl+C in the terminal running the service

# Stop other MCP services
# Press Ctrl+C in respective terminals
```

### Service Status Check

```bash
# Check Qdrant
curl http://localhost:6333/healthz
docker compose ps

# Check Ollama
curl http://localhost:11434/api/tags

# Check IgBLAST (should return server info)
curl http://localhost:8110/health  # If health endpoint exists
```

### View Logs

```bash
# Qdrant logs
cd kb
docker compose logs -f qdrant
docker compose logs --tail=100 qdrant

# IgBLAST logs
# Check the terminal output or log file if running in background
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. uv command not found

**Symptoms:** `uv: command not found` error

**Solutions:**
- **Windows**: Restart PowerShell or reopen terminal after installation
- **Linux/Mac**: Add to PATH:
  ```bash
  export PATH="$HOME/.local/bin:$PATH"
  source ~/.bashrc  # or ~/.zshrc
  ```
- Verify installation: `uv --version`

#### 2. Qdrant Connection Failed

**Symptoms:** Cannot connect to Qdrant service

**Solutions:**
- Check if Qdrant is running: `docker compose ps` (in `kb` directory)
- Verify environment variable: `echo $QDRANT_HOST` (should be `localhost`)
- Check service health: `curl http://localhost:6333/healthz`
- Restart Qdrant: `docker compose restart` (in `kb` directory)
- Check firewall settings if accessing remotely

#### 3. IgBLAST Service Cannot Start

**Symptoms:** Service fails to start or crashes immediately

**Solutions:**
- Ensure conda environment is activated: `conda activate antibody_venv`
- Verify configuration paths are correct in `config/config.py`
- Check if tools are installed: `igblastn -version` and `MakeDb.py -h`
- Verify database files exist: `ls igblast_changeO/igblast/database/`
- Check for port conflicts: `netstat -an | grep 8110` (Linux) or `lsof -i :8110` (Mac)

#### 4. Model API Key Error

**Symptoms:** API authentication failures

**Solutions:**
- Verify API key in `agent/usecases/immunity/config/immunity_config.py`
- Check API key validity and quota on provider's website
- Ensure API key format is correct (no extra spaces or quotes)
- Test API key with a simple request

#### 5. Ollama Model Not Found

**Symptoms:** `model not found` errors when using KB

**Solutions:**
- Pull required model: `ollama pull nomic-embed-text`
- Verify model is available: `ollama list`
- Check Ollama service is running: `curl http://localhost:11434/api/tags`

#### 6. MCP Service Connection Issues

**Symptoms:** Cannot connect to MCP services

**Solutions:**
- Verify service is running and listening on correct port
- Check service IDs in `immunity_config.py` match running services
- Ensure network connectivity between services
- Check service logs for error messages

#### 7. Permission Denied Errors

**Symptoms:** File permission or Docker permission errors

**Solutions:**
- **Docker**: Add user to docker group: `sudo usermod -aG docker $USER` then `newgrp docker`
- **File permissions**: Check directory permissions for output and data directories
- **Conda**: Ensure conda environment has proper permissions

---

## Additional Resources

### Documentation

- **KB Knowledge Base**: Detailed documentation in `kb/README.md`
- **IgBLAST Service**: Deployment guide in `mcp_Igblast/docs/README_CONFIG.md`
- **Project Architecture**: See `doc/` directory for architecture diagrams

### Support

For additional help:
1. Check the troubleshooting section above
2. Review service-specific documentation in respective directories
3. Check service logs for detailed error messages
4. Verify all prerequisites are met using the deployment checklist

### Related Projects

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Ollama Documentation](https://ollama.ai/docs)
- [IgBLAST Documentation](https://ncbi.github.io/igblast/)
- [ChangeO Documentation](https://changeo.readthedocs.io/)

---

**Last Updated**: 2024
