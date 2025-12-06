# SAbDab MCP Server

Structural Antibody Database (SAbDab) data download server.

## Overview

This MCP server provides data download functionality from the SAbDab database at Oxford University. Focus is on **downloading datasets and structures**, not search functionality.

## Features

- **CSV Downloads**: Get complete SAbDab summary data
- **PDB Structures**: Download antibody structures with specified numbering
- **Dataset Export**: Download filtered datasets
- **Database Statistics**: Get SAbDab metadata

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Start Server

```bash
python sabdab_mcp_server.py
# Server runs on port 8098
```

### MCP Tools

#### 1. download_sabdab_summary_csv

Download complete SAbDab summary as CSV.

```python
{
    "filters": {
        "resolution": "<2.5",  # Optional
        "antigen": "yes"       # Optional
    }
}
```

Returns full CSV content of SAbDab database.

#### 2. download_pdb_structure

Download individual PDB structures.

```python
{
    "pdb_id": "6m0j",
    "numbering_scheme": "chothia"  # or kabat, imgt
}
```

Returns PDB file content with specified antibody numbering scheme.

#### 3. download_sabdab_dataset

Download specific dataset types.

```python
{
    "dataset_type": "antigen_bound",  # or all, nanobodies
    "output_format": "csv"
}
```

#### 4. get_sabdab_statistics

Get database statistics.

```python
{}
```

Returns total structures count and metadata.

## API Endpoints

Base URL: `http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred`

- Summary CSV: `/sabdab/summary/all/`
- PDB Download: `/sabdab/pdb/{pdb_id}/?scheme={scheme}`
- Filtered Data: `/sabdab/summary/all/?{filters}`

## Configuration

Server registered in `agent/config/config.py`:

```python
"sabdab": {
    "transport": "sse",
    "url": "http://localhost:8098/sse",
    "timeout": 300,
    "sse_read_timeout": 300,
    "session_kwargs": {},
}
```

## Example Output

### CSV Download

```json
{
    "status": "success",
    "csv_content": "pdb,Hchain,Lchain,antigen,...\n6m0j,H,L,SARS-CoV-2 RBD,...",
    "num_entries": 8543,
    "file_size_bytes": 1234567
}
```

### PDB Download

```json
{
    "status": "success",
    "pdb_id": "6m0j",
    "pdb_content": "ATOM   1  N   GLU H   1...",
    "numbering_scheme": "chothia",
    "file_size_bytes": 123456
}
```

## Notes

- Timeout set to 60-120 seconds for large downloads
- CSV files can be several MB in size
- PDB files are returned with antibody-specific numbering
- Network requests may fail - proper error handling included
