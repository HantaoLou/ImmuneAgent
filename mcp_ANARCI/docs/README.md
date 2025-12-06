# ANARCI MCP Server

Antibody numbering using ANARCI Python API wrapper.

## Overview

This MCP server provides antibody/TCR sequence numbering using the ANARCI library. It returns exactly what ANARCI provides - **NO hardcoded CDR definitions or antibody-specific logic**.

## Features

- **Batch Numbering**: Process multiple sequences efficiently
- **Multiple Schemes**: Chothia, Kabat, IMGT, Martin, AHo, Wolfguy
- **Germline Assignment**: Optional V/J germline gene identification
- **Raw ANARCI Output**: Returns all ANARCI data without modification

## Installation

### On Linux Server

```bash
# Install ANARCI
pip install anarci

# Install HMMER (required dependency)
conda install -c bioconda hmmer
# OR
sudo apt-get install hmmer

# Install MCP server dependencies
pip install -r requirements.txt
```

### On Mac (Development)

ANARCI path is hardcoded for development:
```python
sys.path.insert(0, '/Users/ahleyliu/LocalDoc/ImmuneAgent/mcp_ANARCI/ANARCI_lineage/ANARCI/lib')
```

## Usage

### Start Server

```bash
python anarci_mcp_server.py
# Server runs on port 8095
```

### MCP Tools

#### 1. number_antibody_batch

Batch numbering with ANARCI API.

```python
{
    "sequences": [
        {"id": "heavy1", "sequence": "EVQLQQSG..."},
        {"id": "light1", "sequence": "DIQMTQSP..."}
    ],
    "scheme": "chothia",  # or kabat, imgt, martin, aho
    "assign_germline": true
}
```

Returns ANARCI's raw output including:
- Numbered sequence (position, residue) tuples
- Chain type (H, K, L, A, B)
- Species identification
- E-value and bit score
- Germline assignments (if requested)

#### 2. number_single_sequence

Quick single sequence numbering.

```python
{
    "sequence": "EVQLQQSGAEVVR...",
    "scheme": "imgt"
}
```

## Architecture

```
User → MCP Tool → ANARCI API → Raw Results → Return to User
```

**No intermediate processing** - Results are returned exactly as ANARCI provides them.

## Key Design Principle

⚠️ **CRITICAL**: This server does NOT hardcode any antibody-specific logic:
- NO CDR position definitions
- NO numbering scheme rules
- NO chain type classifications

All such information comes directly from ANARCI's output.

## Configuration

Server registered in `agent/config/config.py`:

```python
"anarci": {
    "transport": "sse",
    "url": "http://localhost:8095/sse",
    "timeout": 300,
    "sse_read_timeout": 300,
    "session_kwargs": {},
}
```

## Example Output

```json
{
    "status": "success",
    "results": [
        {
            "id": "heavy1",
            "numbered": true,
            "numbering": [
                [
                    [((1, ""), "E"), ((2, ""), "V"), ...],
                    0,
                    120
                ]
            ],
            "alignment_details": [{
                "chain_type": "H",
                "species": "human",
                "evalue": 1.2e-50,
                "bitscore": 180.5,
                "germlines": {
                    "v_gene": "IGHV3-23*01",
                    "j_gene": "IGHJ4*02",
                    "v_identity": 0.97,
                    "j_identity": 0.95
                }
            }],
            "hit_tables": [...],
            "scheme": "chothia"
        }
    ]
}
```

## Notes

- ANARCI requires HMMER3 to be installed
- Only works with antibody/TCR sequences
- Returns None/not_numbered for non-immunoglobulin sequences
- Supports multiple domains per sequence
