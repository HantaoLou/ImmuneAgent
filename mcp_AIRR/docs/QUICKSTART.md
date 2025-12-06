# AIRR Data Commons MCP Server - Quick Start

Get up and running with the AIRR MCP server in 5 minutes.

## Prerequisites

- Python 3.12+
- pip or uv package manager

## Installation

```bash
cd /Users/ahleyliu/LocalDoc/ImmuneAgent/mcp_AIRR

# Install dependencies
pip install -r requirements.txt

# Or with uv (faster)
uv pip install -r requirements.txt
```

## Basic Usage

### 1. Test the Server

```bash
# Run tests to verify installation
pytest test_airr_basic.py -v

# Expected output: All tests pass
```

### 2. Run the MCP Server

```bash
# Start the server (stdio transport)
python airr_mcp_server.py
```

### 3. Use with ImmuneAgent

Add to your MCP client configuration:

```python
from langchain_mcp_adapters import create_client_from_stdio

# Create AIRR MCP client
airr_client = create_client_from_stdio(
    ["python", "/Users/ahleyliu/LocalDoc/ImmuneAgent/mcp_AIRR/airr_mcp_server.py"]
)

# Get available tools
tools = airr_client.get_tools()
print(f"Available AIRR tools: {[t.name for t in tools]}")
```

## Quick Examples

### Example 1: Find COVID-19 Antibodies

```python
# 1. Search for COVID-19 repertoires
result = search_airr_repertoires(
    disease="COVID-19",
    tissue="peripheral blood",
    species="human",
    max_results=10
)

print(f"Found {result['total_repertoires']} repertoires")

# 2. Download sequences from first repertoire
if result['repertoires']:
    rep_id = result['repertoires'][0]['repertoire_id']

    sequences = download_airr_sequences(
        repertoire_id=rep_id,
        filters={"productive": True},
        max_sequences=1000
    )

    print(f"Downloaded {sequences['sequences_downloaded']} sequences")
    print(f"File: {sequences['file_path']}")
```

### Example 2: Analyze V Gene Usage

```python
# Get gene usage statistics
gene_stats = filter_by_vdj_genes(
    repertoire_id="your_repertoire_id",
    v_gene="IGHV3",
    combination_logic="AND"
)

print(f"Found {gene_stats['filtered_sequences']} sequences")
print("Top V genes:")
for gene, count in gene_stats['gene_usage_stats']['v_gene_distribution'].items():
    print(f"  {gene}: {count}")
```

### Example 3: Get Study Metadata

```python
# Get detailed study information
metadata = get_airr_study_metadata(
    study_id="PRJNA300878"
)

print(f"Study: {metadata['study']['study_title']}")
print(f"Subjects: {metadata['study']['subjects']}")
print(f"Samples: {metadata['study']['samples']}")
```

## AIRR Format Output

The downloaded sequences are in AIRR TSV format (identical to IgBLAST):

```tsv
sequence_id	sequence	v_call	d_call	j_call	junction	junction_aa	productive
seq_001	CAGGTGCAG...	IGHV3-23*01	IGHD3-10*01	IGHJ4*02	TGTGCG...	CARGLVVV...	T
seq_002	GAGGTGCAG...	IGHV1-69*01	IGHD3-3*01	IGHJ6*02	TGTGCA...	CAKDGTYY...	T
```

## Direct IgBLAST Integration

Since formats are identical, merge with IgBLAST results:

```python
import pandas as pd

# Load AIRR sequences
airr_df = pd.read_csv(airr_result["file_path"], sep='\t')

# Load IgBLAST results
igblast_df = pd.read_csv(igblast_result["file_path"], sep='\t')

# Merge seamlessly - no conversion needed!
combined = pd.concat([airr_df, igblast_df], ignore_index=True)

print(f"Total sequences: {len(combined)}")
```

## Cache Management

The server caches metadata for performance:

```python
from src.cache import CacheManager

cache = CacheManager()

# View cache statistics
stats = cache.get_statistics()
print(f"Cache entries: {stats['total_entries']}")
print(f"Cache size: {stats['total_size_mb']} MB")

# Clear cache if needed
cache.clear_all()
```

## Troubleshooting

### "Module not found" Error

Make sure you're in the correct directory:
```bash
cd /Users/ahleyliu/LocalDoc/ImmuneAgent/mcp_AIRR
python airr_mcp_server.py
```

### Repository Connection Issues

Test repository connections:
```python
from src.repositories import RepositoryManager

repo = RepositoryManager()
status = repo.test_all_connections()

for repo_id, result in status['results'].items():
    print(f"{repo_id}: {'✓' if result['available'] else '✗'}")
```

### Slow Downloads

For large datasets, use smaller `max_sequences`:
```python
# Download in batches
for i in range(0, 100000, 10000):
    result = download_airr_sequences(
        repertoire_id=rep_id,
        max_sequences=10000
    )
```

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design
3. Check [test_airr_basic.py](test_airr_basic.py) for more examples
4. Explore the [config.yaml](config.yaml) for customization

## API Endpoints

The server connects to these AIRR repositories:

- **VDJServer**: https://vdjserver.org/airr/v1/
- **iReceptor Public**: https://ipa1.ireceptor.org/airr/v1/
- **iReceptor COVID-19**: https://covid19-1.ireceptor.org/airr/v1/

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review test examples in `test_airr_basic.py`
3. Consult the full README.md

## Summary

You now have access to:
- **Millions** of BCR sequences from public repositories
- **5 powerful tools** for searching and analysis
- **AIRR format** identical to IgBLAST
- **Seamless integration** with existing pipelines

Happy analyzing!
