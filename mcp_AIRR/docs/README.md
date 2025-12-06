# AIRR Data Commons MCP Server

Programmatic access to millions of B-cell receptor (BCR) sequences from AIRR Data Commons repositories through a unified Model Context Protocol (MCP) interface.

**Key Advantage**: Output format is **IDENTICAL to IgBLAST** - no format conversion needed for pipeline integration!

## Features

- Access to 100+ studies with millions of BCR sequences
- Multi-repository support (VDJServer, iReceptor, COVID-19 Archive)
- AIRR-compliant TSV format (identical to IgBLAST output)
- Intelligent caching for performance
- Pagination for large datasets
- Comprehensive filtering and statistics

## Quick Start

### Installation

```bash
cd mcp_AIRR
pip install -r requirements.txt
```

### Running the Server

```bash
# Start MCP server (stdio transport)
python airr_mcp_server.py
```

### Integration with ImmuneAgent

The server is designed to work with LangChain through FastMCP adapters:

```python
from langchain_mcp_adapters import create_client_from_stdio

# Create MCP client
mcp_client = create_client_from_stdio(
    ["python", "/path/to/airr_mcp_server.py"]
)

# Use as LangChain tools
tools = mcp_client.get_tools()
```

## Available Tools

### 1. search_airr_repertoires

Search for BCR repertoires across repositories.

**Parameters**:
- `disease` (optional): Disease/condition (e.g., "COVID-19", "influenza")
- `tissue` (optional): Tissue type (e.g., "peripheral blood")
- `species` (default: "human"): Species (human, mouse, rat, rabbit)
- `cell_subset` (optional): B cell subset (e.g., "memory", "naive")
- `repository` (default: "all"): Which repository (vdjserver, ireceptor, covid19, all)
- `max_results` (default: 100): Maximum repertoires to return

**Example**:
```python
result = search_airr_repertoires(
    disease="COVID-19",
    tissue="peripheral blood",
    cell_subset="memory B cells",
    species="human"
)

# Returns:
{
    "status": "success",
    "repertoires": [
        {
            "repertoire_id": "6173719481891549676-242ac11c-0001-012",
            "study_id": "PRJNA300878",
            "study_title": "B cell responses to COVID-19",
            "disease_state": "COVID-19 acute phase",
            "tissue": "peripheral blood mononuclear cells",
            "sequence_count": 150000,
            "repository": "vdjserver"
        }
    ],
    "total_repertoires": 42
}
```

### 2. download_airr_sequences

Download BCR sequences in AIRR format (identical to IgBLAST).

**Parameters**:
- `repertoire_id` (required): Repertoire identifier from search
- `filters` (optional): Additional filters
  - `v_call`: V gene (e.g., "IGHV3-23")
  - `j_call`: J gene (e.g., "IGHJ4")
  - `junction_aa_length`: CDR3 length
  - `productive`: Only productive sequences (true/false)
- `format` (default: "airr"): Output format (airr or json)
- `max_sequences` (default: 10000): Maximum sequences

**Example**:
```python
result = download_airr_sequences(
    repertoire_id="6173719481891549676-242ac11c-0001-012",
    filters={
        "v_call": "IGHV3-23",
        "productive": True
    },
    format="airr",
    max_sequences=5000
)

# Returns:
{
    "status": "success",
    "file_path": "/tmp/airr_sequences_abc123.tsv",
    "format": "airr",
    "sequences_downloaded": 4523,
    "compatible_with_igblast": true,
    "repository": "vdjserver"
}
```

**AIRR Format Output** (tab-separated):
```
sequence_id	sequence	v_call	d_call	j_call	junction	junction_aa	productive
seq_001	CAGGTGCAGCTG...	IGHV3-23*01	IGHD3-10*01	IGHJ4*02	TGTGCGAGA...	CARGLVVV...	T
seq_002	GAGGTGCAGCTG...	IGHV1-69*01	IGHD3-3*01	IGHJ6*02	TGTGCAAGA...	CAKDGTYY...	T
```

### 3. get_airr_study_metadata

Get detailed study and sample metadata.

**Parameters**:
- `study_id` (required): Study identifier (e.g., "PRJNA300878")
- `repository` (default: "auto"): Repository to query

**Example**:
```python
result = get_airr_study_metadata(
    study_id="PRJNA300878"
)

# Returns:
{
    "status": "success",
    "study": {
        "study_id": "PRJNA300878",
        "study_title": "Convergent antibody signatures in human dengue",
        "study_type": "Observational",
        "subjects": 10,
        "samples": 45,
        "repertoires": 45
    },
    "samples": [
        {
            "sample_id": "DV01_acute",
            "subject_id": "DV01",
            "tissue": "PBMC",
            "cell_subset": "plasmablasts"
        }
    ]
}
```

### 4. filter_by_vdj_genes

Filter sequences by V/D/J gene usage and get statistics.

**Parameters**:
- `repertoire_id` (required): Repertoire identifier
- `v_gene` (optional): V gene family/allele (e.g., "IGHV3", "IGHV3-23*01")
- `d_gene` (optional): D gene family/allele
- `j_gene` (optional): J gene family/allele
- `combination_logic` (default: "AND"): How to combine filters (AND/OR)

**Example**:
```python
result = filter_by_vdj_genes(
    repertoire_id="6173719481891549676-242ac11c-0001-012",
    v_gene="IGHV3-53",
    j_gene="IGHJ6",
    combination_logic="AND"
)

# Returns:
{
    "status": "success",
    "filtered_sequences": 1250,
    "total_sequences": 150000,
    "percentage": 0.83,
    "gene_usage_stats": {
        "v_gene_distribution": {
            "IGHV3-53*01": 450,
            "IGHV3-53*02": 800
        },
        "j_gene_distribution": {
            "IGHJ6*01": 600,
            "IGHJ6*02": 650
        }
    },
    "download_available": true
}
```

### 5. get_airr_statistics

Get statistical summary of repertoire characteristics.

**Parameters**:
- `repertoire_id` (required): Repertoire identifier
- `metrics` (optional): Statistics to calculate (default: all)

**Example**:
```python
result = get_airr_statistics(
    repertoire_id="6173719481891549676-242ac11c-0001-012"
)

# Returns:
{
    "status": "success",
    "statistics": {
        "total_sequences": 150000,
        "productive_sequences": 140000,
        "v_gene_usage": {
            "IGHV3-23": 12.5,
            "IGHV1-69": 8.3,
            "IGHV4-34": 7.1
        },
        "cdr3_length_distribution": {
            "mean": 45,
            "median": 42,
            "min": 21,
            "max": 81
        }
    }
}
```

## AIRR + IgBLAST Integration

Since AIRR format is identical to IgBLAST output, you can seamlessly combine data:

```python
import pandas as pd

# 1. Download sequences from AIRR Data Commons
airr_result = download_airr_sequences(
    repertoire_id="6173719481891549676-242ac11c-0001-012",
    max_sequences=10000
)

# 2. Run IgBLAST on new sequences
igblast_result = analyze_vdj_batch(new_sequences)

# 3. Load both as DataFrames (identical format!)
airr_df = pd.read_csv(airr_result["file_path"], sep='\t')
igblast_df = pd.DataFrame(igblast_result["results"])

# 4. Merge seamlessly - no format conversion needed!
combined_df = pd.concat([airr_df, igblast_df], ignore_index=True)

# 5. Unified analysis
print(f"Total sequences: {len(combined_df)}")
print(f"V gene usage:\n{combined_df['v_call'].value_counts()}")
```

## Architecture

```
┌─────────────────────────────────────────┐
│     ImmuneAgent (LangChain)             │
│  Uses AIRR tools for BCR analysis      │
└──────────────────┬──────────────────────┘
                   │ stdio transport
                   ▼
┌─────────────────────────────────────────┐
│     AIRR MCP Server (FastMCP)           │
│                                          │
│  5 Tools:                                │
│  • search_airr_repertoires              │
│  • download_airr_sequences              │
│  • get_airr_study_metadata              │
│  • filter_by_vdj_genes                  │
│  • get_airr_statistics                  │
└──────────────────┬──────────────────────┘
                   │ REST API
                   ▼
┌─────────────────────────────────────────┐
│   AIRR Data Commons Repositories        │
│  • VDJServer (2.5B sequences)           │
│  • iReceptor Public Archive             │
│  • iReceptor COVID-19 Archive           │
└─────────────────────────────────────────┘
```

## Repositories

| Repository | Base URL | Coverage |
|------------|----------|----------|
| **VDJServer** | https://vdjserver.org/airr/v1/ | 39 studies, 3,408 repertoires, ~2.5B sequences |
| **iReceptor Public** | https://ipa1.ireceptor.org/airr/v1/ | General immunology studies |
| **iReceptor COVID-19** | https://covid19-1.ireceptor.org/airr/v1/ | COVID-19 specific studies |

## Caching

The server automatically caches:
- Study metadata (1 hour TTL)
- Repertoire metadata (1 hour TTL)
- Query results (15 minutes TTL)

Cache directory: `./cache/`

Manage cache:
```python
from src.cache import CacheManager

cache = CacheManager()
stats = cache.get_statistics()
print(f"Cache entries: {stats['total_entries']}")
print(f"Cache size: {stats['total_size_mb']} MB")

# Clear expired entries
cache.cleanup_expired()

# Clear all cache
cache.clear_all()
```

## Testing

Run comprehensive tests:

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock responses

# Run tests
pytest test_airr_basic.py -v

# Run with coverage
pytest test_airr_basic.py --cov=src --cov-report=html
```

## Performance

### Large Dataset Downloads

For datasets with >100,000 sequences:
- Automatic pagination in chunks of 10,000
- Streaming to disk (no memory overflow)
- Progress tracking available

### Parallel Queries

When searching "all" repositories, queries run in parallel:
- 3 concurrent workers
- Timeout: 30 seconds per repository
- Automatic failover if one repository fails

## Error Handling

All tools return structured error responses:

```json
{
    "status": "error",
    "error": "timeout",
    "message": "Request to vdjserver timed out",
    "suggestion": "Try a different repository or reduce query size"
}
```

## Configuration

Environment variables (optional):

```bash
# Cache settings
export AIRR_CACHE_DIR="/path/to/cache"
export AIRR_CACHE_TTL=3600

# Performance
export AIRR_MAX_CONCURRENT_REQUESTS=3
export AIRR_CHUNK_SIZE=10000

# Timeouts
export AIRR_REQUEST_TIMEOUT=30
export AIRR_DOWNLOAD_TIMEOUT=600
```

## Example Use Cases

### 1. Find COVID-19 Neutralizing Antibodies

```python
# Search for COVID-19 repertoires
repertoires = search_airr_repertoires(
    disease="COVID-19",
    tissue="peripheral blood",
    cell_subset="memory B cells"
)

# Download sequences with VH3-53 (common in neutralizing antibodies)
sequences = download_airr_sequences(
    repertoire_id=repertoires["repertoires"][0]["repertoire_id"],
    filters={"v_call": "IGHV3-53", "productive": True},
    max_sequences=5000
)

# Analyze gene usage
gene_stats = filter_by_vdj_genes(
    repertoire_id=repertoires["repertoires"][0]["repertoire_id"],
    v_gene="IGHV3-53"
)
```

### 2. Compare Repertoires Across Studies

```python
# Get study metadata
study1 = get_airr_study_metadata(study_id="PRJNA300878")
study2 = get_airr_study_metadata(study_id="PRJNA400123")

# Get statistics for comparison
stats1 = get_airr_statistics(repertoire_id=study1_repertoire_id)
stats2 = get_airr_statistics(repertoire_id=study2_repertoire_id)

# Compare V gene usage
compare_v_genes(stats1["statistics"]["v_gene_usage"],
                stats2["statistics"]["v_gene_usage"])
```

### 3. Build Training Dataset for ML

```python
# Search for diverse repertoires
repertoires = search_airr_repertoires(
    species="human",
    max_results=50
)

# Download sequences from each
for rep in repertoires["repertoires"]:
    sequences = download_airr_sequences(
        repertoire_id=rep["repertoire_id"],
        filters={"productive": True},
        max_sequences=10000
    )
    # Add to training dataset
    add_to_dataset(sequences["file_path"])
```

## Troubleshooting

### Connection Issues

If repositories are unavailable:
- Use `repository="vdjserver"` to try specific repository
- Check repository status at https://gateway.ireceptor.org/
- Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`

### Large Downloads Timing Out

For very large datasets:
- Reduce `max_sequences` parameter
- Use pagination manually
- Download in multiple batches by filtering

### Cache Issues

Clear cache if getting stale data:
```bash
rm -rf cache/
```

## Contributing

This MCP server follows the architecture defined in `ARCHITECTURE.md`.

Key components:
- `src/repositories.py` - Multi-repository management
- `src/query_builder.py` - AIRR API query construction
- `src/pagination.py` - Large dataset handling
- `src/cache.py` - Metadata caching
- `src/format_handler.py` - AIRR format processing

## References

- [AIRR Data Commons API](https://docs.airr-community.org/en/stable/api/adc_api.html)
- [AIRR Standards](https://docs.airr-community.org/)
- [VDJServer](https://vdjserver.org/)
- [iReceptor Gateway](https://gateway.ireceptor.org/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)

## License

Part of the ImmuneAgent project.

## Version

1.0.0 - Initial release (2025-10-07)
