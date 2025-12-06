# AIRR Data Commons MCP Server Architecture

## Executive Summary

The AIRR Data Commons MCP server provides programmatic access to millions of B-cell receptor (BCR) sequences from multiple repositories through a unified interface. This server leverages the AIRR Data Commons API v1 specification to search, filter, and download repertoire data in the standardized AIRR format, which is **identical to IgBLAST output format**, enabling seamless integration with existing analysis pipelines.

**Key Advantages:**
- Direct AIRR format compatibility with IgBLAST MCP output
- Access to 100+ studies with millions of sequences
- Both bulk BCR-seq and single-cell BCR-seq data
- Standardized metadata across all repositories
- No format conversion needed - direct pipeline integration

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Client (LangChain)                  │
│                                                              │
│  Tools: search_airr_repertoires, download_airr_sequences... │
└──────────────────────┬──────────────────────────────────────┘
                       │ stdio transport
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  AIRR MCP Server (FastMCP)                  │
│                      Port: 8100                             │
├─────────────────────────────────────────────────────────────┤
│  Components:                                                 │
│  ├─ Repository Manager (multi-repository support)           │
│  ├─ Query Builder (JSON filter construction)                │
│  ├─ Pagination Handler (streaming large datasets)           │
│  ├─ Cache Manager (metadata caching)                        │
│  └─ Format Handler (AIRR TSV/JSON processing)               │
└──────────────────────┬──────────────────────────────────────┘
                       │ REST API calls
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              AIRR Data Commons Repositories                 │
│                                                              │
│  ├─ iReceptor Public Archive (https://ipa1.ireceptor.org)  │
│  ├─ iReceptor COVID-19 (https://covid19-1.ireceptor.org)   │
│  ├─ VDJServer (https://vdjserver.org)                       │
│  └─ Other registered repositories...                        │
└─────────────────────────────────────────────────────────────┘
```

## REST API Integration

### Repository Endpoints

The server supports multiple AIRR Data Commons repositories:

| Repository | Base URL | Coverage |
|------------|----------|----------|
| **VDJServer** | `https://vdjserver.org/airr/v1/` | 39 studies, 3,408 repertoires, ~2.5B sequences |
| **iReceptor Public Archive** | `https://ipa1.ireceptor.org/airr/v1/` | General immunology studies |
| **iReceptor COVID-19** | `https://covid19-1.ireceptor.org/airr/v1/` | COVID-19 specific studies |

### API Endpoints

All repositories implement the standard AIRR Data Commons API v1:

- `/repertoire` - Query and retrieve repertoire metadata
- `/rearrangement` - Query and download sequence data
- `/clone` - Query clonal families
- `/cell` - Query single-cell data
- `/expression` - Query gene expression data
- `/receptor` - Query paired receptor data

### Authentication

**Current Status**: No authentication required (as of AIRR API v1)
**Future**: Will implement authentication when added to AIRR API v2+

## MCP Tool Specifications

### Tool 1: search_airr_repertoires

**Purpose**: Search across AIRR repositories for relevant repertoires

**Parameters**:
```json
{
  "disease": {
    "type": "string",
    "description": "Disease or condition (e.g., 'COVID-19', 'influenza', 'cancer')",
    "required": false
  },
  "tissue": {
    "type": "string",
    "description": "Tissue type (e.g., 'peripheral blood', 'lymph node')",
    "required": false
  },
  "species": {
    "type": "string",
    "enum": ["human", "mouse", "rat", "rabbit"],
    "default": "human",
    "description": "Species",
    "required": false
  },
  "cell_subset": {
    "type": "string",
    "description": "B cell subset (e.g., 'naive', 'memory', 'plasma')",
    "required": false
  },
  "repository": {
    "type": "string",
    "enum": ["vdjserver", "ireceptor", "covid19", "all"],
    "default": "all",
    "description": "Which repository to search",
    "required": false
  },
  "max_results": {
    "type": "integer",
    "default": 100,
    "description": "Maximum number of repertoires to return",
    "required": false
  }
}
```

**Returns**:
```json
{
  "status": "success",
  "repertoires": [
    {
      "repertoire_id": "6173719481891549676-242ac11c-0001-012",
      "study_id": "PRJNA300878",
      "study_title": "B cell repertoire analysis of influenza vaccination",
      "subject_id": "TW01",
      "sample_id": "TW01_T0",
      "disease_state": "influenza vaccination",
      "tissue": "peripheral blood mononuclear cells",
      "cell_subset": "memory B cells",
      "sequence_count": 150000,
      "repository": "vdjserver",
      "download_url": "https://vdjserver.org/airr/v1/rearrangement"
    }
  ],
  "total_repertoires": 42,
  "repositories_searched": ["vdjserver", "ireceptor"]
}
```

### Tool 2: download_airr_sequences

**Purpose**: Download sequences from specific repertoires in AIRR format

**Parameters**:
```json
{
  "repertoire_id": {
    "type": "string",
    "description": "Repertoire identifier from search results",
    "required": true
  },
  "filters": {
    "type": "object",
    "description": "Additional filters for sequence selection",
    "properties": {
      "v_call": {
        "type": "string",
        "description": "V gene filter (e.g., 'IGHV3-23')"
      },
      "j_call": {
        "type": "string",
        "description": "J gene filter (e.g., 'IGHJ4')"
      },
      "junction_aa_length": {
        "type": "integer",
        "description": "CDR3 amino acid length"
      },
      "productive": {
        "type": "boolean",
        "description": "Only productive sequences"
      }
    },
    "required": false
  },
  "format": {
    "type": "string",
    "enum": ["airr", "json"],
    "default": "airr",
    "description": "Output format (AIRR TSV or JSON)",
    "required": false
  },
  "max_sequences": {
    "type": "integer",
    "default": 10000,
    "description": "Maximum sequences to download",
    "required": false
  }
}
```

**Returns**:
```json
{
  "status": "success",
  "file_path": "/tmp/airr_sequences_abc123.tsv",
  "format": "airr",
  "sequences_downloaded": 8543,
  "total_available": 150000,
  "fields": [
    "sequence_id", "sequence", "rev_comp", "productive",
    "v_call", "d_call", "j_call", "c_call",
    "junction", "junction_aa", "junction_length",
    "v_identity", "v_alignment_start", "v_alignment_end"
  ],
  "repository": "vdjserver"
}
```

### Tool 3: get_airr_study_metadata

**Purpose**: Get detailed study and sample metadata

**Parameters**:
```json
{
  "study_id": {
    "type": "string",
    "description": "Study identifier (e.g., PRJNA number)",
    "required": true
  },
  "repository": {
    "type": "string",
    "enum": ["vdjserver", "ireceptor", "covid19", "auto"],
    "default": "auto",
    "description": "Repository to query",
    "required": false
  }
}
```

**Returns**:
```json
{
  "status": "success",
  "study": {
    "study_id": "PRJNA300878",
    "study_title": "Convergent antibody signatures in human dengue",
    "study_type": "Observational",
    "study_description": "Analysis of B cell repertoires...",
    "keywords": ["dengue", "antibody", "B cells"],
    "pub_ids": ["PMID:28630358"],
    "subjects": 10,
    "samples": 45,
    "total_sequences": 2500000
  },
  "samples": [
    {
      "sample_id": "DV01_acute",
      "subject_id": "DV01",
      "tissue": "PBMC",
      "disease_state": "dengue acute phase",
      "collection_time_point": "day 5",
      "cell_subset": "plasmablasts"
    }
  ],
  "repository": "vdjserver"
}
```

### Tool 4: filter_by_vdj_genes

**Purpose**: Filter sequences by V/D/J gene usage patterns

**Parameters**:
```json
{
  "repertoire_id": {
    "type": "string",
    "description": "Repertoire identifier",
    "required": true
  },
  "v_gene": {
    "type": "string",
    "description": "V gene family or allele (e.g., 'IGHV3', 'IGHV3-23*01')",
    "required": false
  },
  "d_gene": {
    "type": "string",
    "description": "D gene family or allele",
    "required": false
  },
  "j_gene": {
    "type": "string",
    "description": "J gene family or allele",
    "required": false
  },
  "combination_logic": {
    "type": "string",
    "enum": ["AND", "OR"],
    "default": "AND",
    "description": "How to combine gene filters",
    "required": false
  }
}
```

**Returns**:
```json
{
  "status": "success",
  "filtered_sequences": 1250,
  "total_sequences": 150000,
  "percentage": 0.83,
  "gene_usage_stats": {
    "v_gene_distribution": {
      "IGHV3-23*01": 450,
      "IGHV3-23*02": 300,
      "IGHV3-23*03": 500
    },
    "j_gene_distribution": {
      "IGHJ4*01": 600,
      "IGHJ4*02": 650
    }
  },
  "download_available": true
}
```

### Tool 5: get_airr_statistics

**Purpose**: Get statistical summary of repertoire characteristics

**Parameters**:
```json
{
  "repertoire_id": {
    "type": "string",
    "description": "Repertoire identifier",
    "required": true
  },
  "metrics": {
    "type": "array",
    "items": {
      "type": "string",
      "enum": ["diversity", "clonality", "v_usage", "cdr3_length", "mutation_frequency"]
    },
    "default": ["diversity", "v_usage", "cdr3_length"],
    "description": "Which statistics to calculate",
    "required": false
  }
}
```

**Returns**:
```json
{
  "status": "success",
  "repertoire_id": "6173719481891549676-242ac11c-0001-012",
  "statistics": {
    "total_sequences": 150000,
    "unique_sequences": 125000,
    "productive_sequences": 140000,
    "diversity_metrics": {
      "shannon_entropy": 11.2,
      "simpson_index": 0.98,
      "clonality": 0.15
    },
    "v_gene_usage": {
      "IGHV1": 15.2,
      "IGHV3": 45.8,
      "IGHV4": 25.3
    },
    "cdr3_length_distribution": {
      "mean": 45,
      "median": 42,
      "mode": 39,
      "range": [21, 81]
    },
    "mutation_frequency": {
      "mean": 0.068,
      "median": 0.055
    }
  }
}
```

## AIRR Format Field Mapping

The AIRR format returned is **identical to IgBLAST output**, containing these standard fields:

| Field | Description | Example |
|-------|-------------|---------|
| `sequence_id` | Unique sequence identifier | "seq_001" |
| `sequence` | Nucleotide sequence | "CAGGTGCAGCTGGTG..." |
| `v_call` | V gene assignment | "IGHV3-23*01" |
| `d_call` | D gene assignment | "IGHD3-10*01" |
| `j_call` | J gene assignment | "IGHJ4*02" |
| `junction` | CDR3 nucleotide sequence | "TGTGCGAGA..." |
| `junction_aa` | CDR3 amino acid sequence | "CARGLVVV..." |
| `productive` | Productive rearrangement | true/false |
| `v_identity` | V gene identity percentage | 0.97 |
| `v_alignment_start` | V alignment start position | 1 |
| `v_alignment_end` | V alignment end position | 296 |

**Critical Advantage**: Since this format matches IgBLAST output exactly, sequences from AIRR Data Commons can be directly merged with IgBLAST analysis results without any conversion!

## Pagination and Streaming Strategy

### Pagination Implementation

```python
def paginate_results(endpoint, filters, page_size=1000):
    """
    Paginate through large result sets
    """
    all_results = []
    from_index = 0

    while True:
        request = {
            "filters": filters,
            "from": from_index,
            "size": page_size,
            "format": "json"
        }

        response = requests.post(endpoint, json=request)
        data = response.json()

        results = data.get("Rearrangement", [])
        all_results.extend(results)

        # Check if more pages exist
        if len(results) < page_size:
            break

        from_index += page_size

    return all_results
```

### Streaming Large Datasets

For very large datasets (>100,000 sequences):

1. **Chunked Download**: Download in batches of 10,000 sequences
2. **Temporary File Storage**: Write to temp files during download
3. **Progressive Processing**: Allow processing while downloading
4. **Memory Management**: Clear processed chunks from memory

```python
def stream_large_dataset(repertoire_id, output_file, chunk_size=10000):
    """
    Stream large datasets to file without loading all in memory
    """
    with open(output_file, 'w') as f:
        # Write AIRR TSV header
        f.write('\t'.join(AIRR_FIELDS) + '\n')

        from_index = 0
        while True:
            chunk = download_chunk(repertoire_id, from_index, chunk_size)

            if not chunk:
                break

            # Write chunk to file
            for record in chunk:
                f.write(format_airr_record(record) + '\n')

            from_index += chunk_size

            # Optional: yield progress updates
            yield {
                "downloaded": from_index,
                "status": "downloading"
            }
```

## File Structure

```
mcp_AIRR/
├── README.md                      # Setup and usage documentation
├── requirements.txt               # Python dependencies
├── pyproject.toml                # Project configuration
│
├── airr_mcp_server.py            # Main MCP server implementation
├── config.yaml                   # Repository configurations
│
├── src/
│   ├── __init__.py
│   ├── repositories.py          # Repository manager
│   ├── query_builder.py         # AIRR query construction
│   ├── pagination.py            # Pagination handling
│   ├── cache.py                 # Metadata caching
│   └── format_handler.py        # AIRR format processing
│
├── tests/
│   ├── test_repositories.py     # Repository connection tests
│   ├── test_query_builder.py    # Query construction tests
│   ├── test_pagination.py       # Pagination tests
│   ├── test_integration.py      # End-to-end tests
│   └── fixtures/                # Test data
│       └── sample_airr.tsv
│
└── cache/                        # Cached metadata (gitignored)
    ├── studies/
    └── repertoires/
```

## Implementation Details

### Repository Manager

```python
class RepositoryManager:
    """Manages connections to multiple AIRR repositories"""

    REPOSITORIES = {
        "vdjserver": {
            "base_url": "https://vdjserver.org/airr/v1",
            "name": "VDJServer",
            "timeout": 30
        },
        "ireceptor": {
            "base_url": "https://ipa1.ireceptor.org/airr/v1",
            "name": "iReceptor Public Archive",
            "timeout": 30
        },
        "covid19": {
            "base_url": "https://covid19-1.ireceptor.org/airr/v1",
            "name": "iReceptor COVID-19",
            "timeout": 30
        }
    }

    def query_all(self, endpoint, filters):
        """Query all repositories in parallel"""
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.query_single, repo, endpoint, filters): repo
                for repo in self.REPOSITORIES
            }

            for future in as_completed(futures):
                repo = futures[future]
                try:
                    results[repo] = future.result()
                except Exception as e:
                    results[repo] = {"error": str(e)}

        return results
```

### Query Builder

```python
class QueryBuilder:
    """Constructs AIRR-compliant JSON queries"""

    def build_repertoire_query(self, disease=None, tissue=None, species="human"):
        """Build repertoire search query"""
        filters = []

        if species:
            filters.append({
                "op": "=",
                "content": {
                    "field": "subject.species.id",
                    "value": self.species_to_ncbi(species)
                }
            })

        if disease:
            filters.append({
                "op": "contains",
                "content": {
                    "field": "subject.diagnosis.disease_diagnosis",
                    "value": disease
                }
            })

        if tissue:
            filters.append({
                "op": "contains",
                "content": {
                    "field": "sample.tissue",
                    "value": tissue
                }
            })

        if len(filters) == 0:
            return {}
        elif len(filters) == 1:
            return {"filters": filters[0]}
        else:
            return {
                "filters": {
                    "op": "and",
                    "content": filters
                }
            }

    def species_to_ncbi(self, species):
        """Convert species name to NCBI taxonomy ID"""
        mapping = {
            "human": "NCBITAXON:9606",
            "mouse": "NCBITAXON:10090",
            "rat": "NCBITAXON:10116",
            "rabbit": "NCBITAXON:9986"
        }
        return mapping.get(species, species)
```

### Cache Manager

```python
class CacheManager:
    """Caches frequently accessed metadata"""

    def __init__(self, cache_dir="cache", ttl=3600):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl  # Time to live in seconds

    def get_cached(self, key):
        """Get cached data if not expired"""
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            stat = cache_file.stat()
            age = time.time() - stat.st_mtime

            if age < self.ttl:
                with open(cache_file) as f:
                    return json.load(f)

        return None

    def set_cache(self, key, data):
        """Store data in cache"""
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
```

## Error Handling and Retry Logic

### Error Types

1. **Network Errors**: Connection timeouts, DNS failures
2. **API Errors**: 400 Bad Request, 404 Not Found, 500 Server Error
3. **Rate Limiting**: 429 Too Many Requests
4. **Data Errors**: Invalid JSON, missing required fields

### Retry Strategy

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def api_request_with_retry(url, payload):
    """Make API request with exponential backoff retry"""
    response = requests.post(url, json=payload, timeout=30)

    if response.status_code == 429:
        # Rate limited - wait longer
        wait_time = int(response.headers.get('Retry-After', 60))
        time.sleep(wait_time)
        raise RateLimitError(f"Rate limited, retry after {wait_time}s")

    response.raise_for_status()
    return response.json()
```

### Error Response Format

```json
{
  "status": "error",
  "error_type": "api_error",
  "error_message": "Repository returned 404: Repertoire not found",
  "repository": "vdjserver",
  "suggestion": "Verify repertoire_id or try different repository"
}
```

## Performance Optimization

### Caching Strategy

1. **Metadata Caching**: Cache study and repertoire metadata (1 hour TTL)
2. **Query Result Caching**: Cache search results (15 minutes TTL)
3. **Statistics Caching**: Cache computed statistics (1 hour TTL)

### Parallel Processing

1. **Multi-Repository Queries**: Query all repositories in parallel
2. **Batch Downloads**: Download sequences in parallel chunks
3. **Async Processing**: Use asyncio for I/O-bound operations

### Memory Management

1. **Streaming Downloads**: Never load entire dataset in memory
2. **Chunk Processing**: Process data in manageable chunks
3. **Temporary Files**: Use temp files for intermediate results

## Integration with IgBLAST Pipeline

Since AIRR format is identical to IgBLAST output:

```python
# 1. Download sequences from AIRR Data Commons
airr_sequences = download_airr_sequences(
    repertoire_id="6173719481891549676-242ac11c-0001-012",
    max_sequences=10000
)

# 2. Directly merge with IgBLAST results (no conversion!)
igblast_results = analyze_vdj_batch(new_sequences)

# 3. Combine AIRR and IgBLAST data
combined_df = pd.concat([
    pd.read_csv(airr_sequences["file_path"], sep='\t'),
    pd.DataFrame(igblast_results["results"])
])

# 4. Unified analysis on combined dataset
analyze_combined_repertoire(combined_df)
```

## Security Considerations

1. **Input Validation**: Validate all user inputs before API calls
2. **SQL Injection**: Use parameterized queries for filters
3. **File Path Security**: Sanitize file paths, use temp directory
4. **API Key Management**: Store keys in environment variables
5. **Rate Limiting**: Implement client-side rate limiting

## Testing Strategy

### Unit Tests
- Test each tool independently
- Mock API responses
- Test error handling

### Integration Tests
- Test real API connections
- Test pagination with large datasets
- Test multi-repository queries

### Performance Tests
- Benchmark download speeds
- Test memory usage with large datasets
- Test concurrent requests

## Deployment Configuration

### Environment Variables

```bash
# Repository preferences
export AIRR_DEFAULT_REPOSITORY="vdjserver"
export AIRR_REPOSITORIES="vdjserver,ireceptor,covid19"

# Performance settings
export AIRR_MAX_CONCURRENT_REQUESTS=3
export AIRR_CHUNK_SIZE=10000
export AIRR_CACHE_TTL=3600

# Timeouts
export AIRR_REQUEST_TIMEOUT=30
export AIRR_DOWNLOAD_TIMEOUT=600

# Cache directory
export AIRR_CACHE_DIR="/tmp/airr_cache"
```

### Docker Support

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8100

CMD ["python", "airr_mcp_server.py"]
```

## Future Enhancements

### Phase 1 (Immediate)
- [x] Basic repository queries
- [x] Sequence download
- [x] Metadata retrieval
- [ ] Implementation and testing

### Phase 2 (Short-term)
- [ ] Authentication support (when added to AIRR API)
- [ ] Advanced filtering (complex queries)
- [ ] Result caching optimization
- [ ] Progress callbacks for large downloads

### Phase 3 (Long-term)
- [ ] GraphQL support (if added to AIRR spec)
- [ ] Real-time streaming updates
- [ ] Federated search optimization
- [ ] Machine learning integration for repertoire analysis

## References

- [AIRR Data Commons API Documentation](https://docs.airr-community.org/en/stable/api/adc_api.html)
- [AIRR Standards](https://docs.airr-community.org/)
- [VDJServer](https://vdjserver.org/)
- [iReceptor Gateway](https://gateway.ireceptor.org/)
- [AIRR Community GitHub](https://github.com/airr-community/)

## Appendix: Example Usage in ImmuneAgent Workflow

```python
# Example: Find COVID-19 neutralizing antibodies

# 1. Search for COVID-19 repertoires
repertoires = search_airr_repertoires(
    disease="COVID-19",
    tissue="peripheral blood",
    cell_subset="memory B cells"
)

# 2. Download sequences with specific V gene
sequences = download_airr_sequences(
    repertoire_id=repertoires["repertoires"][0]["repertoire_id"],
    filters={"v_call": "IGHV3-53", "productive": True},
    max_sequences=5000
)

# 3. Get study metadata
metadata = get_airr_study_metadata(
    study_id=repertoires["repertoires"][0]["study_id"]
)

# 4. Analyze V/D/J gene usage
gene_stats = filter_by_vdj_genes(
    repertoire_id=repertoires["repertoires"][0]["repertoire_id"],
    v_gene="IGHV3-53",
    j_gene="IGHJ6"
)

# 5. Get repertoire statistics
stats = get_airr_statistics(
    repertoire_id=repertoires["repertoires"][0]["repertoire_id"],
    metrics=["diversity", "clonality", "mutation_frequency"]
)

# Result: Comprehensive COVID-19 antibody dataset ready for analysis
```

---

**Document Version**: 1.0
**Created**: 2025-10-07
**Status**: Complete Architecture Design
**Next Steps**: Implementation of airr_mcp_server.py