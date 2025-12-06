"""
AIRR Data Commons MCP Server

Provides programmatic access to millions of BCR sequences from AIRR repositories.
Output format is IDENTICAL to IgBLAST for seamless pipeline integration.
"""

from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, Optional, List
import logging
import tempfile
import uuid
import sys
from pathlib import Path

# Import support modules
from src.repositories import RepositoryManager
from src.query_builder import QueryBuilder
from src.pagination import PaginationHandler, ChunkedDownloader
from src.cache import CacheManager, QueryCache
from src.format_handler import AIRRFormatHandler, IgBLASTCompatibility

# Configure logging to stderr to avoid interfering with MCP protocol
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("AIRR Data Commons Server")

# Initialize components
repo_manager = RepositoryManager()
query_builder = QueryBuilder()
pagination_handler = PaginationHandler(page_size=1000)
cache_manager = CacheManager(cache_dir="D:/PartTimeJob/hd/antibody_gen/mcp_AIRR/cache", ttl=3600)
query_cache = QueryCache(cache_manager, ttl=900)
format_handler = AIRRFormatHandler()


@mcp.tool()
def search_airr_repertoires(
    disease: Optional[str] = None,
    tissue: Optional[str] = None,
    species: str = "human",
    cell_subset: Optional[str] = None,
    repository: str = "all",
    max_results: int = 100
) -> Dict[str, Any]:
    """
    Search across AIRR repositories for relevant BCR repertoires.

    Args:
        disease: Disease or condition (e.g., 'COVID-19', 'influenza', 'cancer')
        tissue: Tissue type (e.g., 'peripheral blood', 'lymph node')
        species: Species - human, mouse, rat, rabbit (default: human)
        cell_subset: B cell subset (e.g., 'naive', 'memory', 'plasma')
        repository: Which repository to search - vdjserver, ireceptor, covid19, all (default: all)
        max_results: Maximum number of repertoires to return (default: 100)

    Returns:
        {
            "status": "success",
            "repertoires": [
                {
                    "repertoire_id": "...",
                    "study_id": "...",
                    "study_title": "...",
                    "subject_id": "...",
                    "disease_state": "...",
                    "tissue": "...",
                    "sequence_count": 150000,
                    "repository": "vdjserver"
                }
            ],
            "total_repertoires": 42,
            "repositories_searched": ["vdjserver", "ireceptor"]
        }
    """
    try:
        logger.info(f"Searching repertoires: disease={disease}, tissue={tissue}, species={species}")

        # Build query
        query = query_builder.build_repertoire_query(
            disease=disease,
            tissue=tissue,
            species=species,
            cell_subset=cell_subset,
            size=max_results
        )

        # Check cache
        cache_key = f"{disease}_{tissue}_{species}_{cell_subset}_{repository}_{max_results}"
        cached = cache_manager.get("queries", cache_key, "repertoire")
        if cached:
            logger.info("Returning cached results")
            return cached

        # Query repositories
        repositories = [repository] if repository != "all" else None

        if repository == "all":
            results = repo_manager.query_all("repertoire", query, repositories)
        else:
            results = {repository: repo_manager.query_single(repository, "repertoire", query)}

        # Process results
        all_repertoires = []
        repositories_searched = []

        for repo_id, response in results.items():
            if "error" not in response:
                repositories_searched.append(repo_id)

                # 打印响应内容以进行调试
                logger.info(f"Response from {repo_id}: {str(response)[:200]}...")
                
                # Extract repertoire list
                repertoires = response.get("Repertoire", [])
                
                # 检查repertoires是否为空
                if not repertoires:
                    logger.warning(f"No repertoires found in response from {repo_id}")
                    continue
                
                # 打印第一个repertoire的类型和内容
                if repertoires:
                    logger.info(f"First repertoire type: {type(repertoires[0])}")
                    logger.info(f"First repertoire content: {str(repertoires[0])[:200]}...")
                
                for rep_item in repertoires:
                    # 确保rep是字典类型
                    if not isinstance(rep_item, dict):
                        try:
                            # 尝试将字符串转换为字典（如果是JSON字符串）
                            if isinstance(rep_item, str):
                                import json
                                rep = json.loads(rep_item)
                                logger.info(f"Successfully converted string to dict: {str(rep)[:100]}...")
                            else:
                                logger.warning(f"Skipping non-dictionary repertoire of type {type(rep_item)}")
                                continue
                        except Exception as e:
                            logger.error(f"Error converting repertoire: {e}")
                            continue
                    else:
                        rep = rep_item
                    
                    # 安全地获取嵌套属性
                    def safe_get(obj, key, default=None):
                        if not isinstance(obj, dict):
                            return default
                        return obj.get(key, default)
                    
                    # 安全地获取嵌套列表的第一个元素
                    def safe_get_first(obj, key, default=None):
                        if not isinstance(obj, dict):
                            return default
                        value = obj.get(key)
                        if isinstance(value, list) and len(value) > 0:
                            return value[0]
                        return default
                    
                    # 构建repertoire对象，使用安全的方法获取属性
                    repertoire_obj = {
                        "repertoire_id": safe_get(rep, "repertoire_id"),
                        "study_id": safe_get(safe_get(rep, "study", {}), "study_id"),
                        "study_title": safe_get(safe_get(rep, "study", {}), "study_title"),
                        "subject_id": safe_get(safe_get(rep, "subject", {}), "subject_id"),
                        "repository": repo_id
                    }
                    
                    # 安全地获取sample相关信息
                    sample = safe_get_first(rep, "sample")
                    if isinstance(sample, dict):
                        repertoire_obj["sample_id"] = safe_get(sample, "sample_id")
                        repertoire_obj["tissue"] = safe_get(sample, "tissue")
                        repertoire_obj["cell_subset"] = safe_get(sample, "cell_subset")
                    else:
                        repertoire_obj["sample_id"] = None
                        repertoire_obj["tissue"] = None
                        repertoire_obj["cell_subset"] = None
                    
                    # 安全地获取disease_state
                    subject = safe_get(rep, "subject", {})
                    diagnosis = safe_get_first(subject, "diagnosis")
                    if isinstance(diagnosis, dict):
                        repertoire_obj["disease_state"] = safe_get(diagnosis, "study_group_description")
                    else:
                        repertoire_obj["disease_state"] = None
                    
                    # 安全地获取sequence_count
                    data_proc = safe_get_first(rep, "data_processing")
                    if isinstance(data_proc, dict):
                        data_proc_files = safe_get_first(data_proc, "data_processing_files")
                        if isinstance(data_proc_files, dict):
                            repertoire_obj["sequence_count"] = safe_get(data_proc_files, "read_count")
                        else:
                            repertoire_obj["sequence_count"] = None
                    else:
                        repertoire_obj["sequence_count"] = None
                    
                    all_repertoires.append(repertoire_obj)

        result = {
            "status": "success",
            "repertoires": all_repertoires[:max_results],
            "total_repertoires": len(all_repertoires),
            "repositories_searched": repositories_searched,
            "query_parameters": {
                "disease": disease,
                "tissue": tissue,
                "species": species,
                "cell_subset": cell_subset
            }
        }

        # Cache results
        cache_manager.set("queries", cache_key, result, "repertoire")

        return result

    except Exception as e:
        logger.error(f"Error searching repertoires: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to search repertoires"
        }


@mcp.tool()
def download_airr_sequences(
    repertoire_id: str,
    filters: Optional[Dict[str, Any]] = None,
    format: str = "airr",
    max_sequences: int = 10000,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Download BCR sequences from specific repertoire in AIRR format.
    Output format is IDENTICAL to IgBLAST for direct pipeline integration.

    Args:
        repertoire_id: Repertoire identifier from search results
        filters: Additional filters:
            - v_call: V gene filter (e.g., 'IGHV3-23')
            - j_call: J gene filter (e.g., 'IGHJ4')
            - junction_aa_length: CDR3 amino acid length
            - productive: Only productive sequences (true/false)
        format: Output format - airr (TSV) or json (default: airr)
        max_sequences: Maximum sequences to download (default: 10000)
        output_dir: Optional directory to save the output file (default: system temp directory)

    Returns:
        {
            "status": "success",
            "file_path": "/tmp/airr_sequences_abc123.tsv",
            "format": "airr",
            "sequences_downloaded": 8543,
            "fields": ["sequence_id", "sequence", "v_call", ...],
            "repository": "vdjserver",
            "compatible_with_igblast": true
        }
    """
    try:
        logger.info(f"Downloading sequences for repertoire: {repertoire_id}")

        filters = filters or {}

        # Build query
        query = query_builder.build_rearrangement_query(
            repertoire_id=repertoire_id,
            v_call=filters.get("v_call"),
            d_call=filters.get("d_call"),
            j_call=filters.get("j_call"),
            junction_aa_length=filters.get("junction_aa_length"),
            productive=filters.get("productive"),
            size=1000  # Page size
        )

        # Try repositories with failover
        response = repo_manager.query_with_failover("rearrangement", query)

        if response.get("status") == "error":
            return response

        repository = response.get("_repository", "unknown")

        # Create fetch function for pagination
        def fetch_func(from_idx: int, size: int) -> Dict[str, Any]:
            q = query.copy()
            q["from"] = from_idx
            q["size"] = size
            return repo_manager.query_single(repository, "rearrangement", q)

        # Generate output file
        session_id = str(uuid.uuid4())[:8]
        file_ext = "tsv" if format == "airr" else "json"
        
        # 使用指定的输出目录或默认的临时目录
        if output_dir:
            # 确保输出目录存在
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using custom output directory: {output_dir}")
        else:
            output_path = Path(tempfile.gettempdir())
            logger.info(f"Using system temp directory: {output_path}")
            
        output_file = output_path / f"airr_sequences_{session_id}.{file_ext}"

        if format == "airr":
            # Stream to TSV file
            result = pagination_handler.stream_to_file(
                fetch_func=fetch_func,
                output_file=output_file,
                format_func=format_handler.json_to_tsv_record,
                header=format_handler.json_to_tsv_header(),
                max_records=max_sequences
            )
        else:
            # Collect JSON results
            all_sequences = pagination_handler.collect_all_results(
                fetch_func=fetch_func,
                max_records=max_sequences
            )

            # Write JSON
            import json
            with open(output_file, 'w') as f:
                json.dump(all_sequences, f, indent=2)

            result = {
                "status": "success",
                "records_written": len(all_sequences),
                "output_file": str(output_file)
            }

        if result["status"] == "success":
            return {
                "status": "success",
                "file_path": result["output_file"],
                "format": format,
                "sequences_downloaded": result["records_written"],
                "fields": format_handler.AIRR_FIELDS if format == "airr" else None,
                "repository": repository,
                "compatible_with_igblast": format == "airr",
                "filters_applied": filters
            }
        else:
            return result

    except Exception as e:
        logger.error(f"Error downloading sequences: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to download sequences"
        }


@mcp.tool()
def get_airr_study_metadata(
    study_id: str,
    repository: str = "auto"
) -> Dict[str, Any]:
    """
    Get detailed study and sample metadata.

    Args:
        study_id: Study identifier (e.g., PRJNA300878)
        repository: Repository to query - vdjserver, ireceptor, covid19, auto (default: auto)

    Returns:
        {
            "status": "success",
            "study": {
                "study_id": "PRJNA300878",
                "study_title": "...",
                "study_description": "...",
                "keywords": ["dengue", "antibody"],
                "subjects": 10,
                "samples": 45
            },
            "samples": [...],
            "repository": "vdjserver"
        }
    """
    try:
        logger.info(f"Getting study metadata: {study_id}")

        # Check cache
        cached = cache_manager.get("studies", study_id)
        if cached:
            logger.info("Returning cached study metadata")
            return cached

        # Build query
        query = query_builder.build_repertoire_query(
            study_id=study_id,
            size=1000  # Get all repertoires for this study
        )

        # Query repositories
        if repository == "auto":
            response = repo_manager.query_with_failover("repertoire", query)
            used_repository = response.get("_repository", "unknown")
        else:
            response = repo_manager.query_single(repository, "repertoire", query)
            used_repository = repository

        if "error" in response:
            return {
                "status": "error",
                "error": response.get("error"),
                "message": f"Failed to get study metadata from {used_repository}"
            }

        # Extract repertoires
        repertoires = response.get("Repertoire", [])

        if not repertoires:
            return {
                "status": "error",
                "message": f"No repertoires found for study {study_id}"
            }

        # Extract study info from first repertoire
        first_rep = repertoires[0]
        study_info = first_rep.get("study", {})

        # Extract samples
        samples = []
        subjects = set()

        for rep in repertoires:
            subject = rep.get("subject", {})
            sample_list = rep.get("sample", [])

            if subject:
                subjects.add(subject.get("subject_id"))

            for sample in sample_list:
                samples.append({
                    "sample_id": sample.get("sample_id"),
                    "subject_id": subject.get("subject_id"),
                    "tissue": sample.get("tissue"),
                    "cell_subset": sample.get("cell_subset"),
                    "collection_time_point": sample.get("collection_time_point_relative")
                })

        result = {
            "status": "success",
            "study": {
                "study_id": study_info.get("study_id"),
                "study_title": study_info.get("study_title"),
                "study_type": study_info.get("study_type"),
                "study_description": study_info.get("study_description"),
                "keywords": study_info.get("keywords_study", []),
                "pub_ids": study_info.get("pub_ids", []),
                "subjects": len(subjects),
                "samples": len(samples),
                "repertoires": len(repertoires)
            },
            "samples": samples,
            "repository": used_repository
        }

        # Cache results
        cache_manager.set("studies", study_id, result)

        return result

    except Exception as e:
        logger.error(f"Error getting study metadata: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to get study metadata"
        }


@mcp.tool()
def filter_by_vdj_genes(
    repertoire_id: str,
    v_gene: Optional[str] = None,
    d_gene: Optional[str] = None,
    j_gene: Optional[str] = None,
    combination_logic: str = "AND"
) -> Dict[str, Any]:
    """
    Filter sequences by V/D/J gene usage patterns and get statistics.

    Args:
        repertoire_id: Repertoire identifier
        v_gene: V gene family or allele (e.g., 'IGHV3', 'IGHV3-23*01')
        d_gene: D gene family or allele
        j_gene: J gene family or allele
        combination_logic: How to combine filters - AND or OR (default: AND)

    Returns:
        {
            "status": "success",
            "filtered_sequences": 1250,
            "total_sequences": 150000,
            "percentage": 0.83,
            "gene_usage_stats": {
                "v_gene_distribution": {"IGHV3-23*01": 450, ...},
                "j_gene_distribution": {"IGHJ4*01": 600, ...}
            },
            "download_available": true
        }
    """
    try:
        logger.info(f"Filtering by genes: V={v_gene}, D={d_gene}, J={j_gene}")

        # Build query
        query = query_builder.build_gene_usage_query(
            repertoire_id=repertoire_id,
            v_gene=v_gene,
            d_gene=d_gene,
            j_gene=j_gene,
            combination_logic=combination_logic.lower()
        )

        # Query with failover
        response = repo_manager.query_with_failover("rearrangement", query)

        if response.get("status") == "error":
            return response

        repository = response.get("_repository", "unknown")

        # Get sequences
        sequences = response.get("Rearrangement", [])

        # Calculate gene usage statistics
        v_gene_dist = {}
        d_gene_dist = {}
        j_gene_dist = {}

        for seq in sequences:
            v = seq.get("v_call")
            if v:
                v_gene_dist[v] = v_gene_dist.get(v, 0) + 1

            d = seq.get("d_call")
            if d:
                d_gene_dist[d] = d_gene_dist.get(d, 0) + 1

            j = seq.get("j_call")
            if j:
                j_gene_dist[j] = j_gene_dist.get(j, 0) + 1

        # Get total count (first page only for estimation)
        total_query = query_builder.build_rearrangement_query(
            repertoire_id=repertoire_id,
            size=1
        )
        total_response = repo_manager.query_single(repository, "rearrangement", total_query)

        # Try to get total count from response info
        total_count = None
        if "Info" in total_response:
            total_count = total_response["Info"].get("total_count")

        filtered_count = len(sequences)

        return {
            "status": "success",
            "filtered_sequences": filtered_count,
            "total_sequences": total_count or "unknown",
            "percentage": (filtered_count / total_count * 100) if total_count else None,
            "gene_usage_stats": {
                "v_gene_distribution": dict(sorted(v_gene_dist.items(), key=lambda x: x[1], reverse=True)[:10]),
                "d_gene_distribution": dict(sorted(d_gene_dist.items(), key=lambda x: x[1], reverse=True)[:10]),
                "j_gene_distribution": dict(sorted(j_gene_dist.items(), key=lambda x: x[1], reverse=True)[:10])
            },
            "filter_parameters": {
                "v_gene": v_gene,
                "d_gene": d_gene,
                "j_gene": j_gene,
                "combination_logic": combination_logic
            },
            "download_available": True,
            "repository": repository
        }

    except Exception as e:
        logger.error(f"Error filtering by genes: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to filter by genes"
        }


@mcp.tool()
def get_airr_statistics(
    repertoire_id: str,
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get statistical summary of repertoire characteristics.

    Args:
        repertoire_id: Repertoire identifier
        metrics: Which statistics to calculate - diversity, clonality, v_usage, cdr3_length, mutation_frequency
                 (default: all)

    Returns:
        {
            "status": "success",
            "repertoire_id": "...",
            "statistics": {
                "total_sequences": 150000,
                "unique_sequences": 125000,
                "productive_sequences": 140000,
                "v_gene_usage": {"IGHV1": 15.2, "IGHV3": 45.8, ...},
                "cdr3_length_distribution": {
                    "mean": 45,
                    "median": 42,
                    "range": [21, 81]
                }
            }
        }
    """
    try:
        logger.info(f"Getting statistics for repertoire: {repertoire_id}")

        # Check cache
        cached = cache_manager.get("repertoires", repertoire_id, "stats")
        if cached:
            logger.info("Returning cached statistics")
            return cached

        # Query sequences (sample for statistics)
        query = query_builder.build_rearrangement_query(
            repertoire_id=repertoire_id,
            size=5000  # Sample size for statistics
        )

        response = repo_manager.query_with_failover("rearrangement", query)

        if response.get("status") == "error":
            return response

        repository = response.get("_repository", "unknown")
        sequences = response.get("Rearrangement", [])

        if not sequences:
            return {
                "status": "error",
                "message": f"No sequences found for repertoire {repertoire_id}"
            }

        # Calculate statistics
        stats = format_handler.create_summary_stats(sequences)

        # Add repertoire info
        result = {
            "status": "success",
            "repertoire_id": repertoire_id,
            "sample_size": len(sequences),
            "statistics": stats,
            "repository": repository,
            "note": "Statistics based on sample of sequences"
        }

        # Cache results
        cache_manager.set("repertoires", repertoire_id, result, "stats")

        return result

    except Exception as e:
        logger.error(f"Error getting statistics: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to get statistics"
        }


if __name__ == "__main__":
    """
    启动 MCP 服务器: python airr_mcp_server.py
    """
    # 设置网络参数
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8091
    
    # 使用SSE模式启动
    mcp.run(transport="sse")
