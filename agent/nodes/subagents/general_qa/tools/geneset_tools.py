"""
Gene Set Tools
============================================
2 tools covering 16 gene set data tables

Tool List:
1. query_msigdb - MSigDB human gene set query (10 tables)
2. query_mousemine - MouseMine mouse gene set query (6 tables)

Data Sources:
- MSigDB: Molecular Signatures Database (https://www.gsea-msigdb.org/)
- MouseMine: Mouse Genome Informatics gene sets
"""

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from .db_config import get_db_path, check_db_exists

# Database path (will be loaded from config)
DB_PATH = get_db_path()


def get_connection():
    """Get read-only database connection"""
    if not DUCKDB_AVAILABLE:
        raise ImportError("duckdb is not installed. Please install it with: pip install duckdb")
    
    db_path = get_db_path()
    if not check_db_exists():
        raise FileNotFoundError(
            f"DuckDB database file not found at: {db_path}\n"
            f"Please set the BIOINFO_DB_PATH or DUCKDB_DB_PATH environment variable to the correct path,\n"
            f"or ensure the database file exists at the default location."
        )
    
    return duckdb.connect(db_path, read_only=True)


# ============================================
# 1. MSigDB Human Gene Set Query (10 tables, 36K records)
# ============================================

class MSigDBCollection(str, Enum):
    """
    MSigDB gene set collection types
    
    MSigDB organizes gene sets by function and source:
    - H (Hallmark): 50 landmark pathways representing well-defined biological states
    - C1: Positional gene sets by chromosome location
    - C2: Curated gene sets (KEGG, Reactome, BioCarta pathways)
    - C3: Regulatory target gene sets (TF targets, miRNA targets)
    - C4: Computational gene sets (cancer modules)
    - C5: GO ontology gene sets (BP/CC/MF)
    - C6: Oncogenic signature gene sets
    - C7: Immunologic signature gene sets
    - C8: Cell type signature gene sets
    """
    hallmark = "msigdb_human_h_hallmark_geneset"
    c1_positional = "msigdb_human_c1_positional_geneset"
    c2_curated = "msigdb_human_c2_curated_geneset"
    c3_regulatory = "msigdb_human_c3_regulatory_target_geneset"
    c3_tf_gtrd = "msigdb_human_c3_subset_transcription_factor_targets_from_gtrd"
    c4_computational = "msigdb_human_c4_computational_geneset"
    c5_ontology = "msigdb_human_c5_ontology_geneset"
    c6_oncogenic = "msigdb_human_c6_oncogenic_signature_geneset"
    c7_immunologic = "msigdb_human_c7_immunologic_signature_geneset"
    c8_celltype = "msigdb_human_c8_celltype_signature_geneset"
    all = "all"


class MSigDBQuery(BaseModel):
    """MSigDB gene set query parameters"""
    gene_symbol: Optional[str] = Field(
        None, 
        description="Gene symbol to find all gene sets containing this gene (e.g., 'TP53', 'BRCA1', 'MYC')"
    )
    geneset_name: Optional[str] = Field(
        None,
        description="Gene set name keyword with fuzzy matching (e.g., 'HALLMARK_APOPTOSIS', 'KEGG_CELL_CYCLE')"
    )
    collection: MSigDBCollection = Field(
        MSigDBCollection.all,
        description="Gene set collection. 'hallmark' recommended for pathway analysis, "
                    "'c7_immunologic' for immunology, 'c2_curated' for KEGG/Reactome pathways."
    )
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_msigdb(query: MSigDBQuery) -> List[Dict[str, Any]]:
    """
    Query MSigDB (Molecular Signatures Database) human gene sets.
    
    USE THIS TOOL WHEN:
    - Performing Gene Set Enrichment Analysis (GSEA)
    - Pathway analysis and functional annotation
    - Identifying biological processes a gene participates in
    - Finding co-expression gene modules
    
    COLLECTIONS:
    - **H (Hallmark, 50)**: Landmark pathways - RECOMMENDED for pathway analysis
    - **C2 (Curated, 7.4K)**: KEGG/Reactome/BioCarta pathways
    - **C5 (Ontology, 16K)**: GO terms (BP/CC/MF)
    - **C7 (Immunologic, 5.2K)**: Immunology signatures
    - **C8 (Cell Type, 840)**: Cell type signatures for single-cell
    - **C3 (Regulatory, 4.8K)**: TF/miRNA targets
    - **C6 (Oncogenic, 189)**: Oncogenic signatures
    
    EXAMPLE QUERIES:
    - "What pathways does TP53 participate in?"
    - "What genes are in HALLMARK_APOPTOSIS?"
    - "Get the KEGG_CELL_CYCLE gene list"
    
    Data source: MSigDB (36K gene set records)
    """
    con = get_connection()
    
    # 确定要查询的表
    if query.collection == MSigDBCollection.all:
        tables = [(e.name, e.value) for e in MSigDBCollection if e != MSigDBCollection.all]
    else:
        tables = [(query.collection.name, query.collection.value)]
    
    all_results = []
    
    for coll_name, table in tables:
        sql = f"""
            SELECT systematicName, collection, geneSymbols, 
                   msigdbURL, exactSource, pmid,
                   '{coll_name}' as collection_type
            FROM {table} WHERE 1=1
        """
        params = []
        
        if query.gene_symbol:
            sql += " AND geneSymbols ILIKE ?"
            params.append(f"%'{query.gene_symbol}'%")
        
        if query.geneset_name:
            sql += " AND systematicName ILIKE ?"
            params.append(f"%{query.geneset_name}%")
        
        remaining = query.limit - len(all_results)
        if remaining <= 0:
            break
        sql += f" LIMIT {remaining}"
        
        try:
            results = con.execute(sql, params).fetchdf()
            all_results.extend(results.to_dict(orient="records"))
        except Exception as e:
            continue
    
    con.close()
    
    if not all_results:
        return [{"message": "No matching gene sets found", 
                 "hint": "Check gene symbol or gene set name. Gene set names are usually uppercase like 'HALLMARK_APOPTOSIS'",
                 "query": query.model_dump()}]
    
    return all_results


# ============================================
# 2. MouseMine Mouse Gene Set Query (6 tables, 16K records)
# ============================================

class MouseMineCollection(str, Enum):
    """
    MouseMine mouse gene set collection types
    
    Mouse ortholog versions of MSigDB human gene sets:
    - MH (Hallmark): Mouse hallmark pathways
    - M1: Positional gene sets
    - M2: Curated gene sets
    - M3: Regulatory target gene sets
    - M5: GO ontology gene sets
    - M8: Cell type signature gene sets
    """
    mh_hallmark = "mousemine_mh_hallmark_geneset"
    m1_positional = "mousemine_m1_positional_geneset"
    m2_curated = "mousemine_m2_curated_geneset"
    m3_regulatory = "mousemine_m3_regulatory_target_geneset"
    m5_ontology = "mousemine_m5_ontology_geneset"
    m8_celltype = "mousemine_m8_celltype_signature_geneset"
    all = "all"


class MouseMineQuery(BaseModel):
    """MouseMine mouse gene set query parameters"""
    gene_symbol: Optional[str] = Field(
        None, 
        description="Mouse gene symbol (e.g., 'Trp53', 'Brca1'). "
                    "Note: Mouse genes use first letter capitalized (Trp53 = human TP53)."
    )
    geneset_name: Optional[str] = Field(
        None,
        description="Gene set name keyword with fuzzy matching."
    )
    collection: MouseMineCollection = Field(
        MouseMineCollection.all,
        description="Gene set collection type."
    )
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_mousemine(query: MouseMineQuery) -> List[Dict[str, Any]]:
    """
    Query MouseMine mouse gene set database.
    
    USE THIS TOOL WHEN:
    - Gene set enrichment analysis in mouse models
    - Mouse gene functional annotation
    - Human-mouse ortholog pathway analysis
    - Mouse phenotype-related gene set queries
    
    RELATIONSHIP TO MSIGDB:
    MouseMine gene sets are mouse ortholog versions of MSigDB human gene sets,
    enabling consistent pathway analysis between human and mouse.
    
    COLLECTION MAPPING:
    - MH = H (Hallmark), M2 = C2 (Curated), M5 = C5 (GO), M8 = C8 (Cell Type)
    
    MOUSE GENE SYMBOL FORMAT:
    - First letter capitalized, rest lowercase (e.g., Trp53 = human TP53)
    
    Data source: MouseMine/MGI (16K mouse gene set records)
    """
    con = get_connection()
    
    # 确定要查询的表
    if query.collection == MouseMineCollection.all:
        tables = [(e.name, e.value) for e in MouseMineCollection if e != MouseMineCollection.all]
    else:
        tables = [(query.collection.name, query.collection.value)]
    
    all_results = []
    
    for coll_name, table in tables:
        sql = f"""
            SELECT systematicName, collection, geneSymbols, 
                   msigdbURL, exactSource, pmid,
                   '{coll_name}' as collection_type
            FROM {table} WHERE 1=1
        """
        params = []
        
        if query.gene_symbol:
            sql += " AND geneSymbols ILIKE ?"
            params.append(f"%'{query.gene_symbol}'%")
        
        if query.geneset_name:
            sql += " AND systematicName ILIKE ?"
            params.append(f"%{query.geneset_name}%")
        
        remaining = query.limit - len(all_results)
        if remaining <= 0:
            break
        sql += f" LIMIT {remaining}"
        
        try:
            results = con.execute(sql, params).fetchdf()
            all_results.extend(results.to_dict(orient="records"))
        except Exception as e:
            continue
    
    con.close()
    
    if not all_results:
        return [{"message": "No matching mouse gene sets found", 
                 "hint": "Mouse gene symbols use first letter capitalized, e.g., 'Trp53' not 'TP53'",
                 "query": query.model_dump()}]
    
    return all_results


# ============================================
# Statistics Tool
# ============================================

def get_geneset_stats() -> Dict[str, Any]:
    """
    Get gene set database statistics.
    
    Returns record counts for MSigDB and MouseMine collections.
    """
    con = get_connection()
    
    stats = {
        "msigdb_human": {},
        "mousemine": {},
        "total_msigdb": 0,
        "total_mousemine": 0
    }
    
    msigdb_tables = [
        ("H_Hallmark", "msigdb_human_h_hallmark_geneset"),
        ("C1_Positional", "msigdb_human_c1_positional_geneset"),
        ("C2_Curated", "msigdb_human_c2_curated_geneset"),
        ("C3_Regulatory", "msigdb_human_c3_regulatory_target_geneset"),
        ("C4_Computational", "msigdb_human_c4_computational_geneset"),
        ("C5_Ontology", "msigdb_human_c5_ontology_geneset"),
        ("C6_Oncogenic", "msigdb_human_c6_oncogenic_signature_geneset"),
        ("C7_Immunologic", "msigdb_human_c7_immunologic_signature_geneset"),
        ("C8_CellType", "msigdb_human_c8_celltype_signature_geneset"),
    ]
    
    mousemine_tables = [
        ("MH_Hallmark", "mousemine_mh_hallmark_geneset"),
        ("M2_Curated", "mousemine_m2_curated_geneset"),
        ("M5_Ontology", "mousemine_m5_ontology_geneset"),
        ("M8_CellType", "mousemine_m8_celltype_signature_geneset"),
    ]
    
    try:
        for name, table in msigdb_tables:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats["msigdb_human"][name] = count
            stats["total_msigdb"] += count
        
        for name, table in mousemine_tables:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats["mousemine"][name] = count
            stats["total_mousemine"] += count
            
    except Exception as e:
        stats["error"] = str(e)
    finally:
        con.close()
    
    return stats
