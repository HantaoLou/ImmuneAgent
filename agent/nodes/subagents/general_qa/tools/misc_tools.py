"""
Miscellaneous Tools
============================================
Covers remaining 17 data tables

Tool List:
1. query_evebio - EVE Bio drug screening data query (8 tables)
2. query_broad_repurposing - Broad drug repurposing query (2 tables)
3. query_virus_host_ppi - Virus-host protein interaction query (1 table)
4. query_depmap - DepMap cancer dependency data query (4 tables)
5. query_celltype_marker - Cell type marker query (1 table)
6. query_czi_census - CZI single-cell dataset query (1 table)
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
# 1. EVE Bio Drug Screening Data Query (8 tables)
# ============================================

class EVEBioDataType(str, Enum):
    """EVE Bio data type"""
    compound = "evebio_compound_table"
    target = "evebio_target_table"
    assay = "evebio_assay_table"
    result_detail = "evebio_detailed_result_table"
    result_summary = "evebio_summary_result_table"


class EVEBioQuery(BaseModel):
    """EVE Bio data query parameters"""
    compound_name: Optional[str] = Field(
        None, description="Compound name with fuzzy matching"
    )
    compound_id: Optional[str] = Field(
        None, description="Compound ID"
    )
    target_name: Optional[str] = Field(
        None, description="Target name"
    )
    target_id: Optional[str] = Field(
        None, description="Target ID"
    )
    data_type: EVEBioDataType = Field(
        EVEBioDataType.result_summary,
        description="Data type: compound (compound info), target (target info), "
                    "assay (experimental methods), result_detail/result_summary (screening results)"
    )
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_evebio(query: EVEBioQuery) -> List[Dict[str, Any]]:
    """
    Query EVE Bio drug screening data.
    
    USE THIS TOOL WHEN:
    - Finding compound activity profiles against targets
    - Discovering active compounds for a target
    - Drug repurposing research
    
    DATA CONTENT:
    - 1.4K compounds with DrugBank ID and CAS numbers
    - 85 targets with UniProt IDs
    - 171 assay methods
    - 239K compound-target activity records
    
    Data source: EVE Bio drug screening platform
    """
    con = get_connection()
    table = query.data_type.value
    
    if query.data_type == EVEBioDataType.compound:
        sql = "SELECT * FROM evebio_compound_table WHERE 1=1"
        params = []
        if query.compound_name:
            sql += " AND Compound ILIKE ?"
            params.append(f"%{query.compound_name}%")
        if query.compound_id:
            sql += " AND Compound_ID = ?"
            params.append(query.compound_id)
    elif query.data_type == EVEBioDataType.target:
        sql = "SELECT * FROM evebio_target_table WHERE 1=1"
        params = []
        if query.target_name:
            sql += " AND Name ILIKE ?"
            params.append(f"%{query.target_name}%")
        if query.target_id:
            sql += " AND Target_ID = ?"
            params.append(query.target_id)
    elif query.data_type == EVEBioDataType.assay:
        sql = "SELECT * FROM evebio_assay_table WHERE 1=1"
        params = []
        if query.target_id:
            sql += " AND Target_ID = ?"
            params.append(query.target_id)
    else:
        sql = f"SELECT * FROM {table} WHERE 1=1"
        params = []
        if query.compound_id:
            sql += " AND Compound_ID = ?"
            params.append(query.compound_id)
        if query.target_id:
            sql += " AND Target_ID = ?"
            params.append(query.target_id)
    
    sql += f" LIMIT {query.limit}"
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 2. Broad Drug Repurposing Query (2 tables, 27K records)
# ============================================

class BroadRepurposingQuery(BaseModel):
    """Broad drug repurposing query parameters"""
    drug_name: Optional[str] = Field(
        None, description="Drug name with fuzzy matching"
    )
    broad_id: Optional[str] = Field(
        None, description="Broad ID"
    )
    target: Optional[str] = Field(
        None, description="Drug target"
    )
    moa: Optional[str] = Field(
        None, description="Mechanism of Action"
    )
    clinical_phase: Optional[str] = Field(
        None, description="Clinical phase (e.g., 'Launched', 'Phase 3')"
    )
    include_smiles: bool = Field(
        True, description="Whether to include SMILES structure"
    )
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_broad_repurposing(query: BroadRepurposingQuery) -> List[Dict[str, Any]]:
    """
    Query Broad Institute Drug Repurposing Hub.
    
    USE THIS TOOL WHEN:
    - Drug repurposing research
    - Finding targets and mechanisms for known drugs
    - Getting drug chemical structures (SMILES)
    - Filtering drugs by clinical phase
    
    DATA CONTENT:
    - 20K+ drug molecules with SMILES structures
    - 6.8K drug-target-mechanism annotations
    - Clinical phase info (Launched/Phase 1-3)
    
    EXAMPLE QUERIES:
    - "Find all launched kinase inhibitors"
    - "What are Metformin's targets and mechanisms?"
    - "What drugs target EGFR?"
    
    Data source: Broad Institute Drug Repurposing Hub
    """
    con = get_connection()
    
    if query.include_smiles:
        sql = """
            SELECT m.broad_id, m.pert_iname as drug_name, m.smiles,
                   p.clinical_phase, p.moa, p.target
            FROM broad_repurposing_hub_molecule_with_smiles m
            LEFT JOIN broad_repurposing_hub_phase_moa_target_info p 
                ON m.pert_iname = p.pert_iname
            WHERE 1=1
        """
    else:
        sql = """
            SELECT pert_iname as drug_name, clinical_phase, moa, target
            FROM broad_repurposing_hub_phase_moa_target_info
            WHERE 1=1
        """
    params = []
    
    if query.drug_name:
        if query.include_smiles:
            sql += " AND m.pert_iname ILIKE ?"
        else:
            sql += " AND pert_iname ILIKE ?"
        params.append(f"%{query.drug_name}%")
    
    if query.broad_id and query.include_smiles:
        sql += " AND m.broad_id = ?"
        params.append(query.broad_id)
    
    if query.target:
        sql += " AND target ILIKE ?"
        params.append(f"%{query.target}%")
    
    if query.moa:
        sql += " AND moa ILIKE ?"
        params.append(f"%{query.moa}%")
    
    if query.clinical_phase:
        sql += " AND clinical_phase ILIKE ?"
        params.append(f"%{query.clinical_phase}%")
    
    sql += f" LIMIT {query.limit}"
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 3. Virus-Host Protein Interaction Query (1 table, 6.7K records)
# ============================================

class VirusHostPPIQuery(BaseModel):
    """Virus-host protein interaction query parameters"""
    viral_protein: Optional[str] = Field(
        None, description="Viral protein name"
    )
    host_gene: Optional[str] = Field(
        None, description="Host gene symbol"
    )
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_virus_host_ppi(query: VirusHostPPIQuery) -> List[Dict[str, Any]]:
    """
    Query virus-host protein interaction data.
    
    USE THIS TOOL WHEN:
    - Researching molecular mechanisms of viral infection
    - Discovering antiviral drug targets
    - Understanding how viruses hijack host cell pathways
    
    DATA CONTENT:
    - Viral protein interactions with human host genes
    - Covers multiple virus types
    
    EXAMPLE QUERIES:
    - "What human proteins does COVID-19 spike protein interact with?"
    - "Which viral proteins interact with ACE2?"
    
    Data source: P-HIPSTer 2020 virus-host interaction database
    """
    con = get_connection()
    
    sql = """
        SELECT "Viral Protien" as viral_protein, Genes as host_genes
        FROM virus_host_ppi_p_hipster_2020 WHERE 1=1
    """
    params = []
    
    if query.viral_protein:
        sql += ' AND "Viral Protien" ILIKE ?'
        params.append(f"%{query.viral_protein}%")
    
    if query.host_gene:
        sql += " AND Genes ILIKE ?"
        params.append(f"%{query.host_gene}%")
    
    sql += f" LIMIT {query.limit}"
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 4. DepMap Cancer Dependency Query (4 tables)
# ============================================

class DepMapDataType(str, Enum):
    """DepMap data type"""
    model = "depmap_model"
    crispr_dependency = "depmap_crisprgenedependency"
    crispr_effect = "depmap_crisprgeneeffect"
    expression = "depmap_omicsexpressionproteincodinggenestpmlogp1"


class DepMapQuery(BaseModel):
    """DepMap query parameters"""
    cell_line: Optional[str] = Field(
        None, description="Cell line name (e.g., 'A549', 'MCF7')"
    )
    data_type: DepMapDataType = Field(
        DepMapDataType.model,
        description="Data type: model (cell line info), crispr_dependency (CRISPR dependency), "
                    "crispr_effect (CRISPR effect), expression (gene expression)"
    )
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_depmap(query: DepMapQuery) -> List[Dict[str, Any]]:
    """
    Query DepMap cancer cell line dependency data.
    
    USE THIS TOOL WHEN:
    - Discovering cancer essential genes
    - Identifying synthetic lethal targets
    - Getting cancer cell line gene expression profiles
    - Predicting drug sensitivity
    
    DATA CONTENT:
    - 2.1K cancer cell line models
    - CRISPR gene knockout dependency scores
    - Gene expression (TPM)
    
    EXAMPLE QUERIES:
    - "What are the essential genes in A549 lung cancer cells?"
    - "Which cell lines are KRAS-dependent?"
    
    Data source: DepMap (Broad Institute Cancer Dependency Map)
    """
    con = get_connection()
    table = query.data_type.value
    
    if query.data_type == DepMapDataType.model:
        sql = "SELECT * FROM depmap_model WHERE 1=1"
        params = []
        if query.cell_line:
            sql += " AND (CellLineName ILIKE ? OR StrippedCellLineName ILIKE ?)"
            params.extend([f"%{query.cell_line}%", f"%{query.cell_line}%"])
    else:
        sql = f"SELECT * FROM {table}"
        params = []
        if query.cell_line:
            sql += " WHERE column00000 ILIKE ?"
            params.append(f"%{query.cell_line}%")
    
    sql += f" LIMIT {query.limit}"
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 5. Cell Type Marker Query (1 table, 16 records)
# ============================================

class CellTypeMarkerQuery(BaseModel):
    """Cell type marker query parameters"""
    cell_type: Optional[str] = Field(
        None, description="Cell type (e.g., 'T cell', 'B cell', 'Macrophage')"
    )
    marker_gene: Optional[str] = Field(
        None, description="Marker gene (e.g., 'CD3', 'CD19')"
    )
    limit: int = Field(50, description="Maximum number of records to return", ge=1, le=100)


def query_celltype_marker(query: CellTypeMarkerQuery) -> List[Dict[str, Any]]:
    """
    Query cell type marker genes.
    
    USE THIS TOOL WHEN:
    - Single-cell sequencing cell type annotation
    - Flow cytometry sorting strategy design
    - Immunohistochemistry marker selection
    
    DATA CONTENT:
    - Classic markers for major immune cell types
    - Cell type descriptions and marker gene lists
    
    EXAMPLE QUERIES:
    - "What are the marker genes for T cells?"
    - "Which cell types express CD8?"
    
    Data source: Immune cell marker reference database
    """
    con = get_connection()
    
    sql = """
        SELECT cell_type, description, marker_genes
        FROM marker_celltype WHERE 1=1
    """
    params = []
    
    if query.cell_type:
        sql += " AND cell_type ILIKE ?"
        params.append(f"%{query.cell_type}%")
    
    if query.marker_gene:
        sql += " AND marker_genes ILIKE ?"
        params.append(f"%{query.marker_gene}%")
    
    sql += f" LIMIT {query.limit}"
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 6. CZI Single-Cell Dataset Query (1 table, 1K records)
# ============================================

class CZICensusQuery(BaseModel):
    """CZI Census dataset query parameters"""
    tissue: Optional[str] = Field(
        None, description="Tissue type"
    )
    disease: Optional[str] = Field(
        None, description="Disease type"
    )
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_czi_census(query: CZICensusQuery) -> List[Dict[str, Any]]:
    """
    Query CZI CELLxGENE Census single-cell dataset catalog.
    
    USE THIS TOOL WHEN:
    - Discovering available single-cell RNA-seq datasets
    - Filtering datasets by tissue and disease
    - Getting dataset citation information
    
    DATA CONTENT:
    - 1000+ public single-cell datasets
    - Dataset metadata and citations
    
    Data source: CZI CELLxGENE Census v4
    """
    con = get_connection()
    
    sql = "SELECT * FROM czi_census_datasets_v4 WHERE 1=1"
    params = []
    
    sql += f" LIMIT {query.limit}"
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()
