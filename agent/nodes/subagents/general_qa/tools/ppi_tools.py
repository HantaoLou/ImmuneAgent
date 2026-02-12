"""
Protein Interaction Tools
============================================
2 tools covering 10 protein/genetic interaction data tables

Tool List:
1. query_ppi - Physical protein-protein interaction query (6 tables: affinity_capture_ms/rna, 
               co_fractionation, proximity_label_ms, two_hybrid, reconstituted_complex)
2. query_synthetic_interaction - Synthetic lethality/genetic interaction query (4 tables: 
               synthetic_lethality, synthetic_growth_defect, synthetic_rescue, dosage_growth_defect)

Data Source: BioGRID (https://thebiogrid.org/)
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
# 1. Physical Protein-Protein Interaction Query (PPI)
# Covers 6 tables: 534,136 records
# ============================================

class PPIExperimentType(str, Enum):
    """
    Protein-protein interaction experimental method types
    
    Each method detects different types of protein interactions:
    - affinity_capture_ms: AP-MS, detects stable protein complexes
    - affinity_capture_rna: RNA affinity purification, detects RBP interactions
    - co_fractionation: CF-MS, detects co-migrating protein complexes
    - proximity_label_ms: BioID/APEX proximity labeling, detects transient/weak interactions
    - two_hybrid: Yeast two-hybrid, detects direct binary interactions
    - reconstituted_complex: In vitro reconstitution, validates direct physical interactions
    """
    affinity_capture_ms = "affinity_capture_ms"
    affinity_capture_rna = "affinity_capture_rna"
    co_fractionation = "co_fractionation"
    proximity_label_ms = "proximity_label_ms"
    two_hybrid = "two_hybrid"
    reconstituted_complex = "reconstituted_complex"
    all = "all"


class PPIOrganismID(int, Enum):
    """Organism ID"""
    human = 9606
    mouse = 10090
    yeast = 559292
    fly = 7227
    worm = 6239


class PPIQuery(BaseModel):
    """Protein-protein interaction query parameters"""
    gene_id: Optional[str] = Field(
        None, 
        description="Gene/protein ID (Ensembl ID like 'ENSG00000141510' or yeast ID like 'YPL227C'). "
                    "Query all interaction partners for this protein."
    )
    gene_id_b: Optional[str] = Field(
        None,
        description="Second gene ID for querying specific protein pair interactions."
    )
    experiment_type: PPIExperimentType = Field(
        PPIExperimentType.all,
        description="Experimental method type. 'all' queries all methods, or select specific method like "
                    "'affinity_capture_ms' (AP-MS protein complexes), "
                    "'proximity_label_ms' (BioID/APEX proximity labeling), "
                    "'two_hybrid' (yeast two-hybrid direct interactions)."
    )
    organism_id: Optional[PPIOrganismID] = Field(
        None,
        description="Organism filter: 9606=human, 10090=mouse, 559292=yeast"
    )
    min_score: Optional[float] = Field(
        None,
        description="Minimum experimental confidence score"
    )
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_ppi(query: PPIQuery) -> List[Dict[str, Any]]:
    """
    Query physical protein-protein interaction (PPI) data.
    
    USE THIS TOOL WHEN:
    - Finding interaction partners and protein complex compositions
    - Building PPI networks for functional enrichment analysis
    - Validating molecular mechanisms of drug targets
    - Studying protein cooperation in signaling pathways
    
    EXPERIMENTAL METHODS:
    - **affinity_capture_ms (347K)**: AP-MS, detects stable protein complexes
    - **proximity_label_ms (80K)**: BioID/APEX, detects transient/weak interactions
    - **co_fractionation (101K)**: CF-MS, infers interactions from co-migration
    - **two_hybrid (3.7K)**: Gold standard for direct binary interactions
    - **affinity_capture_rna (2.2K)**: Detects RNA-binding protein interactions
    - **reconstituted_complex (702)**: In vitro validation of direct interactions
    
    IMPORTANT: Gene IDs must be Ensembl format (human: ENSG00000141510) or yeast systematic names (YPL227C).
    
    Data source: BioGRID (534K physical interaction records)
    """
    con = get_connection()
    
    # 确定要查询的表
    if query.experiment_type == PPIExperimentType.all:
        tables = [e.value for e in PPIExperimentType if e != PPIExperimentType.all]
    else:
        tables = [query.experiment_type.value]
    
    all_results = []
    
    for table in tables:
        sql = f"""
            SELECT interaction_id, gene_a_id, gene_b_id, 
                   '{table}' as experiment_method,
                   pubmed_id, organism_id_a, organism_id_b, 
                   throughput_type, experimental_score
            FROM {table} WHERE 1=1
        """
        params = []
        
        if query.gene_id:
            sql += " AND (gene_a_id = ? OR gene_b_id = ?)"
            params.extend([query.gene_id, query.gene_id])
        
        if query.gene_id_b:
            sql += " AND (gene_a_id = ? OR gene_b_id = ?)"
            params.extend([query.gene_id_b, query.gene_id_b])
        
        if query.organism_id:
            sql += " AND (organism_id_a = ? OR organism_id_b = ?)"
            params.extend([query.organism_id.value, query.organism_id.value])
        
        if query.min_score is not None:
            sql += " AND experimental_score >= ?"
            params.append(query.min_score)
        
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
        return [{"message": "No matching protein interaction records found", "query": query.model_dump()}]
    
    return all_results


# ============================================
# 2. Synthetic Lethality/Genetic Interaction Query
# Covers 4 tables: 4,193 records
# ============================================

class SyntheticInteractionType(str, Enum):
    """
    Synthetic genetic interaction types
    
    These interactions are crucial for cancer drug discovery:
    - synthetic_lethality: Both genes inactivated leads to cell death
    - synthetic_growth_defect: Both inactivated leads to growth impairment
    - synthetic_rescue: One gene inactivation rescues another's phenotype
    - dosage_growth_defect: Dosage-dependent growth defects
    """
    synthetic_lethality = "synthetic_lethality"
    synthetic_growth_defect = "synthetic_growth_defect"
    synthetic_rescue = "synthetic_rescue"
    dosage_growth_defect = "dosage_growth_defect"
    all = "all"


class SyntheticInteractionQuery(BaseModel):
    """Synthetic lethality/genetic interaction query parameters"""
    gene_id: Optional[str] = Field(
        None, 
        description="Gene ID (Ensembl ID or yeast ID). Query all genes with genetic interactions. "
                    "Especially useful for discovering synthetic lethal partner genes."
    )
    gene_id_b: Optional[str] = Field(
        None,
        description="Second gene ID for checking if specific gene pair has genetic interaction."
    )
    interaction_type: SyntheticInteractionType = Field(
        SyntheticInteractionType.all,
        description="Genetic interaction type. 'synthetic_lethality' is most important for cancer drug discovery, "
                    "e.g., PARP inhibitors exploit BRCA1/2-PARP synthetic lethality."
    )
    organism_id: Optional[PPIOrganismID] = Field(
        None,
        description="Organism filter: 9606=human, 559292=yeast"
    )
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_synthetic_interaction(query: SyntheticInteractionQuery) -> List[Dict[str, Any]]:
    """
    Query synthetic lethality and genetic interaction data.
    
    USE THIS TOOL WHEN:
    - Finding synthetic lethal partners for cancer drug targets
    - Discovering drug combinations with synergistic effects
    - Identifying biomarkers predicting drug response
    - Researching precision oncology treatment strategies
    
    KEY CONCEPT:
    Synthetic Lethality means two genes individually inactivated allow cell survival,
    but simultaneous inactivation leads to cell death. This is a core strategy for
    precision cancer therapy (e.g., PARP inhibitors for BRCA-mutant cancers).
    
    INTERACTION TYPES:
    - **synthetic_lethality (1.9K)**: Most important for drug discovery
    - **synthetic_growth_defect (1K)**: Dual knockout impairs proliferation
    - **synthetic_rescue (74)**: One gene compensates another's loss
    - **dosage_growth_defect (1.2K)**: Gene dosage sensitivity
    
    EXAMPLE QUERIES:
    - "What are synthetic lethal targets for TP53-mutant tumors?"
    - "Which genes have genetic interactions with BRCA1?"
    - "Do these two genes have a synthetic lethal relationship?"
    
    IMPORTANT: Data is primarily from yeast models. Use yeast gene IDs like 'YPL227C'.
    
    Data source: BioGRID (4.2K genetic interaction records)
    """
    con = get_connection()
    
    # 确定要查询的表
    if query.interaction_type == SyntheticInteractionType.all:
        tables = [e.value for e in SyntheticInteractionType if e != SyntheticInteractionType.all]
    else:
        tables = [query.interaction_type.value]
    
    all_results = []
    
    for table in tables:
        sql = f"""
            SELECT interaction_id, gene_a_id, gene_b_id, 
                   '{table}' as interaction_type,
                   experimental_system_type,
                   pubmed_id, organism_id_a, organism_id_b, 
                   throughput_type, experimental_score
            FROM {table} WHERE 1=1
        """
        params = []
        
        if query.gene_id:
            sql += " AND (gene_a_id = ? OR gene_b_id = ?)"
            params.extend([query.gene_id, query.gene_id])
        
        if query.gene_id_b:
            sql += " AND (gene_a_id = ? OR gene_b_id = ?)"
            params.extend([query.gene_id_b, query.gene_id_b])
        
        if query.organism_id:
            sql += " AND (organism_id_a = ? OR organism_id_b = ?)"
            params.extend([query.organism_id.value, query.organism_id.value])
        
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
        return [{"message": "No matching genetic interaction records found", 
                 "hint": "Data is primarily from yeast. Use yeast gene IDs like 'YPL227C'",
                 "query": query.model_dump()}]
    
    return all_results


# ============================================
# Statistics Tool
# ============================================

def get_ppi_stats() -> Dict[str, Any]:
    """
    Get protein interaction database statistics.
    
    Returns record counts for each experimental method and species distribution.
    """
    con = get_connection()
    
    stats = {
        "physical_interaction": {},
        "genetic_interaction": {},
        "total_physical": 0,
        "total_genetic": 0
    }
    
    ppi_tables = ['affinity_capture_ms', 'affinity_capture_rna', 'co_fractionation', 
                  'proximity_label_ms', 'two_hybrid', 'reconstituted_complex']
    syn_tables = ['synthetic_lethality', 'synthetic_growth_defect', 
                  'synthetic_rescue', 'dosage_growth_defect']
    
    try:
        for table in ppi_tables:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats["physical_interaction"][table] = count
            stats["total_physical"] += count
        
        for table in syn_tables:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats["genetic_interaction"][table] = count
            stats["total_genetic"] += count
            
    except Exception as e:
        stats["error"] = str(e)
    finally:
        con.close()
    
    return stats
