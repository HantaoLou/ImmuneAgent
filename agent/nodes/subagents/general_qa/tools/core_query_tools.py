"""
Core Query Tools
============================================
10 tools covering 12 core biomedical data tables

Tool List:
1. query_knowledge_graph - Biomedical knowledge graph query (kg)
2. query_tcr_mcpas - T cell receptor & antigen specificity query (mcpas_tcr)
3. query_mirdb - miRNA target prediction query (mirdb_v6_0_results)
4. query_mirtarbase - Experimentally validated miRNA-target query (mirtarbase_*)
5. query_bindingdb - Drug-target binding affinity query (bindingdb_all_202409)
6. query_gtex_expression - Tissue gene expression query (gtex_tissue_gene_tpm)
7. query_sgrna_human - Human CRISPR sgRNA design (sgrna_ko_sp_human)
8. query_sgrna_mouse - Mouse CRISPR sgRNA design (sgrna_ko_sp_mouse)
9. query_genetic_interaction - Genetic interaction query (genetic_interaction)
10. query_variant - Genetic variant query (variant_table)
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
# 1. Knowledge Graph Query (kg - 8.1M rows)
# ============================================

class KGNodeType(str, Enum):
    gene = "gene/protein"
    disease = "disease"
    drug = "drug"
    pathway = "pathway"
    anatomy = "anatomy"
    biological_process = "biological_process"
    cellular_component = "cellular_component"
    molecular_function = "molecular_function"
    effect = "effect/phenotype"
    exposure = "exposure"


class KnowledgeGraphQuery(BaseModel):
    entity_id: Optional[str] = Field(None, description="Entity ID (e.g., gene ID, drug ID)")
    entity_name: Optional[str] = Field(None, description="Entity name with fuzzy matching (e.g., 'BRCA1', 'aspirin')")
    entity_type: Optional[KGNodeType] = Field(None, description="Filter by entity type")
    relation: Optional[str] = Field(None, description="Relation type (e.g., 'treats', 'associates')")
    target_type: Optional[KGNodeType] = Field(None, description="Target node type")
    limit: int = Field(50, description="Maximum number of records to return", ge=1, le=500)


def query_knowledge_graph(query: KnowledgeGraphQuery) -> List[Dict[str, Any]]:
    """
    Query biomedical knowledge graph for gene-disease-drug-pathway associations.
    
    USE THIS TOOL WHEN:
    - Finding relationships between genes and diseases
    - Discovering drug-target or drug-disease associations
    - Exploring pathway connections for a gene
    - Building molecular interaction networks
    
    Supports multi-dimensional queries by entity ID, name, type, and relation type.
    Data source: TxGNN Knowledge Graph (8.1M relationships)
    """
    try:
        con = get_connection()
    except FileNotFoundError as e:
        return [{"error": str(e), "message": "Database file not found. Please configure BIOINFO_DB_PATH or DUCKDB_DB_PATH environment variable."}]
    except ImportError as e:
        return [{"error": str(e), "message": "DuckDB is not installed. Please install it with: pip install duckdb"}]
    
    sql = """
        SELECT x_id, x_name, x_type, relation, display_relation, 
               y_id, y_name, y_type
        FROM kg WHERE 1=1
    """
    params = []
    
    if query.entity_id:
        sql += " AND (x_id = ? OR y_id = ?)"
        params.extend([query.entity_id, query.entity_id])
    
    if query.entity_name:
        if '%' in query.entity_name:
            sql += " AND (x_name LIKE ? OR y_name LIKE ?)"
        else:
            sql += " AND (x_name ILIKE ? OR y_name ILIKE ?)"
            query.entity_name = f"%{query.entity_name}%"
        params.extend([query.entity_name, query.entity_name])
    
    if query.entity_type:
        sql += " AND (x_type = ? OR y_type = ?)"
        params.extend([query.entity_type.value, query.entity_type.value])
    
    if query.relation:
        sql += " AND relation ILIKE ?"
        params.append(f"%{query.relation}%")
    
    if query.target_type:
        sql += " AND y_type = ?"
        params.append(query.target_type.value)
    
    sql += " LIMIT ?"
    params.append(query.limit)
    
    try:
        results = con.execute(sql, params).fetchdf()
        records = results.to_dict(orient="records")
        
        # 如果结果为空，提供诊断信息和建议
        if len(records) == 0:
            suggestions = []
            if query.entity_name:
                # 建议尝试更宽泛的搜索
                first_word = query.entity_name.split()[0] if query.entity_name else None
                if first_word and len(query.entity_name.split()) > 1:
                    suggestions.append(f"Try searching for '{first_word}' (first word only) for broader results")
                suggestions.append("Try removing some filters (e.g., relation or target_type) to expand search")
                suggestions.append("Check if the entity name exists in the database with a different spelling")
            
            return [{
                "result_count": 0,
                "query_info": {
                    "entity_name": query.entity_name,
                    "entity_type": query.entity_type.value if query.entity_type else None,
                    "relation": query.relation,
                    "target_type": query.target_type.value if query.target_type else None,
                },
                "suggestions": suggestions,
                "note": "Query executed successfully but no matching records found. This may indicate: 1) The search terms don't exist in the database, 2) The filters are too restrictive, or 3) The data may be stored under different terminology."
            }]
        
        return records
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 2. TCR/Immune Data Query (mcpas_tcr - 40K rows)
# ============================================

class TCRQuery(BaseModel):
    cdr3_beta: Optional[str] = Field(None, description="CDR3β sequence, supports % wildcard")
    cdr3_alpha: Optional[str] = Field(None, description="CDR3α sequence, supports % wildcard")
    epitope: Optional[str] = Field(None, description="Antigen epitope peptide sequence")
    pathology: Optional[str] = Field(None, description="Disease/pathology (e.g., 'Cancer', 'Influenza')")
    mhc: Optional[str] = Field(None, description="MHC restriction (e.g., 'HLA-A*02:01')")
    species: Optional[str] = Field("Human", description="Species")
    limit: int = Field(50, description="Maximum number of records to return", ge=1, le=500)


def query_tcr_mcpas(query: TCRQuery) -> List[Dict[str, Any]]:
    """
    Query McPAS-TCR database for T cell receptor sequences and antigen specificity data.
    
    USE THIS TOOL WHEN:
    - Searching for TCRs recognizing specific antigens or epitopes
    - Finding disease-associated T cell receptors (cancer, viral infections)
    - Analyzing immune repertoire for specific pathologies
    - Designing TCR-based immunotherapies or vaccines
    
    Contains paired TCRαβ sequences with cognate epitopes and MHC restrictions.
    Data source: McPAS-TCR (40,731 TCR-antigen pairs)
    """
    con = get_connection()
    
    sql = """
        SELECT "CDR3.alpha.aa", "CDR3.beta.aa", "Epitope.peptide", 
               Pathology, MHC, Species, "Antigen.protein", 
               TRAV, TRAJ, TRBV, TRBJ, "PubMed.ID"
        FROM mcpas_tcr WHERE 1=1
    """
    params = []
    
    if query.cdr3_beta:
        if '%' in query.cdr3_beta:
            sql += ' AND "CDR3.beta.aa" LIKE ?'
        else:
            sql += ' AND "CDR3.beta.aa" = ?'
        params.append(query.cdr3_beta)
    
    if query.cdr3_alpha:
        if '%' in query.cdr3_alpha:
            sql += ' AND "CDR3.alpha.aa" LIKE ?'
        else:
            sql += ' AND "CDR3.alpha.aa" = ?'
        params.append(query.cdr3_alpha)
    
    if query.epitope:
        sql += ' AND "Epitope.peptide" ILIKE ?'
        params.append(f"%{query.epitope}%")
    
    if query.pathology:
        sql += " AND Pathology ILIKE ?"
        params.append(f"%{query.pathology}%")
    
    if query.mhc:
        sql += " AND MHC ILIKE ?"
        params.append(f"%{query.mhc}%")
    
    if query.species:
        sql += " AND Species ILIKE ?"
        params.append(f"%{query.species}%")
    
    sql += " LIMIT ?"
    params.append(query.limit)
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 3. miRNA Target Prediction Query (mirdb_v6_0_results - 6.8M rows)
# ============================================

class MiRDBQuery(BaseModel):
    mirna: Optional[str] = Field(None, description="miRNA name (e.g., 'hsa-miR-21-5p')")
    target_gene: Optional[str] = Field(None, description="Target gene symbol (e.g., 'TP53')")
    target_accession: Optional[str] = Field(None, description="Target gene RefSeq accession")
    min_score: float = Field(80.0, description="Minimum prediction score (0-100)", ge=0, le=100)
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=1000)


def query_mirdb(query: MiRDBQuery) -> List[Dict[str, Any]]:
    """
    Query miRDB database for computational miRNA target predictions.
    
    USE THIS TOOL WHEN:
    - Finding predicted target genes for a specific miRNA
    - Discovering miRNAs that may regulate a gene of interest
    - Exploring miRNA-mRNA regulatory networks
    - Prioritizing targets based on prediction confidence scores
    
    Uses machine learning to predict miRNA-mRNA targeting. Higher scores indicate higher confidence.
    Data source: miRDB v6.0 (6.8M prediction records)
    """
    con = get_connection()
    
    sql = """
        SELECT miRNA, target_symbol, target_accession, score
        FROM mirdb_v6_0_results 
        WHERE score >= ?
    """
    params = [query.min_score]
    
    if query.mirna:
        sql += " AND miRNA ILIKE ?"
        params.append(f"%{query.mirna}%")
    
    if query.target_gene:
        sql += " AND target_symbol ILIKE ?"
        params.append(f"%{query.target_gene}%")
    
    if query.target_accession:
        sql += " AND target_accession = ?"
        params.append(query.target_accession)
    
    sql += " ORDER BY score DESC LIMIT ?"
    params.append(query.limit)
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 4. Experimentally Validated miRNA-Target Query (mirtarbase - 557K rows)
# ============================================

class MiRTarBaseQuery(BaseModel):
    mirna: Optional[str] = Field(None, description="miRNA name")
    target_gene: Optional[str] = Field(None, description="Target gene symbol")
    species: Optional[str] = Field(None, description="Species (e.g., 'Homo sapiens')")
    support_type: Optional[str] = Field(None, description="Evidence type (e.g., 'Functional MTI')")
    include_sites: bool = Field(False, description="Whether to include binding site details")
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=1000)


def query_mirtarbase(query: MiRTarBaseQuery) -> List[Dict[str, Any]]:
    """
    Query miRTarBase for experimentally validated miRNA-target gene interactions.
    
    USE THIS TOOL WHEN:
    - Finding validated (not predicted) miRNA-target relationships
    - Verifying computational predictions with experimental evidence
    - Researching miRNA regulation with literature support
    - Getting experimental methods used (Western blot, qPCR, Reporter assay, etc.)
    
    Gold-standard database with experimental validation evidence.
    Data source: miRTarBase (557K validated interaction records)
    """
    con = get_connection()
    
    sql = """
        SELECT "miRTarBase ID", miRNA, "Species (miRNA)", 
               "Target Gene", "Target Gene (Entrez ID)", 
               "Species (Target Gene)", Experiments, "Support Type",
               "References (PMID)"
        FROM mirtarbase_microrna_target_interaction WHERE 1=1
    """
    params = []
    
    if query.mirna:
        sql += " AND miRNA ILIKE ?"
        params.append(f"%{query.mirna}%")
    
    if query.target_gene:
        sql += ' AND "Target Gene" ILIKE ?'
        params.append(f"%{query.target_gene}%")
    
    if query.species:
        sql += ' AND ("Species (miRNA)" ILIKE ? OR "Species (Target Gene)" ILIKE ?)'
        params.extend([f"%{query.species}%", f"%{query.species}%"])
    
    if query.support_type:
        sql += ' AND "Support Type" ILIKE ?'
        params.append(f"%{query.support_type}%")
    
    sql += " LIMIT ?"
    params.append(query.limit)
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 5. Drug-Target Binding Affinity Query (bindingdb_all_202409 - 2.9M rows)
# ============================================

class BindingDBQuery(BaseModel):
    ligand_name: Optional[str] = Field(None, description="Ligand/drug name")
    ligand_inchi_key: Optional[str] = Field(None, description="Ligand InChI Key (exact match)")
    target_name: Optional[str] = Field(None, description="Target protein name")
    min_affinity: Optional[float] = Field(None, description="Minimum affinity (Ki/Kd/IC50, nM)")
    max_affinity: Optional[float] = Field(None, description="Maximum affinity (Ki/Kd/IC50, nM)")
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_bindingdb(query: BindingDBQuery) -> List[Dict[str, Any]]:
    """
    Query BindingDB for drug-target binding affinity data.
    
    USE THIS TOOL WHEN:
    - Finding binding affinity (Ki/Kd/IC50) between drugs and protein targets
    - Discovering compounds that bind to a specific target
    - Drug discovery and virtual screening research
    - Comparing binding potency across different compounds
    
    Contains experimental binding measurements from scientific literature.
    Data source: BindingDB 2024.09 (2.9M binding records)
    """
    con = get_connection()
    
    sql = """
        SELECT "BindingDB Ligand Name", "Ligand SMILES", "Ligand InChI Key",
               "Target Name", "Target Source Organism According to Curator or DataSource",
               "Ki (nM)", "Kd (nM)", "IC50 (nM)", "EC50 (nM)",
               "Curation/DataSource", "PubChem CID"
        FROM bindingdb_all_202409 WHERE 1=1
    """
    params = []
    
    if query.ligand_name:
        sql += ' AND "BindingDB Ligand Name" ILIKE ?'
        params.append(f"%{query.ligand_name}%")
    
    if query.ligand_inchi_key:
        sql += ' AND "Ligand InChI Key" = ?'
        params.append(query.ligand_inchi_key)
    
    if query.target_name:
        sql += ' AND "Target Name" ILIKE ?'
        params.append(f"%{query.target_name}%")
    
    sql += " LIMIT ?"
    params.append(query.limit)
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 6. Tissue Gene Expression Query (gtex_tissue_gene_tpm - 1M rows)
# ============================================

class GTExQuery(BaseModel):
    gene: Optional[str] = Field(None, description="Gene symbol (e.g., 'TP53')")
    tissue: Optional[str] = Field(None, description="Tissue name (e.g., 'Brain', 'Liver')")
    min_expression: float = Field(0.0, description="Minimum expression level (TPM)", ge=0)
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=1000)


def query_gtex_expression(query: GTExQuery) -> List[Dict[str, Any]]:
    """
    Query GTEx database for human normal tissue gene expression profiles.
    
    USE THIS TOOL WHEN:
    - Finding tissue-specific expression patterns for a gene
    - Identifying which tissues express a gene of interest
    - Comparing expression levels across different tissues
    - Selecting tissue-appropriate targets for drug development
    
    Provides gene expression (TPM) across 54 human tissues from healthy donors.
    Data source: GTEx (1M tissue-gene expression records)
    """
    con = get_connection()
    
    sql = """
        SELECT Gene, Description, Tissue, Expression
        FROM gtex_tissue_gene_tpm 
        WHERE Expression >= ?
    """
    params = [query.min_expression]
    
    if query.gene:
        sql += " AND Gene ILIKE ?"
        params.append(f"%{query.gene}%")
    
    if query.tissue:
        sql += " AND Tissue ILIKE ?"
        params.append(f"%{query.tissue}%")
    
    sql += " ORDER BY Expression DESC LIMIT ?"
    params.append(query.limit)
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 7. Human CRISPR sgRNA Design (sgrna_ko_sp_human - 5.3M rows)
# ============================================

class SgRNAQuery(BaseModel):
    target_gene: str = Field(..., description="Target gene symbol (required)")
    min_efficacy: float = Field(0.5, description="Minimum on-target efficacy score", ge=0, le=1)
    max_off_target_rank: int = Field(100, description="Maximum off-target rank", ge=1)
    limit: int = Field(20, description="Maximum number of records to return", ge=1, le=100)


def query_sgrna_human(query: SgRNAQuery) -> List[Dict[str, Any]]:
    """
    Query human CRISPR sgRNA design sequences for gene knockout experiments.
    
    USE THIS TOOL WHEN:
    - Designing CRISPR knockout experiments for human genes
    - Finding optimal sgRNA sequences with high on-target efficacy
    - Selecting guides with minimal off-target effects
    - Planning gene editing studies in human cell lines
    
    Provides sgRNA sequences, PAM, efficacy scores, and off-target rankings.
    Data source: Broad GPP (5.3M human sgRNAs)
    """
    con = get_connection()
    
    sql = """
        SELECT "Target Gene Symbol", "sgRNA Sequence", "PAM Sequence",
               "On-Target Efficacy Score", "Off-Target Rank"
        FROM sgrna_ko_sp_human 
        WHERE "Target Gene Symbol" ILIKE ?
          AND "On-Target Efficacy Score" >= ?
          AND "Off-Target Rank" <= ?
        ORDER BY "On-Target Efficacy Score" DESC, "Off-Target Rank" ASC
        LIMIT ?
    """
    params = [f"%{query.target_gene}%", query.min_efficacy, 
              query.max_off_target_rank, query.limit]
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 8. Mouse CRISPR sgRNA Design (sgrna_ko_sp_mouse - 5.1M rows)
# ============================================

def query_sgrna_mouse(query: SgRNAQuery) -> List[Dict[str, Any]]:
    """
    Query mouse CRISPR sgRNA design sequences for gene knockout experiments.
    
    USE THIS TOOL WHEN:
    - Designing CRISPR knockout experiments for mouse genes
    - Creating mouse models with gene knockouts
    - Planning in vivo gene editing studies
    
    Note: Mouse gene symbols use first letter capitalized (e.g., 'Trp53' not 'TP53').
    Data source: Broad GPP (5.1M mouse sgRNAs)
    """
    con = get_connection()
    
    sql = """
        SELECT "Target Gene Symbol", "sgRNA Sequence", "PAM Sequence",
               "On-Target Efficacy Score", "Off-Target Rank"
        FROM sgrna_ko_sp_mouse 
        WHERE "Target Gene Symbol" ILIKE ?
          AND "On-Target Efficacy Score" >= ?
          AND "Off-Target Rank" <= ?
        ORDER BY "On-Target Efficacy Score" DESC, "Off-Target Rank" ASC
        LIMIT ?
    """
    params = [f"%{query.target_gene}%", query.min_efficacy, 
              query.max_off_target_rank, query.limit]
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 9. Genetic Interaction Query (genetic_interaction - 736K rows)
# ============================================

class OrganismID(int, Enum):
    human = 9606
    mouse = 10090
    yeast = 559292
    celegans = 6239
    ecoli = 316407


class GeneticInteractionQuery(BaseModel):
    gene_a: Optional[str] = Field(None, description="Gene A ID (use Ensembl ID like 'ENSG00000141510' or yeast ID like 'YPL227C', gene symbols NOT supported)")
    gene_b: Optional[str] = Field(None, description="Gene B ID (same format as above)")
    organism_id: Optional[OrganismID] = Field(None, description="Organism ID: 9606=human, 559292=yeast, 6239=C.elegans")
    min_score: Optional[float] = Field(None, description="Minimum experimental score")
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_genetic_interaction(query: GeneticInteractionQuery) -> List[Dict[str, Any]]:
    """
    Query BioGRID genetic interaction data including synthetic lethality and genetic suppression/enhancement.
    
    USE THIS TOOL WHEN:
    - Finding synthetic lethal gene pairs for cancer drug targets
    - Discovering genetic suppressors or enhancers
    - Functional genomics research
    - Understanding gene-gene relationships beyond physical interactions
    
    IMPORTANT: Gene IDs must be Ensembl format (human: ENSG00000141510) or yeast systematic names (YPL227C).
    Gene symbols like 'TP53' are NOT supported. Data is primarily from yeast (66%), human only 0.8%.
    
    Data source: BioGRID (736K genetic interaction records)
    """
    con = get_connection()
    
    sql = """
        SELECT interaction_id, gene_a_id, gene_b_id, 
               experimental_system_type, pubmed_id,
               organism_id_a, organism_id_b, 
               throughput_type, experimental_score
        FROM genetic_interaction WHERE 1=1
    """
    params = []
    
    if query.gene_a:
        sql += " AND gene_a_id ILIKE ?"
        params.append(f"%{query.gene_a}%")
    
    if query.gene_b:
        sql += " AND gene_b_id ILIKE ?"
        params.append(f"%{query.gene_b}%")
    
    if query.organism_id:
        sql += " AND (organism_id_a = ? OR organism_id_b = ?)"
        params.extend([query.organism_id, query.organism_id])
    
    if query.min_score is not None:
        sql += " AND experimental_score >= ?"
        params.append(query.min_score)
    
    sql += " LIMIT ?"
    params.append(query.limit)
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 10. Genetic Variant Query (variant_table - 296K rows)
# ============================================

class VariantQuery(BaseModel):
    rs_id: Optional[str] = Field(None, description="RS ID (e.g., 'rs1234567')")
    chromosome: Optional[str] = Field(None, description="Chromosome (e.g., '1', 'X')")
    position: Optional[int] = Field(None, description="Genomic position")
    position_range: Optional[tuple] = Field(None, description="Position range (start, end)")
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_variant(query: VariantQuery) -> List[Dict[str, Any]]:
    """
    Query genetic variant data including SNP positions and alleles.
    
    USE THIS TOOL WHEN:
    - Looking up SNP information by RS ID
    - Finding variants in a specific genomic region
    - GWAS analysis and variant annotation
    - Genetic association studies
    
    Contains SNP positions, reference and alternate alleles.
    Data source: Variant database (296K variant records)
    """
    con = get_connection()
    
    sql = """
        SELECT RS, ID, CHR, POS, A1, A2
        FROM variant_table WHERE 1=1
    """
    params = []
    
    if query.rs_id:
        sql += " AND RS ILIKE ?"
        params.append(f"%{query.rs_id}%")
    
    if query.chromosome:
        sql += " AND CHR = ?"
        params.append(query.chromosome)
    
    if query.position:
        sql += " AND POS = ?"
        params.append(query.position)
    
    sql += " LIMIT ?"
    params.append(query.limit)
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# Statistics Tool
# ============================================

def get_core_database_stats() -> Dict[str, Any]:
    """
    Get statistics for core data tables including record counts and index status.
    
    USE THIS TOOL WHEN:
    - Checking available data coverage
    - Understanding database size and scope
    - Verifying data availability before queries
    """
    con = get_connection()
    
    core_tables = [
        "kg", "mcpas_tcr", "mirdb_v6_0_results", 
        "mirtarbase_microrna_target_interaction", "mirtarbase_microrna_target_sites",
        "bindingdb_all_202409", "gtex_tissue_gene_tpm",
        "sgrna_ko_sp_human", "sgrna_ko_sp_mouse",
        "genetic_interaction", "variant_table"
    ]
    
    stats = {"tables": {}, "total_records": 0}
    
    try:
        for table in core_tables:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats["tables"][table] = count
            stats["total_records"] += count
        
        # 获取索引信息
        indexes = con.execute("SELECT index_name, table_name FROM duckdb_indexes()").fetchall()
        stats["indexes"] = len([i for i in indexes if i[1] in core_tables])
        
    except Exception as e:
        stats["error"] = str(e)
    finally:
        con.close()
    
    return stats
