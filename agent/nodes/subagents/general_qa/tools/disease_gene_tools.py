"""
Disease/Gene Tools
============================================
4 tools covering 4 disease-gene association data tables

Tool List:
1. query_disgenet - Disease-gene association query (DisGeNET)
2. query_omim - Mendelian inheritance disease query (OMIM)
3. query_proteinatlas - Protein atlas/gene annotation query (Human Protein Atlas)
4. query_gene_info - Gene basic information query (Ensembl)

Data Sources:
- DisGeNET: Disease-gene association database
- OMIM: Online Mendelian Inheritance in Man
- Human Protein Atlas: Human protein expression atlas
- Ensembl: Genome annotation database
"""

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
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
# 1. DisGeNET Disease-Gene Association Query (9.8K rows)
# ============================================

class DisGeNETQuery(BaseModel):
    """DisGeNET disease-gene association query parameters"""
    disorder: Optional[str] = Field(
        None, 
        description="Disease name with fuzzy matching (e.g., 'Diabetes', 'Cancer', 'Alzheimer')"
    )
    gene: Optional[str] = Field(
        None,
        description="Gene symbol to find all associated diseases (e.g., 'TP53', 'BRCA1', 'EGFR')"
    )
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_disgenet(query: DisGeNETQuery) -> List[Dict[str, Any]]:
    """
    Query DisGeNET disease-gene association data.
    
    USE THIS TOOL WHEN:
    - Finding candidate genes associated with a disease
    - Understanding which diseases a gene mutation may cause
    - Analyzing disease associations for drug targets
    - Researching molecular mechanisms of genetic diseases
    
    EXAMPLE QUERIES:
    - "What genes are associated with Diabetes?"
    - "Which diseases are linked to TP53 mutations?"
    - "What are the causative genes for Alzheimer's disease?"
    - "What cancers can BRCA1 mutations cause?"
    
    Returns disorder names with associated gene lists (e.g., ['TP53', 'BRCA1', 'EGFR']).
    Covers rare diseases, common diseases, and various cancer types.
    
    Data source: DisGeNET (9.8K disease-gene association records)
    """
    try:
        con = get_connection()
    except FileNotFoundError as e:
        return [{"error": str(e), "message": "Database file not found. Please configure BIOINFO_DB_PATH or DUCKDB_DB_PATH environment variable."}]
    except ImportError as e:
        return [{"error": str(e), "message": "DuckDB is not installed. Please install it with: pip install duckdb"}]
    
    sql = "SELECT Disorder, Genes FROM disgenet WHERE 1=1"
    params = []
    
    if query.disorder:
        sql += " AND Disorder ILIKE ?"
        params.append(f"%{query.disorder}%")
    
    if query.gene:
        sql += " AND Genes ILIKE ?"
        params.append(f"%{query.gene}%")
    
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
# 2. OMIM Mendelian Inheritance Query (18.6K rows)
# ============================================

class OMIMQuery(BaseModel):
    """OMIM Mendelian inheritance query parameters"""
    gene_name: Optional[str] = Field(
        None, 
        description="Gene name or symbol (e.g., 'BRCA1', 'CFTR', 'DMD')"
    )
    mim_number: Optional[str] = Field(
        None,
        description="OMIM number (e.g., '113705' for BRCA1)"
    )
    phenotype: Optional[str] = Field(
        None,
        description="Phenotype/disease name (e.g., 'Breast cancer', 'Cystic fibrosis')"
    )
    chromosome: Optional[str] = Field(
        None,
        description="Chromosome location (e.g., 'chr17', 'chr7')"
    )
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_omim(query: OMIMQuery) -> List[Dict[str, Any]]:
    """
    Query OMIM (Online Mendelian Inheritance in Man) database for Mendelian genetic diseases.
    
    USE THIS TOOL WHEN:
    - Finding causative genes for single-gene (Mendelian) disorders
    - Understanding inheritance patterns (dominant/recessive/X-linked)
    - Getting chromosomal location information for genes
    - Genetic counseling and prenatal diagnosis research
    
    EXAMPLE QUERIES:
    - "What is BRCA1's information in OMIM?"
    - "What gene causes Cystic fibrosis?"
    - "What genetic disease genes are on chromosome 17?"
    - "What is the inheritance pattern for Duchenne muscular dystrophy?"
    
    OMIM NUMBER INTERPRETATION:
    - Starting with 1 or 2: Autosomal dominant
    - Starting with 3: X-linked
    - Starting with 6: Autosomal recessive
    
    Data source: OMIM (18.6K Mendelian inheritance records)
    """
    try:
        con = get_connection()
    except FileNotFoundError as e:
        return [{"error": str(e), "message": "Database file not found. Please configure BIOINFO_DB_PATH or DUCKDB_DB_PATH environment variable."}]
    except ImportError as e:
        return [{"error": str(e), "message": "DuckDB is not installed. Please install it with: pip install duckdb"}]
    
    sql = """
        SELECT Chromosome, "Cyto Location", "MIM Number", 
               "Gene/Locus And Other Related Symbols" as Gene_Symbols,
               "Gene Name", "Approved Gene Symbol",
               "Ensembl Gene ID", Phenotypes, Comments
        FROM omim WHERE 1=1
    """
    params = []
    
    if query.gene_name:
        sql += """ AND ("Gene Name" ILIKE ? 
                   OR "Gene/Locus And Other Related Symbols" ILIKE ?
                   OR "Approved Gene Symbol" ILIKE ?)"""
        pattern = f"%{query.gene_name}%"
        params.extend([pattern, pattern, pattern])
    
    if query.mim_number:
        sql += ' AND "MIM Number" = ?'
        params.append(query.mim_number)
    
    if query.phenotype:
        sql += " AND Phenotypes ILIKE ?"
        params.append(f"%{query.phenotype}%")
    
    if query.chromosome:
        sql += " AND Chromosome ILIKE ?"
        params.append(f"%{query.chromosome}%")
    
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
# 3. Human Protein Atlas Query (20.2K rows)
# ============================================

class ProteinAtlasQuery(BaseModel):
    """Human Protein Atlas query parameters"""
    gene: Optional[str] = Field(
        None, 
        description="Gene symbol (e.g., 'TP53', 'EGFR', 'CD8A')"
    )
    ensembl_id: Optional[str] = Field(
        None,
        description="Ensembl gene ID (e.g., 'ENSG00000141510')"
    )
    protein_class: Optional[str] = Field(
        None,
        description="Protein class (e.g., 'Kinase', 'Transcription factor', 'Receptor')"
    )
    chromosome: Optional[str] = Field(
        None,
        description="Chromosome number (e.g., '17', 'X')"
    )
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_proteinatlas(query: ProteinAtlasQuery) -> List[Dict[str, Any]]:
    """
    Query Human Protein Atlas (HPA) for comprehensive protein/gene annotations.
    
    USE THIS TOOL WHEN:
    - Getting comprehensive gene/protein annotation information
    - Finding proteins by functional class (kinase, receptor, etc.)
    - Looking up protein descriptions and UniProt cross-references
    - Exploring genes on a specific chromosome
    
    EXAMPLE QUERIES:
    - "What is TP53's protein class and description?"
    - "List all kinase proteins"
    - "What transcription factors are on chromosome 17?"
    - "Get CD8A protein information"
    
    PROTEIN CLASSES:
    - Kinase: Protein kinases
    - Transcription factor: DNA-binding transcription factors
    - Receptor: Cell surface and nuclear receptors
    - Enzyme: Metabolic enzymes
    - Transporter: Membrane transporters
    
    Data source: Human Protein Atlas (20.2K protein records)
    """
    con = get_connection()
    
    sql = """
        SELECT Gene, "Gene synonym", Ensembl, "Gene description",
               Uniprot, Chromosome, Position, "Protein class"
        FROM proteinatlas WHERE 1=1
    """
    params = []
    
    if query.gene:
        sql += ' AND (Gene ILIKE ? OR "Gene synonym" ILIKE ?)'
        pattern = f"%{query.gene}%"
        params.extend([pattern, pattern])
    
    if query.ensembl_id:
        sql += " AND Ensembl = ?"
        params.append(query.ensembl_id)
    
    if query.protein_class:
        sql += ' AND "Protein class" ILIKE ?'
        params.append(f"%{query.protein_class}%")
    
    if query.chromosome:
        sql += " AND Chromosome = ?"
        params.append(query.chromosome)
    
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
# 4. Gene Info Query (63K rows)
# ============================================

class GeneInfoQuery(BaseModel):
    """Gene basic information query parameters"""
    gene_id: Optional[str] = Field(
        None, 
        description="Ensembl gene ID (e.g., 'ENSG00000141510' for TP53)"
    )
    transcript_id: Optional[str] = Field(
        None,
        description="Ensembl transcript ID (e.g., 'ENST00000269305')"
    )
    chromosome: Optional[str] = Field(
        None,
        description="Chromosome number (e.g., '1', '17', 'X')"
    )
    position_start: Optional[int] = Field(
        None,
        description="Genomic start position for region queries"
    )
    position_end: Optional[int] = Field(
        None,
        description="Genomic end position for region queries"
    )
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_gene_info(query: GeneInfoQuery) -> List[Dict[str, Any]]:
    """
    Query gene basic information (genomic coordinates, transcripts).
    
    USE THIS TOOL WHEN:
    - Getting genomic coordinates for a gene by Ensembl ID
    - Finding all transcripts for a gene
    - Annotating genes in a genomic region
    - Cross-database ID mapping using Ensembl IDs
    
    EXAMPLE QUERIES:
    - "What is the genomic position of ENSG00000141510 (TP53)?"
    - "What genes are in chr17:7500000-8000000?"
    - "Which gene does this transcript belong to?"
    
    RETURNS:
    - Ensembl gene ID and transcript ID
    - Chromosome number
    - Gene start and end positions
    - Strand direction (+/-)
    - Transcript coordinates
    
    NOTE: For gene function annotations, use query_proteinatlas.
    For disease associations, use query_disgenet or query_omim.
    
    Data source: Ensembl genome annotation (63K gene/transcript records)
    """
    con = get_connection()
    
    sql = """
        SELECT gene_id, transcript_id, chr, 
               gene_start, gene_end, strand,
               transcript_start, transcript_end
        FROM gene_info WHERE 1=1
    """
    params = []
    
    if query.gene_id:
        sql += " AND gene_id = ?"
        params.append(query.gene_id)
    
    if query.transcript_id:
        sql += " AND transcript_id = ?"
        params.append(query.transcript_id)
    
    if query.chromosome:
        sql += " AND chr = ?"
        params.append(query.chromosome)
    
    if query.position_start and query.position_end:
        sql += " AND gene_start >= ? AND gene_end <= ?"
        params.extend([query.position_start, query.position_end])
    
    sql += " LIMIT ?"
    params.append(query.limit)
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()
