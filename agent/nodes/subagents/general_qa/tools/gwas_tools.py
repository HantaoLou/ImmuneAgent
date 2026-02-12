"""
GWAS Catalog Tools
============================================
1 tool for querying GWAS (Genome-Wide Association Studies) data

Tool:
1. query_gwas_catalog - Query SNP-disease/trait associations from GWAS Catalog

Data Source: NHGRI-EBI GWAS Catalog (https://www.ebi.ac.uk/gwas/)
- 622K+ SNP-trait association records
- Curated from published GWAS studies
- Includes effect sizes, p-values, and mapped genes
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
# GWAS Catalog Query (622K rows)
# ============================================

class GWASCatalogQuery(BaseModel):
    """GWAS Catalog query parameters"""
    disease_trait: Optional[str] = Field(
        None,
        description="Disease or trait name with fuzzy matching (e.g., 'diabetes', 'cancer', "
                    "'blood pressure', 'BMI', 'Alzheimer')"
    )
    gene: Optional[str] = Field(
        None,
        description="Gene symbol to find associated GWAS loci (e.g., 'TP53', 'BRCA1', 'APOE'). "
                    "Searches both reported and mapped genes."
    )
    snp: Optional[str] = Field(
        None,
        description="SNP rsID (e.g., 'rs12345', 'rs429358'). Find specific variant associations."
    )
    chromosome: Optional[str] = Field(
        None,
        description="Chromosome number (e.g., '1', '17', 'X'). Filter by genomic location."
    )
    p_value_threshold: float = Field(
        5e-8,
        description="P-value significance threshold. Default 5e-8 is genome-wide significance. "
                    "Use larger values (e.g., 1e-5) for suggestive associations."
    )
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_gwas_catalog(query: GWASCatalogQuery) -> List[Dict[str, Any]]:
    """
    Query GWAS Catalog for SNP-disease/trait associations.
    
    USE THIS TOOL WHEN:
    - Finding genetic variants (SNPs) associated with a disease or trait
    - Understanding which diseases/traits are linked to a specific gene
    - Looking up information about a specific SNP (rsID)
    - Exploring genetic architecture of complex diseases
    - Finding candidate genes in a genomic region
    
    EXAMPLE QUERIES:
    - "What SNPs are associated with Type 2 Diabetes?"
    - "Which GWAS loci map to the APOE gene?"
    - "What is rs429358 associated with?" (APOE ε4 variant)
    - "What significant GWAS hits are on chromosome 17?"
    - "Find genetic associations with blood pressure"
    
    INTERPRETATION GUIDE:
    - P-VALUE: Smaller is more significant (5e-8 = genome-wide significance)
    - OR/BETA: Effect size (OR>1 = increased risk, BETA = continuous trait effect)
    - MAPPED_GENE: Gene closest to or containing the SNP
    - REPORTED_GENE: Gene reported by study authors
    
    Data source: NHGRI-EBI GWAS Catalog (622K+ records)
    """
    con = get_connection()
    
    # At least one search criterion required
    if not any([query.disease_trait, query.gene, query.snp, query.chromosome]):
        return [{"error": "At least one search parameter required: disease_trait, gene, snp, or chromosome"}]
    
    sql = """
        SELECT 
            "DISEASE/TRAIT" as disease_trait,
            "MAPPED_GENE" as mapped_gene,
            "REPORTED GENE(S)" as reported_genes,
            "SNPS" as snp_id,
            "CHR_ID" as chromosome,
            "CHR_POS" as position,
            "P-VALUE" as p_value,
            "OR or BETA" as effect_size,
            "95% CI (TEXT)" as confidence_interval,
            "RISK ALLELE FREQUENCY" as risk_allele_freq,
            "STUDY" as study,
            "PUBMEDID" as pubmed_id,
            "FIRST AUTHOR" as first_author,
            "JOURNAL" as journal,
            "DATE" as publication_date
        FROM gwas_catalog 
        WHERE "P-VALUE" <= ?
    """
    params = [query.p_value_threshold]
    
    if query.disease_trait:
        sql += ' AND "DISEASE/TRAIT" ILIKE ?'
        params.append(f"%{query.disease_trait}%")
    
    if query.gene:
        sql += ' AND ("MAPPED_GENE" ILIKE ? OR "REPORTED GENE(S)" ILIKE ?)'
        params.append(f"%{query.gene}%")
        params.append(f"%{query.gene}%")
    
    if query.snp:
        sql += ' AND "SNPS" ILIKE ?'
        params.append(f"%{query.snp}%")
    
    if query.chromosome:
        sql += ' AND "CHR_ID" = ?'
        params.append(query.chromosome)
    
    sql += ' ORDER BY "P-VALUE" ASC LIMIT ?'
    params.append(query.limit)
    
    try:
        results = con.execute(sql, params).fetchdf()
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()
