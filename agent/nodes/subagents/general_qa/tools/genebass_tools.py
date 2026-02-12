"""
Genebass Tools
============================================
1 tool for querying gene-level rare variant burden analysis data

Tool:
1. query_genebass - Query gene-phenotype associations from exome sequencing burden tests

Data Source: Genebass (https://genebass.org/)
- UK Biobank exome sequencing data
- 77M+ gene-phenotype association records across 3 variant types:
  - pLoF (predicted Loss of Function): 24M records
  - Missense (LC): 26.5M records  
  - Synonymous: 26.5M records (negative control)
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
# Genebass Gene Burden Query (77M+ rows total)
# ============================================

class VariantType(str, Enum):
    """
    Variant functional consequence type
    
    - plof: Predicted Loss of Function (frameshift, nonsense, splice)
    - missense: Missense variants (amino acid change, low confidence pathogenic)
    - synonymous: Synonymous variants (no amino acid change, used as negative control)
    - all: Query all variant types
    """
    plof = "plof"
    missense = "missense"
    synonymous = "synonymous"
    all = "all"


# Mapping variant type to table name
VARIANT_TABLE_MAP = {
    "plof": "genebass_plof",
    "missense": "genebass_missense_lc",
    "synonymous": "genebass_synonymous"
}


class GenebassQuery(BaseModel):
    """Genebass gene burden query parameters"""
    gene: Optional[str] = Field(
        None,
        description="Gene symbol (e.g., 'BRCA1', 'TP53', 'APOE'). "
                    "Find all phenotype associations for this gene."
    )
    phenotype: Optional[str] = Field(
        None,
        description="Phenotype description with fuzzy matching (e.g., 'diabetes', 'cancer', "
                    "'blood pressure', 'BMI'). Find genes associated with this phenotype."
    )
    variant_type: VariantType = Field(
        VariantType.plof,
        description="Variant functional type. 'plof' for loss-of-function analysis (most impactful), "
                    "'missense' for protein-altering, 'synonymous' as negative control, "
                    "'all' to compare across types."
    )
    p_value_threshold: float = Field(
        1e-6,
        description="P-value significance threshold. Default 1e-6 for gene-level tests. "
                    "Use 2.5e-6 for exome-wide significance (0.05/20000 genes)."
    )
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_genebass(query: GenebassQuery) -> List[Dict[str, Any]]:
    """
    Query Genebass for gene-phenotype associations from rare variant burden tests.
    
    USE THIS TOOL WHEN:
    - Finding phenotypes associated with loss-of-function mutations in a gene
    - Identifying genes where rare variants cause a specific phenotype
    - Understanding functional consequences of gene knockout in humans
    - Validating drug targets (genes with pLoF phenotypes may be druggable)
    - Comparing effect sizes across different variant types
    
    EXAMPLE QUERIES:
    - "What phenotypes are associated with PCSK9 loss-of-function?"
    - "Which genes have pLoF associations with LDL cholesterol?"
    - "What happens when BRCA1 is knocked out in humans?"
    - "Find genes with burden associations for diabetes"
    - "Compare APOE effects across pLoF vs missense variants"
    
    INTERPRETATION GUIDE:
    - Pvalue: Combined burden test p-value (smaller = more significant)
    - Pvalue_Burden: Burden test only (assumes same direction of effect)
    - Pvalue_SKAT: SKAT test (allows bidirectional effects)
    - BETA_Burden: Effect size (negative = decreased trait, positive = increased)
    - SE_Burden: Standard error of effect estimate
    
    VARIANT TYPE GUIDE:
    - pLoF: Most severe, gene knockout equivalent (frameshift, stop-gain, splice)
    - missense: Amino acid changes, variable severity
    - synonymous: No protein change, negative control (should have no association)
    
    Data source: Genebass UK Biobank (77M+ records)
    """
    con = get_connection()
    
    # At least one search criterion required
    if not any([query.gene, query.phenotype]):
        return [{"error": "At least one search parameter required: gene or phenotype"}]
    
    # Determine which tables to query
    if query.variant_type == VariantType.all:
        tables_to_query = list(VARIANT_TABLE_MAP.items())
    else:
        tables_to_query = [(query.variant_type.value, VARIANT_TABLE_MAP[query.variant_type.value])]
    
    all_results = []
    
    for var_type, table_name in tables_to_query:
        sql = f"""
            SELECT 
                '{var_type}' as variant_type,
                gene,
                pheno_description as phenotype,
                annotation,
                Pvalue as p_value,
                Pvalue_Burden as p_value_burden,
                Pvalue_SKAT as p_value_skat,
                BETA_Burden as beta,
                SE_Burden as se
            FROM {table_name}
            WHERE Pvalue <= ?
        """
        params = [query.p_value_threshold]
        
        if query.gene:
            sql += " AND gene ILIKE ?"
            params.append(f"%{query.gene}%")
        
        if query.phenotype:
            sql += " AND pheno_description ILIKE ?"
            params.append(f"%{query.phenotype}%")
        
        sql += " ORDER BY Pvalue ASC LIMIT ?"
        params.append(query.limit)
        
        try:
            results = con.execute(sql, params).fetchdf()
            all_results.extend(results.to_dict(orient="records"))
        except Exception as e:
            all_results.append({"error": f"Query failed for {table_name}: {str(e)}"})
    
    con.close()
    
    # Sort combined results by p-value and limit
    all_results = sorted(
        [r for r in all_results if "error" not in r],
        key=lambda x: x.get("p_value", 1)
    )[:query.limit]
    
    return all_results if all_results else [{"message": "No significant associations found"}]
