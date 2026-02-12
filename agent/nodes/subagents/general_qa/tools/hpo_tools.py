"""
Human Phenotype Ontology (HPO) Tools
============================================
3 tools for querying Human Phenotype Ontology data

Tools:
1. query_hpo_term - Search HPO terms by ID, name, or keyword
2. query_hpo_hierarchy - Query HPO term ancestors or descendants
3. query_hpo_xref - Query HPO cross-references (UMLS, SNOMED, etc.)

Data Source: Human Phenotype Ontology (hp.obo)
- 19,533 total terms (19,077 active)
- 23,593 synonyms
- 18,165 cross-references (19 sources: UMLS, SNOMED, etc.)
- 23,434 hierarchy relationships

HPO is a standardized vocabulary for describing human disease phenotypes.
Main categories:
- HP:0000118: Phenotypic abnormality (main branch)
- HP:0000005: Mode of inheritance
- HP:0012823: Clinical modifier
- HP:0040279: Frequency
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

# HPO root and main categories
HPO_CATEGORIES = {
    "all": "HP:0000001",
    "phenotypic_abnormality": "HP:0000118",
    "mode_of_inheritance": "HP:0000005",
    "clinical_modifier": "HP:0012823",
    "frequency": "HP:0040279",
    "blood_group": "HP:0032223",
    "past_medical_history": "HP:0032443",
}


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


def _normalize_hpo_id(hpo_id: str) -> str:
    """Normalize HPO ID to standard HP:XXXXXXX format"""
    if not hpo_id:
        return hpo_id
    hpo_id = hpo_id.strip().upper()
    if hpo_id.startswith("HP:"):
        return hpo_id
    # Handle HP_0001250 or 0001250 format
    hpo_id = hpo_id.replace("HP_", "").replace("HP", "")
    if hpo_id.isdigit():
        return f"HP:{hpo_id.zfill(7)}"
    return f"HP:{hpo_id}"


# ============================================
# 1. HPO Term Search
# ============================================

class HPOTermQuery(BaseModel):
    """HPO term search query parameters"""
    term_id: Optional[str] = Field(
        None,
        description="HPO term ID (e.g., 'HP:0001250', 'HP_0001250', or '0001250'). "
                    "Returns exact match for the specified term."
    )
    name: Optional[str] = Field(
        None,
        description="HPO term name with exact matching (e.g., 'Seizure'). "
                    "Case-insensitive."
    )
    keyword: Optional[str] = Field(
        None,
        description="Keyword to search in term names and definitions (e.g., 'seizure', "
                    "'cardiac', 'immune'). Fuzzy matching."
    )
    include_synonyms: bool = Field(
        True,
        description="Include synonym matching in keyword search. Default True."
    )
    include_obsolete: bool = Field(
        False,
        description="Include obsolete/deprecated terms. Default False."
    )
    limit: int = Field(50, description="Maximum number of terms to return", ge=1, le=200)


def query_hpo_term(query: HPOTermQuery) -> List[Dict[str, Any]]:
    """
    Search Human Phenotype Ontology terms by ID, name, or keyword.
    
    USE THIS TOOL WHEN:
    - Looking up a specific phenotype by its HPO ID
    - Finding phenotypes related to a clinical concept
    - Understanding standardized phenotype definitions
    - Mapping clinical observations to HPO terms
    
    EXAMPLE QUERIES:
    - "What is HP:0001250?" (Seizure)
    - "Find phenotypes related to cardiac abnormalities"
    - "Search for immune system phenotypes"
    - "What phenotypes involve the kidney?"
    
    HPO STRUCTURE:
    - HP:0000118: Phenotypic abnormality (main branch with organ system categories)
    - HP:0000005: Mode of inheritance (autosomal dominant, recessive, etc.)
    - HP:0012823: Clinical modifier (severity, onset, etc.)
    
    Data source: Human Phenotype Ontology (19K+ terms)
    """
    con = get_connection()
    
    if not any([query.term_id, query.name, query.keyword]):
        return [{"error": "At least one search parameter required: term_id, name, or keyword"}]
    
    results = []
    
    # Search by term ID (exact match)
    if query.term_id:
        normalized_id = _normalize_hpo_id(query.term_id)
        sql = """
            SELECT 
                t.id as hpo_id,
                t.name,
                t.definition,
                t.comment,
                t.is_obsolete,
                t.replaced_by
            FROM hpo_terms t
            WHERE t.id = ?
        """
        df = con.execute(sql, [normalized_id]).fetchdf()
        if not df.empty:
            for _, row in df.iterrows():
                # Get synonyms
                syn_df = con.execute(
                    "SELECT synonym, synonym_type FROM hpo_synonyms WHERE term_id = ?",
                    [normalized_id]
                ).fetchdf()
                synonyms = [{"synonym": r['synonym'], "type": r['synonym_type']} 
                           for _, r in syn_df.iterrows()]
                
                results.append({
                    "hpo_id": row['hpo_id'],
                    "name": row['name'],
                    "definition": row['definition'],
                    "comment": row['comment'],
                    "is_obsolete": row['is_obsolete'],
                    "replaced_by": row['replaced_by'],
                    "synonyms": synonyms if synonyms else None
                })
            con.close()
            return results
    
    # Build search conditions
    conditions = []
    params = []
    
    if not query.include_obsolete:
        conditions.append("t.is_obsolete = false")
    
    if query.name:
        conditions.append("t.name ILIKE ?")
        params.append(query.name)
    
    if query.keyword:
        keyword_conditions = ["t.name ILIKE ?", "t.definition ILIKE ?"]
        params.extend([f"%{query.keyword}%", f"%{query.keyword}%"])
        
        if query.include_synonyms:
            keyword_conditions.append("""
                t.id IN (SELECT term_id FROM hpo_synonyms WHERE synonym ILIKE ?)
            """)
            params.append(f"%{query.keyword}%")
        
        conditions.append(f"({' OR '.join(keyword_conditions)})")
    
    sql = f"""
        SELECT DISTINCT
            t.id as hpo_id,
            t.name,
            t.definition,
            t.is_obsolete,
            t.replaced_by
        FROM hpo_terms t
        WHERE {' AND '.join(conditions)}
        ORDER BY t.name
        LIMIT ?
    """
    params.append(query.limit)
    
    try:
        df = con.execute(sql, params).fetchdf()
        
        for _, row in df.iterrows():
            results.append({
                "hpo_id": row['hpo_id'],
                "name": row['name'],
                "definition": row['definition'],
                "is_obsolete": row['is_obsolete'],
                "replaced_by": row['replaced_by']
            })
        
        return results if results else [{"message": "No HPO terms found matching criteria"}]
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 2. HPO Hierarchy Query
# ============================================

class HPOHierarchyDirection(str, Enum):
    """Direction for hierarchy traversal"""
    ancestors = "ancestors"
    descendants = "descendants"


class HPOHierarchyQuery(BaseModel):
    """HPO hierarchy query parameters"""
    term_id: str = Field(
        ...,
        description="HPO term ID (e.g., 'HP:0001250'). Required."
    )
    direction: HPOHierarchyDirection = Field(
        HPOHierarchyDirection.ancestors,
        description="Direction: 'ancestors' (parents up to root) or 'descendants' (children)."
    )
    max_depth: int = Field(
        10,
        description="Maximum depth to traverse. Use -1 for unlimited.",
        ge=-1, le=20
    )
    limit: int = Field(100, description="Maximum number of terms to return", ge=1, le=500)


def query_hpo_hierarchy(query: HPOHierarchyQuery) -> Dict[str, Any]:
    """
    Query HPO term hierarchy (ancestors or descendants).
    
    USE THIS TOOL WHEN:
    - Finding all parent phenotypes of a specific phenotype
    - Finding all child (more specific) phenotypes
    - Understanding the hierarchical organization of phenotypes
    - Navigating from specific to general phenotypes or vice versa
    
    EXAMPLE QUERIES:
    - "Get all ancestors of HP:0001250 (Seizure)"
    - "Find all child phenotypes of HP:0000118 (Phenotypic abnormality)"
    - "What are the parent terms of cardiac arrhythmia?"
    - "List all specific types of anemia"
    
    HIERARCHY GUIDE:
    - ancestors: More general phenotypes (e.g., Seizure → Abnormal nervous system)
    - descendants: More specific phenotypes (e.g., Seizure → Focal seizure)
    - All phenotypes eventually trace back to HP:0000001 (All)
    
    Data source: Human Phenotype Ontology (23K+ relationships)
    """
    con = get_connection()
    
    term_id = _normalize_hpo_id(query.term_id)
    
    # Determine depth limit
    depth_limit = "" if query.max_depth == -1 else f"AND depth <= {query.max_depth}"
    
    if query.direction == HPOHierarchyDirection.ancestors:
        sql = f"""
            WITH RECURSIVE hierarchy AS (
                SELECT 
                    parent_id as related_id,
                    1 as depth
                FROM hpo_hierarchy
                WHERE term_id = ?
                
                UNION ALL
                
                SELECT 
                    h.parent_id,
                    hier.depth + 1
                FROM hpo_hierarchy h
                JOIN hierarchy hier ON h.term_id = hier.related_id
                WHERE 1=1 {depth_limit}
            )
            SELECT DISTINCT
                hier.related_id as hpo_id,
                t.name,
                hier.depth
            FROM hierarchy hier
            JOIN hpo_terms t ON hier.related_id = t.id
            ORDER BY hier.depth, t.name
            LIMIT ?
        """
    else:  # descendants
        sql = f"""
            WITH RECURSIVE hierarchy AS (
                SELECT 
                    term_id as related_id,
                    1 as depth
                FROM hpo_hierarchy
                WHERE parent_id = ?
                
                UNION ALL
                
                SELECT 
                    h.term_id,
                    hier.depth + 1
                FROM hpo_hierarchy h
                JOIN hierarchy hier ON h.parent_id = hier.related_id
                WHERE 1=1 {depth_limit}
            )
            SELECT DISTINCT
                hier.related_id as hpo_id,
                t.name,
                hier.depth
            FROM hierarchy hier
            JOIN hpo_terms t ON hier.related_id = t.id
            ORDER BY hier.depth, t.name
            LIMIT ?
        """
    
    try:
        results = con.execute(sql, [term_id, query.limit]).fetchdf()
        
        # Get the queried term info
        term_info = con.execute(
            "SELECT name FROM hpo_terms WHERE id = ?", [term_id]
        ).fetchdf()
        term_name = term_info.iloc[0]['name'] if not term_info.empty else "Unknown"
        
        formatted = []
        for _, row in results.iterrows():
            formatted.append({
                "hpo_id": row['hpo_id'],
                "name": row['name'],
                "depth": int(row['depth'])
            })
        
        return {
            "query_term": {
                "hpo_id": term_id,
                "name": term_name
            },
            "direction": query.direction.value,
            "total_found": len(formatted),
            "results": formatted
        }
    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}
    finally:
        con.close()


# ============================================
# 3. HPO Cross-Reference Query
# ============================================

class HPOXrefSource(str, Enum):
    """HPO cross-reference sources"""
    UMLS = "UMLS"
    SNOMED = "SNOMEDCT_US"
    NCIT = "NCIT"
    MEDDRA = "MEDDRA"
    ICD10 = "ICD-10"
    ORPHA = "ORPHA"
    ALL = "ALL"


class HPOXrefQuery(BaseModel):
    """HPO cross-reference query parameters"""
    term_id: Optional[str] = Field(
        None,
        description="HPO term ID to get cross-references for (e.g., 'HP:0001250')."
    )
    xref_id: Optional[str] = Field(
        None,
        description="External ID to find corresponding HPO term (e.g., 'C0036572' for UMLS)."
    )
    xref_source: HPOXrefSource = Field(
        HPOXrefSource.ALL,
        description="Filter by cross-reference source: UMLS, SNOMEDCT_US, NCIT, MEDDRA, etc."
    )
    limit: int = Field(50, description="Maximum number of results to return", ge=1, le=200)


def query_hpo_xref(query: HPOXrefQuery) -> List[Dict[str, Any]]:
    """
    Query HPO cross-references to external coding systems.
    
    USE THIS TOOL WHEN:
    - Mapping HPO terms to UMLS concepts
    - Finding SNOMED-CT codes for phenotypes
    - Converting between different medical coding systems
    - Linking phenotypes to external databases
    
    EXAMPLE QUERIES:
    - "What is the UMLS code for HP:0001250 (Seizure)?"
    - "Find the HPO term for UMLS concept C0036572"
    - "Get all SNOMED codes for cardiac phenotypes"
    - "Map this phenotype to MedDRA terminology"
    
    CROSS-REFERENCE SOURCES:
    - UMLS: Unified Medical Language System (12,921 mappings)
    - SNOMEDCT_US: SNOMED Clinical Terms (4,621 mappings)
    - NCIT: NCI Thesaurus (221 mappings)
    - MEDDRA: Medical Dictionary for Regulatory Activities
    - ICD-10: International Classification of Diseases
    - ORPHA: Orphanet rare disease database
    
    Data source: Human Phenotype Ontology (18K+ cross-references)
    """
    con = get_connection()
    
    if not any([query.term_id, query.xref_id]):
        return [{"error": "At least one parameter required: term_id or xref_id"}]
    
    conditions = []
    params = []
    
    if query.term_id:
        term_id = _normalize_hpo_id(query.term_id)
        conditions.append("x.term_id = ?")
        params.append(term_id)
    
    if query.xref_id:
        conditions.append("x.xref_id ILIKE ?")
        params.append(f"%{query.xref_id}%")
    
    if query.xref_source != HPOXrefSource.ALL:
        conditions.append("x.xref_source = ?")
        params.append(query.xref_source.value)
    
    sql = f"""
        SELECT 
            x.term_id as hpo_id,
            t.name as hpo_name,
            x.xref_source,
            x.xref_id
        FROM hpo_xrefs x
        JOIN hpo_terms t ON x.term_id = t.id
        WHERE {' AND '.join(conditions)}
        ORDER BY x.xref_source, x.term_id
        LIMIT ?
    """
    params.append(query.limit)
    
    try:
        results = con.execute(sql, params).fetchdf()
        
        formatted = []
        for _, row in results.iterrows():
            formatted.append({
                "hpo_id": row['hpo_id'],
                "hpo_name": row['hpo_name'],
                "xref_source": row['xref_source'],
                "xref_id": row['xref_id']
            })
        
        return formatted if formatted else [{"message": "No cross-references found matching criteria"}]
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()
