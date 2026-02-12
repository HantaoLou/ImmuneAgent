"""
Gene Ontology (GO) Tools
============================================
3 tools for querying Gene Ontology data

Tools:
1. query_go_term - Search GO terms by name, ID, or keyword
2. query_go_hierarchy - Query GO term ancestors or descendants
3. query_go_relations - Query GO term relationships (part_of, regulates, etc.)

Data Source: Gene Ontology (go-plus.json)
- 51,742 GO terms (40,363 active)
- 216,544 relationships
- Three namespaces: biological_process, molecular_function, cellular_component
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

# GO namespace root IDs
GO_NAMESPACES = {
    "biological_process": "http://purl.obolibrary.org/obo/GO_0008150",
    "molecular_function": "http://purl.obolibrary.org/obo/GO_0003674",
    "cellular_component": "http://purl.obolibrary.org/obo/GO_0005575",
}

# Common relation predicates
RELATION_PREDICATES = {
    "is_a": "is_a",
    "part_of": "http://purl.obolibrary.org/obo/BFO_0000050",
    "has_part": "http://purl.obolibrary.org/obo/BFO_0000051",
    "regulates": "http://purl.obolibrary.org/obo/RO_0002211",
    "positively_regulates": "http://purl.obolibrary.org/obo/RO_0002213",
    "negatively_regulates": "http://purl.obolibrary.org/obo/RO_0002212",
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


def _normalize_go_id(go_id: str) -> str:
    """Normalize GO ID to full URI format"""
    if go_id.startswith("http://"):
        return go_id
    # Handle GO:0006955 or GO_0006955 format
    go_id = go_id.replace(":", "_").upper()
    if not go_id.startswith("GO_"):
        go_id = "GO_" + go_id
    return f"http://purl.obolibrary.org/obo/{go_id}"


def _format_go_id(uri: str) -> str:
    """Format GO URI to readable GO:XXXXXXX format"""
    if uri and "GO_" in uri:
        go_part = uri.split("/")[-1]
        return go_part.replace("_", ":")
    return uri


# ============================================
# 1. GO Term Search (51K+ terms)
# ============================================

class GONamespace(str, Enum):
    """GO namespace (aspect)"""
    biological_process = "biological_process"
    molecular_function = "molecular_function"
    cellular_component = "cellular_component"
    all = "all"


class GOTermQuery(BaseModel):
    """GO term search query parameters"""
    term_id: Optional[str] = Field(
        None,
        description="GO term ID (e.g., 'GO:0006955', 'GO_0006955', or '0006955'). "
                    "Returns exact match for the specified term."
    )
    name: Optional[str] = Field(
        None,
        description="GO term name with exact matching (e.g., 'immune response'). "
                    "Case-insensitive."
    )
    keyword: Optional[str] = Field(
        None,
        description="Keyword to search in term names and definitions (e.g., 'apoptosis', "
                    "'T cell', 'cytokine'). Fuzzy matching."
    )
    namespace: GONamespace = Field(
        GONamespace.all,
        description="Filter by GO namespace: 'biological_process' (BP), 'molecular_function' (MF), "
                    "'cellular_component' (CC), or 'all'."
    )
    include_obsolete: bool = Field(
        False,
        description="Include obsolete/deprecated GO terms. Default False."
    )
    limit: int = Field(50, description="Maximum number of terms to return", ge=1, le=200)


def query_go_term(query: GOTermQuery) -> List[Dict[str, Any]]:
    """
    Search Gene Ontology terms by ID, name, or keyword.
    
    USE THIS TOOL WHEN:
    - Looking up a specific GO term by its ID
    - Finding GO terms related to a biological concept
    - Exploring GO annotations for genes or proteins
    - Understanding biological processes, molecular functions, or cellular locations
    
    EXAMPLE QUERIES:
    - "What is GO:0006955?" (immune response)
    - "Find GO terms related to apoptosis"
    - "Search for T cell activation GO terms"
    - "What molecular functions involve kinase activity?"
    - "Find cellular components related to mitochondria"
    
    GO NAMESPACE GUIDE:
    - biological_process (BP): Biological objectives (e.g., apoptosis, immune response)
    - molecular_function (MF): Molecular activities (e.g., kinase activity, DNA binding)
    - cellular_component (CC): Cellular locations (e.g., nucleus, membrane)
    
    Data source: Gene Ontology (51K+ terms)
    """
    con = get_connection()
    
    if not any([query.term_id, query.name, query.keyword]):
        return [{"error": "At least one search parameter required: term_id, name, or keyword"}]
    
    # Build SQL query
    conditions = ["id LIKE '%GO_%'"]
    params = []
    
    if not query.include_obsolete:
        conditions.append("deprecated = false")
    
    if query.term_id:
        full_id = _normalize_go_id(query.term_id)
        conditions.append("id = ?")
        params.append(full_id)
    
    if query.name:
        conditions.append("lbl ILIKE ?")
        params.append(query.name)
    
    if query.keyword:
        conditions.append("(lbl ILIKE ? OR definition ILIKE ?)")
        params.append(f"%{query.keyword}%")
        params.append(f"%{query.keyword}%")
    
    sql = f"""
        SELECT 
            REPLACE(id, 'http://purl.obolibrary.org/obo/', '') as go_id,
            lbl as name,
            definition,
            synonyms,
            deprecated as is_obsolete
        FROM go_nodes 
        WHERE {' AND '.join(conditions)}
        LIMIT ?
    """
    params.append(query.limit)
    
    try:
        results = con.execute(sql, params).fetchdf()
        
        # Format results
        formatted = []
        for _, row in results.iterrows():
            go_id = row['go_id'].replace('_', ':') if row['go_id'] else None
            formatted.append({
                "go_id": go_id,
                "name": row['name'],
                "definition": row['definition'],
                "synonyms": row['synonyms'],
                "is_obsolete": row['is_obsolete']
            })
        
        # Filter by namespace if specified (requires hierarchy check)
        if query.namespace != GONamespace.all and formatted:
            namespace_root = GO_NAMESPACES.get(query.namespace.value)
            if namespace_root:
                filtered = []
                for term in formatted:
                    term_uri = _normalize_go_id(term['go_id'])
                    # Check if term is under the namespace root
                    check_sql = """
                        WITH RECURSIVE ancestors AS (
                            SELECT object FROM go_edges 
                            WHERE subject = ? AND predicate = 'is_a'
                            UNION
                            SELECT e.object FROM go_edges e
                            JOIN ancestors a ON e.subject = a.object
                            WHERE e.predicate = 'is_a'
                        )
                        SELECT COUNT(*) FROM ancestors WHERE object = ?
                    """
                    count = con.execute(check_sql, [term_uri, namespace_root]).fetchone()[0]
                    if count > 0 or term_uri == namespace_root:
                        filtered.append(term)
                formatted = filtered[:query.limit]
        
        return formatted if formatted else [{"message": "No GO terms found matching criteria"}]
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 2. GO Hierarchy Query
# ============================================

class HierarchyDirection(str, Enum):
    """Direction for hierarchy traversal"""
    ancestors = "ancestors"
    descendants = "descendants"


class GOHierarchyQuery(BaseModel):
    """GO hierarchy query parameters"""
    term_id: str = Field(
        ...,
        description="GO term ID (e.g., 'GO:0006955'). Required."
    )
    direction: HierarchyDirection = Field(
        HierarchyDirection.ancestors,
        description="Direction: 'ancestors' (parents up to root) or 'descendants' (children)."
    )
    include_part_of: bool = Field(
        True,
        description="Include part_of relationships in addition to is_a. Default True."
    )
    max_depth: int = Field(
        5,
        description="Maximum depth to traverse. Use -1 for unlimited.",
        ge=-1, le=20
    )
    limit: int = Field(100, description="Maximum number of terms to return", ge=1, le=500)


def query_go_hierarchy(query: GOHierarchyQuery) -> List[Dict[str, Any]]:
    """
    Query GO term hierarchy (ancestors or descendants).
    
    USE THIS TOOL WHEN:
    - Finding all parent terms of a GO term (generalization)
    - Finding all child terms of a GO term (specialization)
    - Understanding the hierarchical structure of GO
    - Navigating from specific to general terms or vice versa
    
    EXAMPLE QUERIES:
    - "Get all ancestors of GO:0006955 (immune response)"
    - "Find all child terms of GO:0008150 (biological process)"
    - "What are the parent terms of T cell activation?"
    - "List all descendants of apoptotic process"
    
    HIERARCHY GUIDE:
    - ancestors: More general terms (parents, grandparents, etc.)
    - descendants: More specific terms (children, grandchildren, etc.)
    - is_a: Subclass relationship (A is_a B means A is a type of B)
    - part_of: Part-whole relationship (A part_of B means A is a component of B)
    
    Data source: Gene Ontology (216K+ relationships)
    """
    con = get_connection()
    
    term_uri = _normalize_go_id(query.term_id)
    
    # Build predicate filter
    predicates = ["'is_a'"]
    if query.include_part_of:
        predicates.append(f"'{RELATION_PREDICATES['part_of']}'")
    predicate_filter = f"predicate IN ({', '.join(predicates)})"
    
    # Determine depth limit
    depth_limit = "" if query.max_depth == -1 else f"AND depth <= {query.max_depth}"
    
    if query.direction == HierarchyDirection.ancestors:
        sql = f"""
            WITH RECURSIVE hierarchy AS (
                SELECT 
                    e.object as term_id,
                    e.predicate as relation,
                    1 as depth
                FROM go_edges e
                WHERE e.subject = ? AND {predicate_filter}
                
                UNION ALL
                
                SELECT 
                    e.object,
                    e.predicate,
                    h.depth + 1
                FROM go_edges e
                JOIN hierarchy h ON e.subject = h.term_id
                WHERE {predicate_filter} {depth_limit}
            )
            SELECT DISTINCT
                REPLACE(h.term_id, 'http://purl.obolibrary.org/obo/', '') as go_id,
                n.lbl as name,
                REPLACE(h.relation, 'http://purl.obolibrary.org/obo/', '') as relation,
                h.depth
            FROM hierarchy h
            JOIN go_nodes n ON h.term_id = n.id
            ORDER BY h.depth, n.lbl
            LIMIT ?
        """
    else:  # descendants
        sql = f"""
            WITH RECURSIVE hierarchy AS (
                SELECT 
                    e.subject as term_id,
                    e.predicate as relation,
                    1 as depth
                FROM go_edges e
                WHERE e.object = ? AND {predicate_filter}
                
                UNION ALL
                
                SELECT 
                    e.subject,
                    e.predicate,
                    h.depth + 1
                FROM go_edges e
                JOIN hierarchy h ON e.object = h.term_id
                WHERE {predicate_filter} {depth_limit}
            )
            SELECT DISTINCT
                REPLACE(h.term_id, 'http://purl.obolibrary.org/obo/', '') as go_id,
                n.lbl as name,
                REPLACE(h.relation, 'http://purl.obolibrary.org/obo/', '') as relation,
                h.depth
            FROM hierarchy h
            JOIN go_nodes n ON h.term_id = n.id
            ORDER BY h.depth, n.lbl
            LIMIT ?
        """
    
    try:
        results = con.execute(sql, [term_uri, query.limit]).fetchdf()
        
        # Get the queried term info
        term_info = con.execute("""
            SELECT lbl as name FROM go_nodes WHERE id = ?
        """, [term_uri]).fetchdf()
        
        term_name = term_info.iloc[0]['name'] if not term_info.empty else "Unknown"
        
        formatted = []
        for _, row in results.iterrows():
            go_id = row['go_id'].replace('_', ':') if row['go_id'] else None
            relation = row['relation']
            if relation == "is_a":
                relation_label = "is_a"
            elif "BFO_0000050" in str(relation):
                relation_label = "part_of"
            else:
                relation_label = relation
            
            formatted.append({
                "go_id": go_id,
                "name": row['name'],
                "relation": relation_label,
                "depth": int(row['depth'])
            })
        
        return {
            "query_term": {
                "go_id": _format_go_id(term_uri),
                "name": term_name
            },
            "direction": query.direction.value,
            "total_found": len(formatted),
            "results": formatted
        }
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


# ============================================
# 3. GO Relations Query
# ============================================

class GORelationType(str, Enum):
    """GO relationship types"""
    is_a = "is_a"
    part_of = "part_of"
    has_part = "has_part"
    regulates = "regulates"
    positively_regulates = "positively_regulates"
    negatively_regulates = "negatively_regulates"
    all = "all"


class GORelationsQuery(BaseModel):
    """GO relations query parameters"""
    term_id: str = Field(
        ...,
        description="GO term ID (e.g., 'GO:0006915'). Required."
    )
    relation_type: GORelationType = Field(
        GORelationType.all,
        description="Type of relationship to query. 'all' returns all relationship types."
    )
    as_subject: bool = Field(
        True,
        description="Include relationships where the term is the subject (outgoing)."
    )
    as_object: bool = Field(
        True,
        description="Include relationships where the term is the object (incoming)."
    )
    limit: int = Field(50, description="Maximum number of relationships to return", ge=1, le=200)


def query_go_relations(query: GORelationsQuery) -> List[Dict[str, Any]]:
    """
    Query GO term relationships (part_of, regulates, etc.).
    
    USE THIS TOOL WHEN:
    - Finding what processes regulate a given process
    - Understanding part-whole relationships in GO
    - Exploring regulatory networks in biological processes
    - Finding processes that are regulated by a given term
    
    EXAMPLE QUERIES:
    - "What regulates apoptosis (GO:0006915)?"
    - "What is immune response part of?"
    - "Find all processes that positively regulate cell proliferation"
    - "What cellular components has_part relationship with nucleus?"
    
    RELATIONSHIP TYPES:
    - is_a: Subclass (A is_a B = A is a type of B)
    - part_of: Component (A part_of B = A is a part of B)
    - has_part: Contains (A has_part B = A contains B)
    - regulates: General regulation
    - positively_regulates: Increases/activates
    - negatively_regulates: Decreases/inhibits
    
    Data source: Gene Ontology (216K+ relationships)
    """
    con = get_connection()
    
    term_uri = _normalize_go_id(query.term_id)
    
    # Build predicate filter
    if query.relation_type == GORelationType.all:
        predicate_filter = "e.predicate != 'is_a'"  # Exclude is_a for relations query
    else:
        predicate_uri = RELATION_PREDICATES.get(query.relation_type.value, query.relation_type.value)
        predicate_filter = f"e.predicate = '{predicate_uri}'"
    
    # Build direction filter
    direction_conditions = []
    if query.as_subject:
        direction_conditions.append(f"e.subject = '{term_uri}'")
    if query.as_object:
        direction_conditions.append(f"e.object = '{term_uri}'")
    
    if not direction_conditions:
        return [{"error": "At least one of as_subject or as_object must be True"}]
    
    direction_filter = f"({' OR '.join(direction_conditions)})"
    
    sql = f"""
        SELECT 
            REPLACE(e.subject, 'http://purl.obolibrary.org/obo/', '') as subject_id,
            n1.lbl as subject_name,
            REPLACE(e.predicate, 'http://purl.obolibrary.org/obo/', '') as relation,
            REPLACE(e.object, 'http://purl.obolibrary.org/obo/', '') as object_id,
            n2.lbl as object_name
        FROM go_edges e
        JOIN go_nodes n1 ON e.subject = n1.id
        JOIN go_nodes n2 ON e.object = n2.id
        WHERE {predicate_filter} AND {direction_filter}
        LIMIT ?
    """
    
    try:
        results = con.execute(sql, [query.limit]).fetchdf()
        
        # Get the queried term info
        term_info = con.execute("""
            SELECT lbl as name FROM go_nodes WHERE id = ?
        """, [term_uri]).fetchdf()
        
        term_name = term_info.iloc[0]['name'] if not term_info.empty else "Unknown"
        
        formatted = []
        for _, row in results.iterrows():
            subject_id = row['subject_id'].replace('_', ':') if row['subject_id'] else None
            object_id = row['object_id'].replace('_', ':') if row['object_id'] else None
            
            # Map relation URI to readable name
            relation = row['relation']
            relation_map = {
                "BFO_0000050": "part_of",
                "BFO_0000051": "has_part",
                "RO_0002211": "regulates",
                "RO_0002213": "positively_regulates",
                "RO_0002212": "negatively_regulates",
                "RO_0002224": "starts_with",
                "RO_0002230": "ends_with",
            }
            relation_label = relation_map.get(relation, relation)
            
            formatted.append({
                "subject_id": subject_id,
                "subject_name": row['subject_name'],
                "relation": relation_label,
                "object_id": object_id,
                "object_name": row['object_name']
            })
        
        return {
            "query_term": {
                "go_id": _format_go_id(term_uri),
                "name": term_name
            },
            "relation_type": query.relation_type.value,
            "total_found": len(formatted),
            "results": formatted
        }
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()
