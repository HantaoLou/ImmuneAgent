"""
TxGNN Drug Repurposing Tools
============================================
2 tools for querying AI-based drug repurposing predictions

Tools:
1. query_drug_for_disease - Find predicted drugs for a disease
2. query_disease_for_drug - Find predicted diseases treatable by a drug

Data Source: TxGNN (https://github.com/mims-harvard/TxGNN)
- Graph neural network-based drug repurposing model
- 3.4M drug-disease prediction pairs
- 8K drugs × 17K diseases
- Trained on biomedical knowledge graphs
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
# TxGNN Drug-Disease Prediction Queries
# ============================================

class DrugForDiseaseQuery(BaseModel):
    """Query parameters for finding drugs for a disease"""
    disease_name: str = Field(
        ...,
        description="Disease name with fuzzy matching (e.g., 'diabetes', 'cancer', 'Alzheimer', "
                    "'immunodeficiency', 'leukemia')"
    )
    min_score: float = Field(
        5.0,
        description="Minimum prediction score threshold. Higher scores indicate stronger predictions. "
                    "Typical range: 5-20. Use lower values for rare diseases."
    )
    top_k: int = Field(20, description="Return top K predicted drugs", ge=1, le=100)


class DiseaseForDrugQuery(BaseModel):
    """Query parameters for finding diseases treatable by a drug"""
    drug_name: str = Field(
        ...,
        description="Drug name with fuzzy matching (e.g., 'Metformin', 'Aspirin', 'Ibuprofen', "
                    "'Rituximab', 'Pembrolizumab')"
    )
    min_score: float = Field(
        5.0,
        description="Minimum prediction score threshold. Higher scores indicate stronger predictions. "
                    "Typical range: 5-20."
    )
    top_k: int = Field(20, description="Return top K predicted diseases", ge=1, le=100)


def query_drug_for_disease(query: DrugForDiseaseQuery) -> List[Dict[str, Any]]:
    """
    Find predicted drugs for treating a specific disease using TxGNN model.
    
    USE THIS TOOL WHEN:
    - Exploring drug repurposing candidates for a disease
    - Finding existing drugs that may treat a new indication
    - Identifying potential therapeutics for rare diseases
    - Generating hypotheses for drug development
    - Looking for alternative treatments for a condition
    
    EXAMPLE QUERIES:
    - "What drugs might treat immunodeficiency?"
    - "Find drug repurposing candidates for Alzheimer's disease"
    - "What existing drugs could work for leukemia?"
    - "Suggest drugs for autoimmune diseases"
    - "Find potential treatments for rare genetic disorders"
    
    INTERPRETATION GUIDE:
    - prediction_score: Model confidence (higher = stronger prediction)
    - Scores > 10: Strong candidates for experimental validation
    - Scores 5-10: Moderate candidates, require more evidence
    - These are AI predictions, not clinical recommendations
    
    Data source: TxGNN (3.4M predictions, 8K drugs × 17K diseases)
    """
    con = get_connection()
    
    sql = """
        SELECT 
            p.disease_name,
            d.drug_name,
            p.drug_id,
            p.prediction_score
        FROM txgnn_predictions p
        JOIN txgnn_drug_id2name d ON p.drug_id = d.drug_id
        WHERE p.disease_name ILIKE ?
          AND p.prediction_score >= ?
        ORDER BY p.prediction_score DESC
        LIMIT ?
    """
    params = [f"%{query.disease_name}%", query.min_score, query.top_k]
    
    try:
        results = con.execute(sql, params).fetchdf()
        if results.empty:
            # Try exact disease name lookup
            sql_diseases = """
                SELECT DISTINCT disease_name 
                FROM txgnn_predictions 
                WHERE disease_name ILIKE ? 
                LIMIT 10
            """
            diseases = con.execute(sql_diseases, [f"%{query.disease_name}%"]).fetchdf()
            if diseases.empty:
                return [{"message": f"No diseases found matching '{query.disease_name}'",
                        "suggestion": "Try broader search terms or check spelling"}]
            return [{"message": "No predictions above threshold",
                    "matching_diseases": diseases['disease_name'].tolist(),
                    "suggestion": "Try lowering min_score or selecting a specific disease"}]
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()


def query_disease_for_drug(query: DiseaseForDrugQuery) -> List[Dict[str, Any]]:
    """
    Find predicted diseases that a specific drug might treat using TxGNN model.
    
    USE THIS TOOL WHEN:
    - Exploring new indications for an existing drug
    - Understanding potential off-label uses
    - Drug repurposing research
    - Identifying diseases where a drug mechanism might be beneficial
    - Expanding therapeutic applications of known compounds
    
    EXAMPLE QUERIES:
    - "What diseases might Metformin treat besides diabetes?"
    - "Find new indications for Rituximab"
    - "What conditions could Aspirin help with?"
    - "Explore repurposing opportunities for Ibuprofen"
    - "What diseases might respond to Pembrolizumab?"
    
    INTERPRETATION GUIDE:
    - prediction_score: Model confidence (higher = stronger prediction)
    - Scores > 10: Strong candidates for experimental validation
    - Scores 5-10: Moderate candidates, require more evidence
    - These are AI predictions, not clinical recommendations
    
    Data source: TxGNN (3.4M predictions, 8K drugs × 17K diseases)
    """
    con = get_connection()
    
    # First find matching drug IDs
    sql_drug = """
        SELECT drug_id, drug_name 
        FROM txgnn_drug_id2name 
        WHERE drug_name ILIKE ?
        LIMIT 10
    """
    
    try:
        drugs = con.execute(sql_drug, [f"%{query.drug_name}%"]).fetchdf()
        
        if drugs.empty:
            return [{"message": f"No drugs found matching '{query.drug_name}'",
                    "suggestion": "Try different drug name or check spelling"}]
        
        # Get predictions for matching drugs
        drug_ids = drugs['drug_id'].tolist()
        placeholders = ', '.join(['?' for _ in drug_ids])
        
        sql = f"""
            SELECT 
                d.drug_name,
                p.disease_name,
                p.drug_id,
                p.prediction_score
            FROM txgnn_predictions p
            JOIN txgnn_drug_id2name d ON p.drug_id = d.drug_id
            WHERE p.drug_id IN ({placeholders})
              AND p.prediction_score >= ?
            ORDER BY p.prediction_score DESC
            LIMIT ?
        """
        params = drug_ids + [query.min_score, query.top_k]
        
        results = con.execute(sql, params).fetchdf()
        
        if results.empty:
            return [{"message": "No predictions above threshold",
                    "matching_drugs": drugs.to_dict(orient="records"),
                    "suggestion": "Try lowering min_score"}]
        
        return results.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]
    finally:
        con.close()
