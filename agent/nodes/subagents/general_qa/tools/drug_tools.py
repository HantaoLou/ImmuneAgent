"""
Drug Interaction Tools
============================================
1 tool covering 8 drug-drug interaction data tables

Tool:
1. query_drug_interaction - Drug-drug interaction query (DDInter 8 tables)

Data Source: DDInter (http://ddinter.scbdd.com/)
- Drug interaction database with 222K+ drug pair interaction records
- Organized into 8 sub-tables by therapeutic category
- Provides interaction severity levels (Major/Moderate/Minor/Unknown)
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
# Drug-Drug Interaction Query (DDInter)
# Covers 8 tables: 222,383 records
# ============================================

class DrugCategory(str, Enum):
    """
    Drug therapeutic category (ATC classification)
    
    Classified by primary therapeutic use:
    - antineoplastic: Antineoplastic drugs (chemotherapy, targeted therapy, immunotherapy)
    - alimentary: Alimentary and metabolism drugs (antidiabetics, GI drugs)
    - blood: Blood and blood-forming organ drugs (anticoagulants, antiplatelets)
    - dermatological: Dermatological drugs
    - respiratory: Respiratory system drugs (bronchodilators, antitussives)
    - hormonal: Hormonal preparations (corticosteroids, sex hormones)
    - antiparasitic: Antiparasitic drugs
    - various: Miscellaneous drugs
    """
    antineoplastic = "ddinter_antineoplastic"
    alimentary = "ddinter_alimentary_tract_metabolism"
    blood = "ddinter_blood_organs"
    dermatological = "ddinter_dermatological"
    respiratory = "ddinter_respiratory"
    hormonal = "ddinter_hormonal"
    antiparasitic = "ddinter_antiparasitic"
    various = "ddinter_various"
    all = "all"


class InteractionLevel(str, Enum):
    """
    Drug interaction severity level
    
    - Major: Life-threatening, avoid combination
    - Moderate: May require dose adjustment or monitoring
    - Minor: Limited clinical significance
    - Unknown: Severity not established
    """
    major = "Major"
    moderate = "Moderate"
    minor = "Minor"
    unknown = "Unknown"
    all = "all"


class DrugInteractionQuery(BaseModel):
    """Drug interaction query parameters"""
    drug_name: Optional[str] = Field(
        None, 
        description="Drug name with fuzzy matching (e.g., 'Metformin', 'Warfarin', 'Aspirin'). "
                    "Query all interactions for this drug."
    )
    drug_name_b: Optional[str] = Field(
        None,
        description="Second drug name to check for specific drug pair interaction."
    )
    category: DrugCategory = Field(
        DrugCategory.all,
        description="Drug therapeutic category. 'all' queries all categories, or select specific like "
                    "'antineoplastic' (cancer drugs), 'blood' (anticoagulants). "
                    "Recommend 'antineoplastic' for oncology patients."
    )
    severity: InteractionLevel = Field(
        InteractionLevel.all,
        description="Severity filter. 'Major' = avoid combination, "
                    "'Moderate' = adjust dose, 'Minor' = limited impact."
    )
    limit: int = Field(100, description="Maximum number of records to return", ge=1, le=500)


def query_drug_interaction(query: DrugInteractionQuery) -> List[Dict[str, Any]]:
    """
    Query drug-drug interaction (DDI) data.
    
    USE THIS TOOL WHEN:
    - Checking for dangerous drug interactions in patient medication regimens
    - Evaluating safety of drug combinations
    - Guiding clinical medication decisions and dose adjustments
    - Pharmacovigilance and adverse reaction prevention
    
    SEVERITY LEVELS:
    - **Major**: Life-threatening or permanent damage risk, AVOID combination
      Example: Warfarin + Aspirin (significantly increased bleeding risk)
    - **Moderate**: May worsen condition or require treatment adjustment, monitor closely
      Example: Metformin + contrast agents (lactic acidosis risk)
    - **Minor**: Limited clinical significance, usually no special handling needed
    - **Unknown**: Interaction mechanism or severity not established
    
    DRUG CATEGORIES:
    - **antineoplastic (65K)**: Cancer drugs (cisplatin, paclitaxel, imatinib, checkpoint inhibitors)
    - **alimentary (56K)**: GI/metabolic drugs (antidiabetics, PPIs, antiemetics)
    - **blood (15K)**: Blood drugs (anticoagulants, antiplatelets)
    - **respiratory (31K)**: Respiratory drugs (bronchodilators, corticosteroids)
    
    EXAMPLE QUERIES:
    - "What drugs have major interactions with Warfarin?"
    - "What should cancer patients on Imatinib avoid?"
    - "Is it safe to combine Aspirin and Clopidogrel?"
    
    Data source: DDInter (222K drug interaction records)
    """
    con = get_connection()
    
    # 确定要查询的表
    if query.category == DrugCategory.all:
        tables = [e.value for e in DrugCategory if e != DrugCategory.all]
    else:
        tables = [query.category.value]
    
    all_results = []
    
    for table in tables:
        # 从表名提取类别名称
        category_name = table.replace("ddinter_", "")
        
        sql = f"""
            SELECT DDInterID_A, Drug_A, DDInterID_B, Drug_B, Level,
                   '{category_name}' as drug_category
            FROM {table} WHERE 1=1
        """
        params = []
        
        if query.drug_name:
            sql += " AND (Drug_A ILIKE ? OR Drug_B ILIKE ?)"
            pattern = f"%{query.drug_name}%"
            params.extend([pattern, pattern])
        
        if query.drug_name_b:
            sql += " AND (Drug_A ILIKE ? OR Drug_B ILIKE ?)"
            pattern_b = f"%{query.drug_name_b}%"
            params.extend([pattern_b, pattern_b])
        
        if query.severity != InteractionLevel.all:
            sql += " AND Level = ?"
            params.append(query.severity.value)
        
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
        return [{"message": "No matching drug interaction records found", 
                 "hint": "Check drug name spelling. Use generic English names like 'Warfarin', 'Aspirin'",
                 "query": query.model_dump()}]
    
    return all_results


def get_drug_interaction_stats() -> Dict[str, Any]:
    """
    Get drug interaction database statistics.
    
    Returns record counts by drug category and severity level distribution.
    """
    con = get_connection()
    
    tables = [
        ("antineoplastic", "ddinter_antineoplastic"),
        ("alimentary", "ddinter_alimentary_tract_metabolism"),
        ("blood", "ddinter_blood_organs"),
        ("dermatological", "ddinter_dermatological"),
        ("respiratory", "ddinter_respiratory"),
        ("hormonal", "ddinter_hormonal"),
        ("antiparasitic", "ddinter_antiparasitic"),
        ("various", "ddinter_various"),
    ]
    
    stats = {"categories": {}, "total": 0, "by_severity": {"Major": 0, "Moderate": 0, "Minor": 0, "Unknown": 0}}
    
    try:
        for name, table in tables:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats["categories"][name] = count
            stats["total"] += count
            
            # 严重程度分布
            levels = con.execute(f"SELECT Level, COUNT(*) FROM {table} GROUP BY Level").fetchall()
            for level, cnt in levels:
                if level in stats["by_severity"]:
                    stats["by_severity"][level] += cnt
                    
    except Exception as e:
        stats["error"] = str(e)
    finally:
        con.close()
    
    return stats
