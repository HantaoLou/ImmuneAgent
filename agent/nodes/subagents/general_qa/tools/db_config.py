"""
Database Configuration for Biomedical Tools
============================================
Centralized configuration for DuckDB database path.
Supports environment variable override and cross-platform paths.
"""

import os
from pathlib import Path

# Default database path (Linux/Unix style)
DEFAULT_DB_PATH = "/data/duckdb/db/bioinfo_warehouse.duckdb"

# Try to get database path from environment variable
DB_PATH_ENV = os.getenv("BIOINFO_DB_PATH") or os.getenv("DUCKDB_DB_PATH")

if DB_PATH_ENV:
    # Use environment variable if set
    DB_PATH = DB_PATH_ENV
else:
    # Use default path
    DB_PATH = DEFAULT_DB_PATH

# Normalize path (handle Windows paths)
if os.name == 'nt' and DB_PATH.startswith('/'):
    # On Windows, if path starts with /, try to convert to Windows path
    # First check if it exists as-is (WSL or mounted drive)
    if not os.path.exists(DB_PATH):
        # Try common Windows locations
        possible_paths = [
            r"D:\data\duckdb\db\bioinfo_warehouse.duckdb",
            r"C:\data\duckdb\db\bioinfo_warehouse.duckdb",
            str(Path.home() / "data" / "duckdb" / "db" / "bioinfo_warehouse.duckdb"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                DB_PATH = path
                break


def get_db_path() -> str:
    """
    Get the database path, with validation.
    
    Returns:
        Database file path
        
    Raises:
        FileNotFoundError: If database file doesn't exist and DUCKDB_AVAILABLE is True
    """
    return DB_PATH


def check_db_exists() -> bool:
    """
    Check if the database file exists.
    
    Returns:
        True if database file exists, False otherwise
    """
    return os.path.exists(DB_PATH)


def get_db_path_info() -> dict:
    """
    Get information about the database path configuration.
    
    Returns:
        Dictionary with path information
    """
    return {
        "db_path": DB_PATH,
        "exists": check_db_exists(),
        "from_env": DB_PATH_ENV is not None,
        "env_var": DB_PATH_ENV,
        "default_path": DEFAULT_DB_PATH,
    }

