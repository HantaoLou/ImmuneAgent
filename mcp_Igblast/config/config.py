"""
IgBLAST MCP Server Configuration

Essential configuration for IgBLAST and ChangeO pipeline.
"""

from pathlib import Path

# Base paths - Updated for Linux environment
IGBLAST_BASE = Path("/data_new/workspace/antibody_gen/mcp_Igblast/igblast_changeO")

# Database directories (used by igblast_mcp_server.py)
IGBLAST_ROOT = IGBLAST_BASE / "igblast"
IGBLAST_DB = IGBLAST_ROOT / "database"
IGBLAST_OPTIONAL = IGBLAST_ROOT / "optional_file"

# Configurable temporary files directory
# You can change this to any absolute path with write permissions
# Examples:
# - "/tmp" (default, system temporary directory)
# - "/data/temp" (custom temporary directory)
# - "/home/user/igblast_temp" (user-specific directory)
TEMP_DIR = Path("/tmp")

# Output directory for analysis results
OUTPUT_DIR = Path("/data_new/workspace/antibody_gen/mcp_Igblast/output")
# Automatically create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
