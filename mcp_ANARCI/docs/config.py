"""
Configuration for ANARCI MCP Server

This file defines paths and settings for the ANARCI antibody numbering tool.
Update ANARCI_BIN path when deploying to Linux server.
"""

from pathlib import Path
import os

# ANARCI binary path - UPDATE THIS ON LINUX SERVER
ANARCI_BIN = os.environ.get("ANARCI_BIN", "anarci")  # Default to 'anarci' in PATH

# Supported numbering schemes
SUPPORTED_SCHEMES = ["chothia", "kabat", "imgt", "martin", "aho"]

# Default settings
DEFAULT_SCHEME = "chothia"
DEFAULT_TIMEOUT = 300  # 5 minutes

# Temp directory for processing
TEMP_DIR = Path("/tmp")

# NOTE: CDR positions are NOT hardcoded here!
# They are extracted from ANARCI's output for each sequence.
# ANARCI identifies CDR regions based on the numbering scheme and sequence analysis.
