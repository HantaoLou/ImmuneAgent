"""
Utility functions for ANARCI MCP Server
"""

import uuid
from pathlib import Path
from typing import List, Dict
from config import TEMP_DIR


def write_fasta(sequences: List[Dict[str, str]], session_id: str) -> Path:
    """
    Write sequences to a temporary FASTA file.

    Args:
        sequences: List of {"id": "seq1", "sequence": "EVQL..."}
        session_id: Unique session identifier

    Returns:
        Path to the created FASTA file
    """
    fasta_file = TEMP_DIR / f"anarci_input_{session_id}.fasta"

    with open(fasta_file, 'w') as f:
        for seq in sequences:
            seq_id = seq.get('id', 'unknown')
            sequence = seq.get('sequence', '')
            f.write(f">{seq_id}\n{sequence}\n")

    return fasta_file


def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())[:8]


def cleanup_files(*file_paths: Path) -> None:
    """
    Clean up temporary files.

    Args:
        *file_paths: Variable number of Path objects to delete
    """
    for file_path in file_paths:
        if file_path and Path(file_path).exists():
            Path(file_path).unlink(missing_ok=True)
