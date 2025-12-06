"""
SAbDab Core Functions (without MCP decorators)

Downloads data from Structural Antibody Database.
Focus: Download CSV, PDB structures, and database dumps.

This module contains the actual implementation functions that can be called
directly from Python scripts or wrapped by the MCP server.
"""

import requests
from typing import Dict, Any, Optional
import time
from .file_manager import file_manager

BASE_URL = "http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred"


def download_sabdab_summary_csv(
    filters: Optional[Dict[str, str]] = None,
    save_file: bool = True
) -> Dict[str, Any]:
    """
    Download SAbDab summary data as CSV.

    Args:
        filters: Optional filters like {"resolution": "<2.5", "antigen": "yes"}
        save_file: Whether to save the file to disk (default: True)

    Returns:
        {
            "status": "success",
            "csv_content": "pdb,Hchain,Lchain,...",
            "num_entries": 1234,
            "file_size_bytes": 56789,
            "file_info": {
                "file_path": "/path/to/file.csv",
                "file_size_bytes": 56789,
                "created_at": "2024-12-01T14:30:22Z"
            }
        }
    """
    try:
        # SAbDab summary download endpoint
        url = f"{BASE_URL}/sabdab/summary/all/"

        params = {}
        if filters:
            params.update(filters)

        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        csv_content = response.text
        num_lines = len(csv_content.split('\n'))

        result = {
            "status": "success",
            "csv_content": csv_content,
            "num_entries": num_lines - 1,  # Minus header
            "file_size_bytes": len(csv_content)
        }

        # 保存文件（如果启用）
        if save_file:
            file_info = file_manager.save_csv_file(csv_content, "sabdab_summary")
            result["file_info"] = file_info

        return result

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


def download_pdb_structure(
    pdb_id: str,
    numbering_scheme: str = "chothia",
    save_file: bool = True
) -> Dict[str, Any]:
    """
    Download PDB structure from SAbDab with specified numbering.

    Args:
        pdb_id: PDB ID (e.g., "6m0j")
        numbering_scheme: chothia, kabat, or imgt
        save_file: Whether to save the file to disk (default: True)

    Returns:
        {
            "status": "success",
            "pdb_id": "6m0j",
            "pdb_content": "ATOM   1  N ...",
            "numbering_scheme": "chothia",
            "file_size_bytes": 123456,
            "file_info": {
                "file_path": "/path/to/file.pdb",
                "file_size_bytes": 123456,
                "created_at": "2024-12-01T14:30:22Z"
            }
        }
    """
    try:
        url = f"{BASE_URL}/sabdab/pdb/{pdb_id.lower()}/"
        params = {"scheme": numbering_scheme}

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        result = {
            "status": "success",
            "pdb_id": pdb_id,
            "pdb_content": response.text,
            "numbering_scheme": numbering_scheme,
            "file_size_bytes": len(response.text)
        }

        # 保存文件（如果启用）
        if save_file:
            file_info = file_manager.save_pdb_file(response.text, pdb_id, numbering_scheme)
            result["file_info"] = file_info

        return result

    except Exception as e:
        return {
            "status": "error",
            "pdb_id": pdb_id,
            "message": str(e)
        }


def download_sabdab_dataset(
    dataset_type: str = "all",
    output_format: str = "csv",
    save_file: bool = True
) -> Dict[str, Any]:
    """
    Download complete SAbDab datasets.

    Args:
        dataset_type: all, antigen_bound, nanobodies, etc.
        output_format: csv, json, or fasta
        save_file: Whether to save the file to disk (default: True)

    Returns:
        Dataset content and metadata with optional file info
    """
    try:
        # Construct download URL based on dataset type
        if dataset_type == "all":
            url = f"{BASE_URL}/sabdab/summary/all/"
        elif dataset_type == "antigen_bound":
            url = f"{BASE_URL}/sabdab/summary/all/?antigen=yes"
        elif dataset_type == "nanobodies":
            url = f"{BASE_URL}/sabdab/summary/all/?nanobody=yes"
        else:
            url = f"{BASE_URL}/sabdab/summary/all/"

        response = requests.get(url, timeout=120)
        response.raise_for_status()

        result = {
            "status": "success",
            "dataset_type": dataset_type,
            "format": output_format,
            "content": response.text,
            "size_bytes": len(response.text)
        }

        # 保存文件（如果启用）
        if save_file:
            file_info = file_manager.save_dataset_file(response.text, dataset_type, output_format)
            result["file_info"] = file_info

        return result

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


def get_sabdab_statistics(save_file: bool = True) -> Dict[str, Any]:
    """
    Get SAbDab database statistics.

    Args:
        save_file: Whether to save the statistics to a JSON file (default: True)

    Returns:
        Database statistics and metadata with optional file info
    """
    try:
        url = f"{BASE_URL}/sabdab/summary/all/"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Count entries
        lines = response.text.split('\n')
        total_structures = len(lines) - 1  # Minus header

        result = {
            "status": "success",
            "total_structures": total_structures,
            "last_updated": "Check SAbDab website for update info",
            "database_url": BASE_URL
        }

        # 保存文件（如果启用）
        if save_file:
            import json
            stats_json = json.dumps(result, indent=2)
            file_info = file_manager.save_json_file(stats_json, "sabdab_statistics")
            result["file_info"] = file_info

        return result

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
