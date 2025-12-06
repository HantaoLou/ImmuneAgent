"""
Format Handler for AIRR Data

Handles conversion between JSON and AIRR TSV format.
AIRR format is IDENTICAL to IgBLAST output format.
"""

import csv
import logging
from typing import Dict, Any, List, Optional
from io import StringIO

logger = logging.getLogger(__name__)


class AIRRFormatHandler:
    """Handles AIRR format conversion and validation"""

    # Standard AIRR fields (subset most commonly used)
    # Full spec: https://docs.airr-community.org/en/stable/datarep/rearrangements.html
    AIRR_FIELDS = [
        "sequence_id",
        "sequence",
        "sequence_aa",
        "rev_comp",
        "productive",
        "v_call",
        "d_call",
        "j_call",
        "c_call",
        "sequence_alignment",
        "germline_alignment",
        "junction",
        "junction_aa",
        "junction_length",
        "junction_aa_length",
        "v_score",
        "v_identity",
        "v_support",
        "v_cigar",
        "d_score",
        "d_identity",
        "d_support",
        "d_cigar",
        "j_score",
        "j_identity",
        "j_support",
        "j_cigar",
        "v_sequence_start",
        "v_sequence_end",
        "v_germline_start",
        "v_germline_end",
        "d_sequence_start",
        "d_sequence_end",
        "d_germline_start",
        "d_germline_end",
        "j_sequence_start",
        "j_sequence_end",
        "j_germline_start",
        "j_germline_end",
        "np1_length",
        "np2_length",
        "cdr1",
        "cdr1_aa",
        "cdr2",
        "cdr2_aa",
        "cdr3",
        "cdr3_aa",
        "fwr1",
        "fwr1_aa",
        "fwr2",
        "fwr2_aa",
        "fwr3",
        "fwr3_aa",
        "fwr4",
        "fwr4_aa",
        "repertoire_id",
        "sample_processing_id",
        "data_processing_id",
        "clone_id",
        "cell_id"
    ]

    # Essential fields that should always be present
    ESSENTIAL_FIELDS = [
        "sequence_id",
        "sequence",
        "v_call",
        "d_call",
        "j_call",
        "junction",
        "junction_aa",
        "productive"
    ]

    def __init__(self, fields: Optional[List[str]] = None):
        """
        Initialize format handler

        Args:
            fields: Optional custom field list (defaults to AIRR_FIELDS)
        """
        self.fields = fields or self.AIRR_FIELDS

    def json_to_tsv_header(self) -> str:
        """
        Generate AIRR TSV header

        Returns:
            Tab-separated header string
        """
        return '\t'.join(self.fields)

    def json_to_tsv_record(self, record: Dict[str, Any]) -> str:
        """
        Convert single JSON record to AIRR TSV format

        Args:
            record: JSON record

        Returns:
            Tab-separated record string
        """
        values = []

        for field in self.fields:
            value = record.get(field, "")

            # Handle None values
            if value is None:
                value = ""

            # Convert booleans to string
            elif isinstance(value, bool):
                value = "T" if value else "F"

            # Convert everything else to string
            else:
                value = str(value)

            values.append(value)

        return '\t'.join(values)

    def json_to_tsv(self, records: List[Dict[str, Any]]) -> str:
        """
        Convert list of JSON records to complete AIRR TSV format

        Args:
            records: List of JSON records

        Returns:
            Complete TSV string with header
        """
        output = StringIO()

        # Write header
        output.write(self.json_to_tsv_header() + '\n')

        # Write records
        for record in records:
            output.write(self.json_to_tsv_record(record) + '\n')

        return output.getvalue()

    def tsv_to_json(self, tsv_content: str) -> List[Dict[str, Any]]:
        """
        Parse AIRR TSV format to JSON records

        Args:
            tsv_content: Complete TSV content with header

        Returns:
            List of JSON records
        """
        records = []

        try:
            reader = csv.DictReader(StringIO(tsv_content), delimiter='\t')

            for row in reader:
                # Convert string booleans back to bool
                record = {}
                for key, value in row.items():
                    if value == "T":
                        record[key] = True
                    elif value == "F":
                        record[key] = False
                    elif value == "":
                        record[key] = None
                    else:
                        record[key] = value

                records.append(record)

        except Exception as e:
            logger.error(f"Error parsing TSV: {e}")
            raise

        return records

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """
        Validate that record has essential fields

        Args:
            record: JSON record

        Returns:
            True if valid
        """
        for field in self.ESSENTIAL_FIELDS:
            if field not in record:
                logger.warning(f"Missing essential field: {field}")
                return False

        return True

    def filter_fields(
        self,
        records: List[Dict[str, Any]],
        fields: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter records to only include specified fields

        Args:
            records: List of records
            fields: Fields to keep

        Returns:
            Filtered records
        """
        filtered = []

        for record in records:
            filtered_record = {
                field: record.get(field)
                for field in fields
                if field in record
            }
            filtered.append(filtered_record)

        return filtered

    def get_available_fields(self, records: List[Dict[str, Any]]) -> List[str]:
        """
        Get list of all fields present in records

        Args:
            records: List of records

        Returns:
            List of unique field names
        """
        all_fields = set()

        for record in records:
            all_fields.update(record.keys())

        # Return in AIRR standard order where possible
        ordered_fields = []
        for field in self.AIRR_FIELDS:
            if field in all_fields:
                ordered_fields.append(field)
                all_fields.remove(field)

        # Add any additional fields not in standard list
        ordered_fields.extend(sorted(all_fields))

        return ordered_fields

    def standardize_field_names(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize field names to AIRR format

        Some repositories may use slightly different field names.
        This function normalizes them.

        Args:
            record: Input record

        Returns:
            Record with standardized field names
        """
        # Common field name mappings
        field_mappings = {
            "seq_id": "sequence_id",
            "seq": "sequence",
            "v_gene": "v_call",
            "d_gene": "d_call",
            "j_gene": "j_call",
            "c_gene": "c_call",
            "cdr3_nt": "junction",
            "cdr3_aa": "junction_aa",
            "cdr3": "junction",
            "is_productive": "productive"
        }

        standardized = {}

        for key, value in record.items():
            # Use mapping if available, otherwise keep original
            standard_key = field_mappings.get(key, key)
            standardized[standard_key] = value

        return standardized

    def create_summary_stats(
        self,
        records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create summary statistics from records

        Args:
            records: List of records

        Returns:
            Summary statistics
        """
        stats = {
            "total_sequences": len(records),
            "productive_sequences": 0,
            "v_genes": {},
            "j_genes": {},
            "junction_lengths": []
        }

        for record in records:
            # Count productive
            if record.get("productive") is True:
                stats["productive_sequences"] += 1

            # Count V genes
            v_call = record.get("v_call")
            if v_call:
                stats["v_genes"][v_call] = stats["v_genes"].get(v_call, 0) + 1

            # Count J genes
            j_call = record.get("j_call")
            if j_call:
                stats["j_genes"][j_call] = stats["j_genes"].get(j_call, 0) + 1

            # Collect junction lengths
            junction_length = record.get("junction_aa_length")
            if junction_length:
                stats["junction_lengths"].append(int(junction_length))

        # Calculate junction length stats
        if stats["junction_lengths"]:
            stats["junction_length_mean"] = sum(stats["junction_lengths"]) / len(stats["junction_lengths"])
            stats["junction_length_min"] = min(stats["junction_lengths"])
            stats["junction_length_max"] = max(stats["junction_lengths"])

        # Sort gene usage by frequency
        stats["v_genes"] = dict(sorted(
            stats["v_genes"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])  # Top 10

        stats["j_genes"] = dict(sorted(
            stats["j_genes"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])  # Top 10

        # Remove raw lengths list (can be large)
        del stats["junction_lengths"]

        return stats


class IgBLASTCompatibility:
    """Ensures compatibility with IgBLAST output format"""

    @staticmethod
    def is_compatible(airr_record: Dict[str, Any]) -> bool:
        """
        Check if AIRR record is compatible with IgBLAST format

        Args:
            airr_record: AIRR format record

        Returns:
            True if compatible
        """
        # IgBLAST output requires these fields
        required_fields = [
            "sequence_id",
            "v_call",
            "d_call",
            "j_call",
            "junction",
            "junction_aa"
        ]

        return all(field in airr_record for field in required_fields)

    @staticmethod
    def convert_to_igblast_format(airr_records: List[Dict[str, Any]]) -> str:
        """
        Convert AIRR records to IgBLAST-compatible format

        Since AIRR format is identical to IgBLAST output,
        this is just a passthrough with validation.

        Args:
            airr_records: List of AIRR records

        Returns:
            IgBLAST-compatible TSV format
        """
        handler = AIRRFormatHandler()
        return handler.json_to_tsv(airr_records)

    @staticmethod
    def merge_with_igblast(
        airr_records: List[Dict[str, Any]],
        igblast_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge AIRR Data Commons records with IgBLAST analysis results

        Args:
            airr_records: Records from AIRR Data Commons
            igblast_records: Records from IgBLAST analysis

        Returns:
            Combined records
        """
        # Since formats are identical, just concatenate
        return airr_records + igblast_records
