"""
Query Builder for AIRR API

Constructs AIRR-compliant JSON queries following the API specification.
"""

from typing import Dict, Any, Optional, List
import logging

# 配置日志记录器
logger = logging.getLogger(__name__)


class QueryBuilder:
    """Constructs AIRR-compliant JSON queries"""

    # Species to NCBI Taxonomy ID mapping
    SPECIES_MAP = {
        "human": "NCBITAXON:9606",
        "mouse": "NCBITAXON:10090",
        "rat": "NCBITAXON:10116",
        "rabbit": "NCBITAXON:9986",
        "rhesus": "NCBITAXON:9544",
        "pig": "NCBITAXON:9823"
    }

    def __init__(self):
        pass

    def species_to_ncbi(self, species: str) -> str:
        """
        Convert species common name to NCBI taxonomy ID

        Args:
            species: Common name (human, mouse, etc.)

        Returns:
            NCBI taxonomy ID
        """
        return self.SPECIES_MAP.get(species.lower(), species)

    def build_filter(
        self,
        field: str,
        value: Any,
        operator: str = "="
    ) -> Dict[str, Any]:
        """
        Build a single filter clause

        Args:
            field: AIRR field name (e.g., "subject.species.id")
            value: Filter value
            operator: Filter operator (=, !=, <, >, contains, in)

        Returns:
            Filter clause
        """
        return {
            "op": operator,
            "content": {
                "field": field,
                "value": value
            }
        }

    def combine_filters(
        self,
        filters: List[Dict[str, Any]],
        logic: str = "and"
    ) -> Dict[str, Any]:
        """
        Combine multiple filters with AND or OR logic

        Args:
            filters: List of filter clauses
            logic: "and" or "or"

        Returns:
            Combined filter
        """
        if len(filters) == 0:
            return {}
        elif len(filters) == 1:
            return filters[0]
        else:
            return {
                "op": logic,
                "content": filters
            }

    def build_repertoire_query(
        self,
        disease: Optional[str] = None,
        tissue: Optional[str] = None,
        species: str = "human",
        cell_subset: Optional[str] = None,
        study_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        from_index: int = 0,
        size: int = 100
    ) -> Dict[str, Any]:
        """
        Build repertoire search query

        Args:
            disease: Disease or condition
            tissue: Tissue type
            species: Species (human, mouse, etc.)
            cell_subset: B cell subset
            study_id: Study identifier
            subject_id: Subject identifier
            from_index: Starting index for pagination
            size: Number of results to return

        Returns:
            Complete query object
        """
        filters = []

        # Species filter
        if species:
            filters.append(self.build_filter(
                "subject.species.id",
                self.species_to_ncbi(species),
                "="
            ))

        # Disease filter
        if disease:
            filters.append(self.build_filter(
                "subject.diagnosis.study_group_description",
                disease,
                "contains"
            ))

        # Tissue filter
        if tissue:
            filters.append(self.build_filter(
                "sample.tissue",
                tissue,
                "contains"
            ))

        # Cell subset filter
        if cell_subset:
            filters.append(self.build_filter(
                "sample.cell_subset",
                cell_subset,
                "contains"
            ))

        # Study ID filter
        if study_id:
            filters.append(self.build_filter(
                "study.study_id",
                study_id,
                "="
            ))

        # Subject ID filter
        if subject_id:
            filters.append(self.build_filter(
                "subject.subject_id",
                subject_id,
                "="
            ))

        query = {
            "from": from_index,
            "size": size
        }

        # 只有当有过滤器时才添加filters字段
        if filters:
            query["filters"] = self.combine_filters(filters, "and")
        else:
            # 如果没有过滤器，添加一个空的filters对象以符合API要求
            # 注意：某些API可能需要一个空的filters对象，而其他API可能不需要
            # 这里我们不添加空的filters对象，让repositories.py中的代码处理这种情况
            pass

        # 打印最终的查询对象以进行调试
        logger.info(f"Built repertoire query: {query}")

        return query

    def build_rearrangement_query(
        self,
        repertoire_id: Optional[str] = None,
        v_call: Optional[str] = None,
        d_call: Optional[str] = None,
        j_call: Optional[str] = None,
        junction_aa_length: Optional[int] = None,
        productive: Optional[bool] = None,
        from_index: int = 0,
        size: int = 1000,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build rearrangement (sequence) query

        Args:
            repertoire_id: Repertoire identifier
            v_call: V gene filter (e.g., "IGHV3-23")
            d_call: D gene filter
            j_call: J gene filter
            junction_aa_length: CDR3 amino acid length
            productive: Only productive sequences
            from_index: Starting index for pagination
            size: Number of results to return
            additional_filters: Additional custom filters

        Returns:
            Complete query object
        """
        filters = []

        # Repertoire ID filter (required for most queries)
        if repertoire_id:
            filters.append(self.build_filter(
                "repertoire_id",
                repertoire_id,
                "="
            ))

        # V gene filter
        if v_call:
            filters.append(self.build_filter(
                "v_call",
                v_call,
                "contains"
            ))

        # D gene filter
        if d_call:
            filters.append(self.build_filter(
                "d_call",
                d_call,
                "contains"
            ))

        # J gene filter
        if j_call:
            filters.append(self.build_filter(
                "j_call",
                j_call,
                "contains"
            ))

        # Junction length filter
        if junction_aa_length is not None:
            filters.append(self.build_filter(
                "junction_aa_length",
                junction_aa_length,
                "="
            ))

        # Productive filter
        if productive is not None:
            filters.append(self.build_filter(
                "productive",
                productive,
                "="
            ))

        # Additional custom filters
        if additional_filters:
            filters.append(additional_filters)

        query = {
            "from": from_index,
            "size": size,
            "format": "json"
        }

        if filters:
            query["filters"] = self.combine_filters(filters, "and")

        return query

    def build_gene_usage_query(
        self,
        repertoire_id: str,
        v_gene: Optional[str] = None,
        d_gene: Optional[str] = None,
        j_gene: Optional[str] = None,
        combination_logic: str = "and"
    ) -> Dict[str, Any]:
        """
        Build query for V/D/J gene usage patterns

        Args:
            repertoire_id: Repertoire identifier
            v_gene: V gene family or allele
            d_gene: D gene family or allele
            j_gene: J gene family or allele
            combination_logic: "and" or "or"

        Returns:
            Query object for gene filtering
        """
        filters = [
            self.build_filter("repertoire_id", repertoire_id, "=")
        ]

        gene_filters = []

        if v_gene:
            gene_filters.append(self.build_filter("v_call", v_gene, "contains"))

        if d_gene:
            gene_filters.append(self.build_filter("d_call", d_gene, "contains"))

        if j_gene:
            gene_filters.append(self.build_filter("j_call", j_gene, "contains"))

        if gene_filters:
            combined_gene_filter = self.combine_filters(gene_filters, combination_logic)
            filters.append(combined_gene_filter)

        return {
            "filters": self.combine_filters(filters, "and"),
            "from": 0,
            "size": 1000,
            "format": "json"
        }

    def build_facets_query(
        self,
        repertoire_id: str,
        facets: List[str]
    ) -> Dict[str, Any]:
        """
        Build query to get faceted statistics

        Args:
            repertoire_id: Repertoire identifier
            facets: List of fields to facet on (e.g., ["v_call", "j_call"])

        Returns:
            Query with facets
        """
        return {
            "filters": self.build_filter("repertoire_id", repertoire_id, "="),
            "facets": facets,
            "from": 0,
            "size": 0  # We only want facet counts, not actual sequences
        }

    def validate_query(self, query: Dict[str, Any]) -> bool:
        """
        Validate query structure

        Args:
            query: Query object to validate

        Returns:
            True if valid, False otherwise
        """
        # Basic validation - ensure required fields exist
        if not isinstance(query, dict):
            return False

        # Must have 'from' and 'size' for pagination
        if "from" not in query or "size" not in query:
            return False

        # If filters exist, validate structure
        if "filters" in query:
            return self._validate_filter(query["filters"])

        return True

    def _validate_filter(self, filter_obj: Dict[str, Any]) -> bool:
        """
        Recursively validate filter structure

        Args:
            filter_obj: Filter object to validate

        Returns:
            True if valid
        """
        if not isinstance(filter_obj, dict):
            return False

        if "op" not in filter_obj:
            return False

        valid_ops = ["=", "!=", "<", ">", "<=", ">=", "contains", "in", "and", "or"]
        if filter_obj["op"] not in valid_ops:
            return False

        if "content" not in filter_obj:
            return False

        # For logical operators (and, or), content should be a list
        if filter_obj["op"] in ["and", "or"]:
            if not isinstance(filter_obj["content"], list):
                return False
            return all(self._validate_filter(f) for f in filter_obj["content"])

        # For comparison operators, content should have field and value
        if isinstance(filter_obj["content"], dict):
            return "field" in filter_obj["content"] and "value" in filter_obj["content"]

        return False
