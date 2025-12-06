"""
Basic integration tests for AIRR MCP Server

Tests basic functionality with mocked API responses.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.repositories import RepositoryManager
from src.query_builder import QueryBuilder
from src.pagination import PaginationHandler
from src.cache import CacheManager
from src.format_handler import AIRRFormatHandler, IgBLASTCompatibility


class TestRepositoryManager:
    """Test repository management"""

    def test_get_repository_url(self):
        """Test URL construction"""
        repo = RepositoryManager()
        url = repo.get_repository_url("vdjserver", "repertoire")
        assert url == "https://vdjserver.org/airr/v1/repertoire"

    def test_invalid_repository(self):
        """Test invalid repository handling"""
        repo = RepositoryManager()
        with pytest.raises(ValueError):
            repo.get_repository_url("invalid_repo", "repertoire")

    def test_get_repository_info(self):
        """Test repository info retrieval"""
        repo = RepositoryManager()
        info = repo.get_repository_info()
        assert "repositories" in info
        assert "vdjserver" in info["repositories"]
        assert info["total_count"] >= 3

    @patch('requests.Session.post')
    def test_query_single_success(self, mock_post):
        """Test successful single repository query"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"Repertoire": [{"repertoire_id": "test123"}]}
        mock_post.return_value = mock_response

        repo = RepositoryManager()
        result = repo.query_single("vdjserver", "repertoire", {})

        assert "Repertoire" in result
        assert len(result["Repertoire"]) == 1

    @patch('requests.Session.post')
    def test_query_single_timeout(self, mock_post):
        """Test timeout handling"""
        mock_post.side_effect = Exception("Timeout")

        repo = RepositoryManager()
        result = repo.query_single("vdjserver", "repertoire", {})

        assert "error" in result


class TestQueryBuilder:
    """Test query construction"""

    def test_species_to_ncbi(self):
        """Test species conversion"""
        qb = QueryBuilder()
        assert qb.species_to_ncbi("human") == "NCBITAXON:9606"
        assert qb.species_to_ncbi("mouse") == "NCBITAXON:10090"

    def test_build_filter(self):
        """Test filter construction"""
        qb = QueryBuilder()
        filter_obj = qb.build_filter("subject.species.id", "NCBITAXON:9606", "=")

        assert filter_obj["op"] == "="
        assert filter_obj["content"]["field"] == "subject.species.id"
        assert filter_obj["content"]["value"] == "NCBITAXON:9606"

    def test_combine_filters_single(self):
        """Test combining single filter"""
        qb = QueryBuilder()
        filters = [qb.build_filter("field1", "value1")]
        result = qb.combine_filters(filters)

        assert result["op"] == "="

    def test_combine_filters_multiple(self):
        """Test combining multiple filters"""
        qb = QueryBuilder()
        filters = [
            qb.build_filter("field1", "value1"),
            qb.build_filter("field2", "value2")
        ]
        result = qb.combine_filters(filters, "and")

        assert result["op"] == "and"
        assert len(result["content"]) == 2

    def test_build_repertoire_query(self):
        """Test repertoire query construction"""
        qb = QueryBuilder()
        query = qb.build_repertoire_query(
            disease="COVID-19",
            species="human",
            from_index=0,
            size=100
        )

        assert query["from"] == 0
        assert query["size"] == 100
        assert "filters" in query

    def test_build_rearrangement_query(self):
        """Test rearrangement query construction"""
        qb = QueryBuilder()
        query = qb.build_rearrangement_query(
            repertoire_id="test123",
            v_call="IGHV3-23",
            productive=True,
            size=1000
        )

        assert query["size"] == 1000
        assert query["format"] == "json"
        assert "filters" in query

    def test_validate_query_valid(self):
        """Test query validation - valid"""
        qb = QueryBuilder()
        query = {"from": 0, "size": 100}
        assert qb.validate_query(query) is True

    def test_validate_query_invalid(self):
        """Test query validation - invalid"""
        qb = QueryBuilder()
        query = {"size": 100}  # Missing 'from'
        assert qb.validate_query(query) is False


class TestPaginationHandler:
    """Test pagination"""

    def test_pagination_basic(self):
        """Test basic pagination"""
        # Mock fetch function
        def mock_fetch(from_idx, size):
            if from_idx >= 100:
                return {"Rearrangement": []}
            return {"Rearrangement": [{"id": i} for i in range(from_idx, min(from_idx + size, 100))]}

        ph = PaginationHandler(page_size=25)
        all_results = []

        for page in ph.paginate_results(mock_fetch, max_records=100):
            all_results.extend(page)

        assert len(all_results) == 100

    def test_pagination_max_records(self):
        """Test pagination with max records limit"""
        def mock_fetch(from_idx, size):
            return {"Rearrangement": [{"id": i} for i in range(from_idx, from_idx + size)]}

        ph = PaginationHandler(page_size=25)
        all_results = ph.collect_all_results(mock_fetch, max_records=50)

        assert len(all_results) == 50

    def test_extract_results_rearrangement(self):
        """Test extracting rearrangement results"""
        ph = PaginationHandler()
        response = {"Rearrangement": [{"id": 1}, {"id": 2}]}
        results = ph._extract_results(response)

        assert len(results) == 2

    def test_extract_results_repertoire(self):
        """Test extracting repertoire results"""
        ph = PaginationHandler()
        response = {"Repertoire": [{"id": 1}]}
        results = ph._extract_results(response)

        assert len(results) == 1


class TestCacheManager:
    """Test caching"""

    def test_cache_set_get(self, tmp_path):
        """Test setting and getting cache"""
        cm = CacheManager(cache_dir=str(tmp_path), ttl=3600)
        test_data = {"key": "value", "number": 42}

        # Set cache
        success = cm.set("studies", "test_id", test_data)
        assert success is True

        # Get cache
        cached = cm.get("studies", "test_id")
        assert cached is not None
        assert cached["key"] == "value"
        assert cached["number"] == 42

    def test_cache_disabled(self):
        """Test cache when disabled"""
        cm = CacheManager(enabled=False)
        cm.set("studies", "test_id", {"data": "test"})
        result = cm.get("studies", "test_id")

        assert result is None

    def test_cache_statistics(self, tmp_path):
        """Test cache statistics"""
        cm = CacheManager(cache_dir=str(tmp_path))
        cm.set("studies", "test1", {"data": "test1"})
        cm.set("studies", "test2", {"data": "test2"})

        stats = cm.get_statistics()

        assert stats["enabled"] is True
        assert stats["total_entries"] >= 2

    def test_cache_invalidate(self, tmp_path):
        """Test cache invalidation"""
        cm = CacheManager(cache_dir=str(tmp_path))
        cm.set("studies", "test_id", {"data": "test"})

        # Invalidate
        count = cm.invalidate("studies", "test_id")
        assert count == 1

        # Check it's gone
        cached = cm.get("studies", "test_id")
        assert cached is None


class TestAIRRFormatHandler:
    """Test AIRR format handling"""

    def test_json_to_tsv_header(self):
        """Test TSV header generation"""
        fh = AIRRFormatHandler()
        header = fh.json_to_tsv_header()

        assert "sequence_id" in header
        assert "v_call" in header
        assert "\t" in header

    def test_json_to_tsv_record(self):
        """Test single record conversion"""
        fh = AIRRFormatHandler()
        record = {
            "sequence_id": "seq1",
            "sequence": "ATGC",
            "v_call": "IGHV3-23",
            "productive": True
        }

        tsv = fh.json_to_tsv_record(record)

        assert "seq1" in tsv
        assert "ATGC" in tsv
        assert "IGHV3-23" in tsv
        assert "T" in tsv  # Boolean converted

    def test_json_to_tsv_complete(self):
        """Test complete JSON to TSV conversion"""
        fh = AIRRFormatHandler()
        records = [
            {"sequence_id": "seq1", "v_call": "IGHV3-23"},
            {"sequence_id": "seq2", "v_call": "IGHV1-69"}
        ]

        tsv = fh.json_to_tsv(records)

        lines = tsv.strip().split('\n')
        assert len(lines) == 3  # Header + 2 records

    def test_tsv_to_json(self):
        """Test TSV to JSON parsing"""
        fh = AIRRFormatHandler()
        tsv = "sequence_id\tv_call\tproductive\nseq1\tIGHV3-23\tT\nseq2\tIGHV1-69\tF"

        records = fh.tsv_to_json(tsv)

        assert len(records) == 2
        assert records[0]["sequence_id"] == "seq1"
        assert records[0]["productive"] is True
        assert records[1]["productive"] is False

    def test_validate_record_valid(self):
        """Test record validation - valid"""
        fh = AIRRFormatHandler()
        record = {
            "sequence_id": "seq1",
            "sequence": "ATGC",
            "v_call": "IGHV3-23",
            "d_call": "IGHD3-10",
            "j_call": "IGHJ4",
            "junction": "TGTGCG",
            "junction_aa": "CAR",
            "productive": True
        }

        assert fh.validate_record(record) is True

    def test_validate_record_invalid(self):
        """Test record validation - invalid"""
        fh = AIRRFormatHandler()
        record = {"sequence_id": "seq1"}  # Missing essential fields

        assert fh.validate_record(record) is False

    def test_create_summary_stats(self):
        """Test summary statistics creation"""
        fh = AIRRFormatHandler()
        records = [
            {"productive": True, "v_call": "IGHV3-23", "j_call": "IGHJ4", "junction_aa_length": 15},
            {"productive": True, "v_call": "IGHV3-23", "j_call": "IGHJ4", "junction_aa_length": 18},
            {"productive": False, "v_call": "IGHV1-69", "j_call": "IGHJ6", "junction_aa_length": 12}
        ]

        stats = fh.create_summary_stats(records)

        assert stats["total_sequences"] == 3
        assert stats["productive_sequences"] == 2
        assert "IGHV3-23" in stats["v_genes"]
        assert stats["v_genes"]["IGHV3-23"] == 2


class TestIgBLASTCompatibility:
    """Test IgBLAST compatibility"""

    def test_is_compatible_valid(self):
        """Test compatibility check - valid record"""
        record = {
            "sequence_id": "seq1",
            "v_call": "IGHV3-23",
            "d_call": "IGHD3-10",
            "j_call": "IGHJ4",
            "junction": "TGTGCG",
            "junction_aa": "CAR"
        }

        assert IgBLASTCompatibility.is_compatible(record) is True

    def test_is_compatible_invalid(self):
        """Test compatibility check - invalid record"""
        record = {"sequence_id": "seq1"}

        assert IgBLASTCompatibility.is_compatible(record) is False

    def test_convert_to_igblast_format(self):
        """Test conversion to IgBLAST format"""
        records = [
            {
                "sequence_id": "seq1",
                "sequence": "ATGC",
                "v_call": "IGHV3-23",
                "d_call": "IGHD3-10",
                "j_call": "IGHJ4",
                "junction": "TGTGCG",
                "junction_aa": "CAR",
                "productive": True
            }
        ]

        result = IgBLASTCompatibility.convert_to_igblast_format(records)

        assert isinstance(result, str)
        assert "sequence_id" in result
        assert "seq1" in result

    def test_merge_with_igblast(self):
        """Test merging AIRR and IgBLAST results"""
        airr_records = [{"sequence_id": "seq1", "v_call": "IGHV3-23"}]
        igblast_records = [{"sequence_id": "seq2", "v_call": "IGHV1-69"}]

        merged = IgBLASTCompatibility.merge_with_igblast(airr_records, igblast_records)

        assert len(merged) == 2
        assert merged[0]["sequence_id"] == "seq1"
        assert merged[1]["sequence_id"] == "seq2"


# Integration test fixtures
@pytest.fixture
def mock_repertoire_response():
    """Mock repertoire API response"""
    return {
        "Repertoire": [
            {
                "repertoire_id": "test123",
                "study": {
                    "study_id": "PRJNA123",
                    "study_title": "Test Study"
                },
                "subject": {
                    "subject_id": "S1",
                    "species": {"id": "NCBITAXON:9606"},
                    "diagnosis": [{"study_group_description": "COVID-19"}]
                },
                "sample": [
                    {
                        "sample_id": "SA1",
                        "tissue": "PBMC",
                        "cell_subset": "B cells"
                    }
                ]
            }
        ]
    }


@pytest.fixture
def mock_rearrangement_response():
    """Mock rearrangement API response"""
    return {
        "Rearrangement": [
            {
                "sequence_id": "seq1",
                "sequence": "ATGCATGC",
                "v_call": "IGHV3-23*01",
                "d_call": "IGHD3-10*01",
                "j_call": "IGHJ4*02",
                "junction": "TGTGCGAGA",
                "junction_aa": "CARGLVVV",
                "productive": True,
                "v_identity": 0.97
            }
        ]
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
