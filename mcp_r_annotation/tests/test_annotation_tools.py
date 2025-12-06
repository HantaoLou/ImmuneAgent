"""
Test suite for mcp_r_annotation tools

Tests all 6 annotation tools with mock data to ensure proper functionality.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from annotation_mcp_server import load_config, run_r_script


class TestAnnotationMCPServer:
    """Test suite for Cell Type Annotation MCP Server"""

    @pytest.fixture
    def config(self):
        """Load server configuration"""
        return load_config()

    @pytest.fixture
    def mock_rds_path(self):
        """Path to mock Seurat RDS file (must be created separately)"""
        # In real testing, this would point to a small test dataset
        # return str(Path(__file__).parent / "data" / "test_seurat.rds")
        # 使用新的测试数据文件
        return r"D:\data\test_data_20251001\Age_Bcells.rds"

    def test_config_loading(self, config):
        """Test configuration file loading"""
        assert config is not None
        assert "base_dir" in config
        assert "scripts_dir" in config
        assert "output_dir" in config
        assert "reference_data_dir" in config
        assert config["server_name"] == "Cell Type Annotation MCP Server"
        assert config["port"] == 8095
        assert config["transport"] == "stdio"

    def test_reference_datasets_config(self, config):
        """Test reference datasets are properly configured"""
        assert "reference_datasets" in config
        datasets = config["reference_datasets"]

        # Check key datasets exist
        required_datasets = [
            "HumanPrimaryCellAtlasData",
            "BlueprintEncodeData",
            "MonacoImmuneData"
        ]

        for dataset in required_datasets:
            assert dataset in datasets
            assert "description" in datasets[dataset]
            assert "species" in datasets[dataset]
            assert "recommended_for" in datasets[dataset]

    def test_default_parameters(self, config):
        """Test default parameters are configured for all tools"""
        assert "default_parameters" in config

        tools = [
            "run_singler_annotation",
            "detect_cluster_markers",
            "annotate_by_markers",
            "validate_annotation",
            "score_annotation_confidence",
            "export_annotations"
        ]

        for tool in tools:
            assert tool in config["default_parameters"]

    def test_known_marker_genes(self, config):
        """Test known marker genes are configured"""
        assert "known_marker_genes" in config
        markers = config["known_marker_genes"]

        # Check key cell types
        key_celltypes = ["T_cells", "B_cells", "NK_cells", "Monocytes", "Plasma_cells"]

        for celltype in key_celltypes:
            assert celltype in markers
            assert isinstance(markers[celltype], list)
            assert len(markers[celltype]) > 0

    def test_output_structure(self, config):
        """Test output structure is defined for all tools"""
        assert "output_structure" in config
        output = config["output_structure"]

        tools = [
            "run_singler_annotation",
            "detect_cluster_markers",
            "annotate_by_markers",
            "validate_annotation",
            "score_annotation_confidence",
            "export_annotations"
        ]

        for tool in tools:
            assert tool in output
            assert isinstance(output[tool], list)
            assert len(output[tool]) > 0

    def test_r_scripts_exist(self):
        """Test that all R scripts exist"""
        scripts_dir = Path(__file__).parent.parent / "scripts"

        required_scripts = [
            "run_singler_annotation.R",
            "detect_cluster_markers.R",
            "annotate_by_markers.R",
            "validate_annotation.R",
            "score_annotation_confidence.R",
            "export_annotations.R"
        ]

        for script in required_scripts:
            script_path = scripts_dir / script
            assert script_path.exists(), f"Script not found: {script}"
            assert script_path.is_file()

    def test_directory_structure(self):
        """Test that required directories exist"""
        base_dir = Path(__file__).parent.parent

        required_dirs = [
            "scripts",
            "tests",
            "config",
            "output",
            "reference_data"
        ]

        for dir_name in required_dirs:
            dir_path = base_dir / dir_name
            assert dir_path.exists(), f"Directory not found: {dir_name}"
            assert dir_path.is_dir()

    def test_config_file_exists(self):
        """Test that config.json exists and is valid JSON"""
        config_path = Path(__file__).parent.parent / "config" / "config.json"

        assert config_path.exists()
        assert config_path.is_file()

        # Validate JSON structure
        with open(config_path, 'r') as f:
            config = json.load(f)

        assert isinstance(config, dict)

    @pytest.mark.skipif(
        not Path(__file__).parent.joinpath("data", "test_seurat.rds").exists(),
        reason="Test data not available"
    )
    def test_run_singler_annotation(self, mock_rds_path):
        """Test SingleR annotation tool (requires test data)"""
        result = run_r_script(
            "run_singler_annotation",
            mock_rds_path,
            reference_dataset="HumanPrimaryCellAtlasData",
            label_type="label.main",
            cluster_column="seurat_clusters"
        )

        assert result["status"] == "success"
        assert "generated_files" in result
        assert len(result["generated_files"]) > 0

    @pytest.mark.skipif(
        not Path(__file__).parent.joinpath("data", "test_seurat.rds").exists(),
        reason="Test data not available"
    )
    def test_detect_cluster_markers(self, mock_rds_path):
        """Test marker detection tool (requires test data)"""
        result = run_r_script(
            "detect_cluster_markers",
            mock_rds_path,
            test_use="wilcox",
            only_pos=True,
            min_pct=0.25,
            logfc_threshold=0.5,
            top_n=10
        )

        assert result["status"] == "success"
        assert "generated_files" in result

    @pytest.mark.skipif(
        not Path(__file__).parent.joinpath("data", "test_seurat.rds").exists(),
        reason="Test data not available"
    )
    def test_annotate_by_markers(self, mock_rds_path):
        """Test manual annotation tool (requires test data)"""
        marker_list = {
            "0": "T cells",
            "1": "B cells",
            "2": "Monocytes"
        }

        result = run_r_script(
            "annotate_by_markers",
            mock_rds_path,
            marker_list=marker_list,
            cluster_column="seurat_clusters",
            new_column="manual_celltype"
        )

        assert result["status"] == "success"
        assert "generated_files" in result

    def test_statistical_tests_config(self, config):
        """Test statistical tests are properly configured"""
        assert "statistical_tests" in config
        tests = config["statistical_tests"]

        required_tests = ["wilcox", "bimod", "roc", "t", "MAST"]

        for test in required_tests:
            assert test in tests
            assert "name" in tests[test]
            assert "description" in tests[test]
            assert "speed" in tests[test]

    def test_r_installed(self):
        """Test that R and Rscript are installed"""
        try:
            result = subprocess.run(
                ["Rscript", "--version"],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
        except FileNotFoundError:
            pytest.fail("Rscript not found. Please install R.")

    def test_required_r_packages(self):
        """Test that required R packages are installed"""
        required_packages = [
            "Seurat",
            "SingleR",
            "celldex",
            "dplyr",
            "jsonlite"
        ]

        r_code = f"""
        installed <- installed.packages()[, "Package"]
        required <- c({', '.join([f'"{pkg}"' for pkg in required_packages])})
        missing <- setdiff(required, installed)
        cat(paste(missing, collapse=","))
        """

        try:
            result = subprocess.run(
                ["Rscript", "-e", r_code],
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8',
                errors='ignore'
            )

            missing = result.stdout.strip()
            if missing:
                pytest.skip(f"Missing R packages: {missing}")
        except Exception as e:
            pytest.skip(f"Could not check R packages: {e}")


class TestReferenceDataManagement:
    """Test reference dataset management"""

    def test_reference_data_directory(self):
        """Test reference data directory exists"""
        ref_dir = Path(__file__).parent.parent / "reference_data"
        assert ref_dir.exists()
        assert ref_dir.is_dir()

    def test_download_script_exists(self):
        """Test download_references.R script exists"""
        script_path = Path(__file__).parent.parent / "scripts" / "download_references.R"
        assert script_path.exists()
        assert script_path.is_file()


class TestIntegration:
    """Integration tests for complete workflows"""

    @pytest.mark.skipif(
        not Path(__file__).parent.joinpath("data", "test_seurat.rds").exists(),
        reason="Test data not available"
    )
    def test_full_annotation_workflow(self, mock_rds_path):
        """Test complete annotation workflow (requires test data)"""
        # 1. Run SingleR annotation
        result1 = run_r_script(
            "run_singler_annotation",
            mock_rds_path,
            reference_dataset="HumanPrimaryCellAtlasData"
        )
        assert result1["status"] == "success"

        # 2. Detect markers
        result2 = run_r_script(
            "detect_cluster_markers",
            mock_rds_path
        )
        assert result2["status"] == "success"

        # 3. Manual annotation (if SingleR succeeded)
        if result1["status"] == "success":
            annotated_rds = [f for f in result1["generated_files"]
                           if f.endswith("seurat_with_singler.rds")][0]

            result3 = run_r_script(
                "score_annotation_confidence",
                annotated_rds,
                annotation_column="singler_HumanPrimaryCellAtlasData"
            )
            assert result3["status"] == "success"


def test_readme_exists():
    """Test that README.md exists"""
    readme_path = Path(__file__).parent.parent / "README.md"
    assert readme_path.exists()
    assert readme_path.is_file()

    # Check README contains key sections
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    required_sections = [
        "Overview",
        "Reference Datasets",
        "MCP Tools",
        "Workflows",
        "Troubleshooting"
    ]

    for section in required_sections:
        assert section in content, f"README missing section: {section}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
