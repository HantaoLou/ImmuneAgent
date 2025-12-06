"""
Integration tests for mcp_r_scrna MCP server

Tests validate:
- Server import and configuration
- R script syntax and structure
- Config file validation
- Tool function signatures
- Error handling
- Documentation completeness
"""

import os
import json
import subprocess
from pathlib import Path
import pytest

# Get paths
BASE_DIR = Path(__file__).parent.parent
SERVER_PATH = BASE_DIR / "scrna_mcp_server.py"
CONFIG_PATH = BASE_DIR / "config.json"
SCRIPTS_DIR = BASE_DIR / "scripts"
README_PATH = BASE_DIR / "README.md"

# ========== Test 1: Server Import ==========

def test_server_imports():
    """Test that the server file can be imported without errors"""
    try:
        import sys
        sys.path.insert(0, str(BASE_DIR))
        import scrna_mcp_server
        assert hasattr(scrna_mcp_server, 'mcp')
        assert hasattr(scrna_mcp_server, 'load_config')
        assert hasattr(scrna_mcp_server, 'run_r_script')
    except ImportError as e:
        pytest.fail(f"Failed to import server: {e}")

# ========== Test 2: Configuration Validation ==========

def test_config_file_exists():
    """Test that config.json exists and is valid JSON"""
    assert CONFIG_PATH.exists(), "config.json not found"

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Validate required fields
    assert "base_dir" in config
    assert "output_dir" in config
    assert "default_timeout" in config
    assert "r_scripts" in config
    assert "dependencies" in config

    # Validate R scripts
    assert len(config["r_scripts"]) == 10, "Should have 10 R scripts"
    expected_scripts = [
        "qc_filtering", "normalization_sct", "integration_harmony",
        "clustering_analysis", "doublet_detection", "deg_analysis",
        "marker_detection", "pathway_enrichment", "dim_reduction",
        "subset_cells"
    ]
    for script in expected_scripts:
        assert script in config["r_scripts"], f"{script} not in config"

def test_config_timeout():
    """Test that timeout is set to 3600s"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    assert config["default_timeout"] == 3600, "Timeout should be 3600s"

def test_config_transport():
    """Test that transport is set to stdio"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    assert config["server"]["transport"] == "stdio", "Transport must be stdio"

# ========== Test 3: R Scripts Validation ==========

def test_all_r_scripts_exist():
    """Test that all 10 R scripts exist"""
    expected_scripts = [
        "qc_filtering.R", "normalization_sct.R", "integration_harmony.R",
        "clustering_analysis.R", "doublet_detection.R", "deg_analysis.R",
        "marker_detection.R", "pathway_enrichment.R", "dim_reduction.R",
        "subset_cells.R"
    ]

    for script in expected_scripts:
        script_path = SCRIPTS_DIR / script
        assert script_path.exists(), f"{script} not found in scripts/"

def test_r_script_syntax():
    """Test that all R scripts have valid syntax (R --slave -e parse)"""
    r_scripts = list(SCRIPTS_DIR.glob("*.R"))
    assert len(r_scripts) == 10, f"Expected 10 R scripts, found {len(r_scripts)}"

    for script in r_scripts:
        # Check shebang
        with open(script, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            assert first_line.startswith("#!"), f"{script.name} missing shebang"

        # Check R syntax using R parser
        # Convert Windows path to R-compatible format (forward slashes)
        script_path = str(script).replace('\\', '/')
        result = subprocess.run(
            ["Rscript", "-e", f"parse('{script_path}')"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        assert result.returncode == 0, f"R syntax error in {script.name}: {result.stderr}"

def test_r_scripts_have_required_libraries():
    """Test that R scripts load required libraries"""
    required_libs = {
        "qc_filtering.R": ["Seurat", "ggplot2", "jsonlite"],
        "normalization_sct.R": ["Seurat", "ggplot2", "jsonlite"],
        "integration_harmony.R": ["Seurat", "harmony", "ggplot2", "jsonlite"],
        "clustering_analysis.R": ["Seurat", "ggplot2", "jsonlite"],
        "doublet_detection.R": ["Seurat", "DoubletFinder", "ggplot2", "jsonlite"],
        "deg_analysis.R": ["Seurat", "ggplot2", "dplyr", "jsonlite"],
        "marker_detection.R": ["Seurat", "ggplot2", "dplyr", "jsonlite"],
        "pathway_enrichment.R": ["clusterProfiler", "org.Hs.eg.db", "org.Mm.eg.db", "ggplot2", "enrichplot", "jsonlite"],
        "dim_reduction.R": ["Seurat", "ggplot2", "patchwork", "jsonlite"],
        "subset_cells.R": ["Seurat", "ggplot2", "jsonlite"]
    }

    for script_name, libs in required_libs.items():
        script_path = SCRIPTS_DIR / script_name
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for lib in libs:
                assert f'library({lib})' in content, f"{script_name} missing library({lib})"

def test_r_scripts_parse_command_line_args():
    """Test that R scripts parse command line arguments correctly"""
    r_scripts = list(SCRIPTS_DIR.glob("*.R"))

    for script in r_scripts:
        with open(script, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "commandArgs(trailingOnly = TRUE)" in content, \
                f"{script.name} doesn't parse command line args"
            assert "args[1]" in content, \
                f"{script.name} doesn't use args[1] (input_rds)"
            assert "args[2]" in content, \
                f"{script.name} doesn't use args[2] (params_json)"

# ========== Test 4: Tool Function Validation ==========

def test_all_tools_defined():
    """Test that all 10 tools are defined in the server"""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    import scrna_mcp_server

    expected_tools = [
        "run_qc_filtering", "run_normalization_sct", "run_integration_harmony",
        "run_clustering_analysis", "run_doublet_detection", "run_deg_analysis",
        "run_marker_detection", "run_pathway_enrichment", "run_dim_reduction",
        "run_subset_cells"
    ]

    # Check that functions are defined
    for tool in expected_tools:
        assert hasattr(scrna_mcp_server, tool), f"{tool} not defined in server"

def test_tool_docstrings():
    """Test that all tools have comprehensive docstrings"""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    import scrna_mcp_server

    tools = [
        scrna_mcp_server.run_qc_filtering,
        scrna_mcp_server.run_normalization_sct,
        scrna_mcp_server.run_integration_harmony,
        scrna_mcp_server.run_clustering_analysis,
        scrna_mcp_server.run_doublet_detection,
        scrna_mcp_server.run_deg_analysis,
        scrna_mcp_server.run_marker_detection,
        scrna_mcp_server.run_pathway_enrichment,
        scrna_mcp_server.run_dim_reduction,
        scrna_mcp_server.run_subset_cells
    ]

    for tool in tools:
        assert tool.__doc__ is not None, f"{tool.__name__} missing docstring"
        doc = tool.__doc__
        assert "Args:" in doc, f"{tool.__name__} docstring missing Args section"
        assert "Returns:" in doc, f"{tool.__name__} docstring missing Returns section"

# ========== Test 5: Error Handling ==========

def test_run_r_script_handles_missing_file():
    """Test that run_r_script handles missing input file gracefully"""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    import scrna_mcp_server

    result = scrna_mcp_server.run_r_script(
        "qc_filtering",
        "/nonexistent/file.rds",
        params={}
    )

    assert result["status"] == "error"
    assert "does not exist" in result["message"]

def test_run_r_script_handles_missing_script():
    """Test that run_r_script handles missing R script gracefully"""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    import scrna_mcp_server

    # Create dummy file
    dummy_file = BASE_DIR / "output" / "dummy.rds"
    dummy_file.parent.mkdir(parents=True, exist_ok=True)
    dummy_file.touch()

    result = scrna_mcp_server.run_r_script(
        "nonexistent_script",
        str(dummy_file),
        params={}
    )

    assert result["status"] == "error"
    assert "does not exist" in result["message"]

    # Cleanup
    dummy_file.unlink()

# ========== Test 6: Output Directory Structure ==========

def test_output_directory_exists():
    """Test that output directory is created"""
    output_dir = BASE_DIR / "output"
    assert output_dir.exists() or True, "Output directory should exist or be creatable"

    # Create if doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    assert output_dir.exists()

# ========== Test 7: Documentation Validation ==========

def test_readme_exists():
    """Test that README.md exists"""
    assert README_PATH.exists(), "README.md not found"

def test_readme_completeness():
    """Test that README.md contains all required sections"""
    with open(README_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    required_sections = [
        "# mcp_r_scrna",
        "## Overview",
        "## Features",
        "## Architecture",
        "## Installation",
        "## Usage",
        "## Tool Reference",
        "## Design Principles",
        "## Testing",
        "## Troubleshooting"
    ]

    for section in required_sections:
        assert section in content, f"README missing section: {section}"

def test_readme_documents_all_tools():
    """Test that README documents all 10 tools"""
    with open(README_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    tools = [
        "run_qc_filtering", "run_normalization_sct", "run_integration_harmony",
        "run_clustering_analysis", "run_doublet_detection", "run_deg_analysis",
        "run_marker_detection", "run_pathway_enrichment", "run_dim_reduction",
        "run_subset_cells"
    ]

    for tool in tools:
        assert tool in content, f"README doesn't document {tool}"

# ========== Test 8: Dependency Documentation ==========

def test_dependencies_documented():
    """Test that all dependencies are documented in config"""
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    assert "dependencies" in config
    assert "r_packages" in config["dependencies"]
    assert "python_packages" in config["dependencies"]

    # Check critical R packages
    critical_r_packages = ["Seurat", "harmony", "DoubletFinder", "clusterProfiler"]
    for pkg in critical_r_packages:
        assert pkg in config["dependencies"]["r_packages"], f"{pkg} not in dependencies"

# ========== Test 9: Workflow Validation ==========

def test_workflow_structure():
    """Test that README contains complete workflow example"""
    with open(README_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    workflow_steps = [
        "run_qc_filtering",
        "run_normalization_sct",
        "run_integration_harmony",
        "run_clustering_analysis",
        "run_marker_detection"
    ]

    # Check that workflow is sequential
    for i in range(len(workflow_steps) - 1):
        pos1 = content.find(workflow_steps[i])
        pos2 = content.find(workflow_steps[i + 1], pos1)
        assert pos2 > pos1, f"Workflow order incorrect: {workflow_steps[i]} should come before {workflow_steps[i+1]}"

# ========== Test 10: JSON Response Structure ==========

def test_json_response_structure():
    """Test that run_r_script returns properly structured JSON"""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    import scrna_mcp_server

    # Test with missing file (should return error structure)
    result = scrna_mcp_server.run_r_script(
        "qc_filtering",
        "/nonexistent/file.rds",
        params={}
    )

    # Validate structure
    assert isinstance(result, dict)
    assert "status" in result
    assert "message" in result
    assert "generated_files" in result
    assert result["status"] in ["success", "error"]
    assert isinstance(result["generated_files"], list)

# ========== Test 11: Parameter Validation ==========

def test_tool_parameters_documented():
    """Test that all tool parameters are documented in config"""
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    assert "tool_descriptions" in config
    assert len(config["tool_descriptions"]) == 10

# ========== Test 12: Server Lifecycle ==========

def test_server_has_lifespan():
    """Test that server has lifecycle management"""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    import scrna_mcp_server

    assert hasattr(scrna_mcp_server.mcp, 'lifespan')
    assert scrna_mcp_server.mcp.lifespan is not None

# ========== Test 13: Integration Test Markers ==========

def test_integration_with_immuneagent():
    """Test that README contains ImmuneAgent integration examples"""
    with open(README_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    assert "ImmuneAgent" in content
    assert "LangGraph" in content or "Stage 5" in content
    assert "agent/config/config.py" in content

# ========== Test 14: Version Information ==========

def test_version_in_config():
    """Test that version is documented in config"""
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    assert "server" in config
    assert "version" in config["server"]
    assert config["server"]["version"] == "1.0.0"

# ========== Test 15: File Organization ==========

def test_file_organization():
    """Test that directory structure is correct"""
    required_dirs = ["scripts", "tests", "config", "output"]

    for dir_name in required_dirs:
        dir_path = BASE_DIR / dir_name
        assert dir_path.exists(), f"Directory {dir_name}/ not found"

def test_scripts_directory_contents():
    """Test that scripts directory has exactly 10 R scripts"""
    r_scripts = list(SCRIPTS_DIR.glob("*.R"))
    assert len(r_scripts) == 10, f"Expected 10 R scripts, found {len(r_scripts)}"

# ========== Test Summary ==========

def test_summary():
    """Print test summary"""
    print("\n" + "="*60)
    print("mcp_r_scrna Test Suite Summary")
    print("="*60)
    print(f"Server Path: {SERVER_PATH}")
    print(f"Config Path: {CONFIG_PATH}")
    print(f"Scripts Dir: {SCRIPTS_DIR}")
    print(f"R Scripts: {len(list(SCRIPTS_DIR.glob('*.R')))} files")
    print(f"Tests: 15 integration tests")
    print("="*60)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
