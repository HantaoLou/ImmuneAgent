#!/usr/bin/env python3
"""
Test run_subset_cells method functionality

This test file validates cell subsetting functionality including:
1. Check actual metadata columns in the data
2. Cell subsetting based on actual cluster values
3. Inverse subsetting (exclude specified values)
4. Error handling tests

From bioinformatics perspective ensuring:
- Biological significance of cell subsetting
- Data integrity validation
- Statistical accuracy
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import server module
from scrna_mcp_server import run_subset_cells, load_config

def check_seurat_metadata(input_rds):
    """Check Seurat object metadata information"""
    print("Checking Seurat object metadata...")
    
    # Create temporary R script to check metadata
    r_script = """
    suppressPackageStartupMessages(library(Seurat))
    args <- commandArgs(trailingOnly = TRUE)
    input_rds <- args[1]
    
    seurat_obj <- readRDS(input_rds)
    
    cat("=== Basic Seurat Object Info ===\\n")
    cat("Number of cells:", ncol(seurat_obj), "\\n")
    cat("Number of genes:", nrow(seurat_obj), "\\n")
    cat("\\n=== Metadata Columns ===\\n")
    print(colnames(seurat_obj@meta.data))
    
    # Check clustering information
    if ("seurat_clusters" %in% colnames(seurat_obj@meta.data)) {
        cat("\\n=== seurat_clusters Distribution ===\\n")
        cluster_table <- table(seurat_obj@meta.data$seurat_clusters)
        print(cluster_table)
        cat("Number of clusters:", length(unique(seurat_obj@meta.data$seurat_clusters)), "\\n")
    }
    
    # Check other possible grouping columns
    possible_columns <- c("celltype", "cell_type", "cluster", "clusters", "orig.ident", "sample")
    for (col in possible_columns) {
        if (col %in% colnames(seurat_obj@meta.data)) {
            cat("\\n===", col, "Distribution ===\\n")
            col_table <- table(seurat_obj@meta.data[[col]])
            print(col_table)
        }
    }
    """
    
    # Write temporary R script
    temp_script = current_dir / "temp_check_metadata.R"
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(r_script)
    
    try:
        # Run R script
        result = subprocess.run(
            ["Rscript", str(temp_script), input_rds],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        
        if result.returncode == 0:
            print("✅ Metadata check successful")
            print(result.stdout)
            return result.stdout
        else:
            print(f"❌ Metadata check failed: {result.stderr}")
            return None
    finally:
        # Clean up temporary file
        if temp_script.exists():
            temp_script.unlink()

def test_subset_cells():
    """Test cell subsetting functionality"""
    
    # Test data path
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    print("=" * 80)
    print("Single-cell RNA-seq Cell Subsetting Functionality Test")
    print("=" * 80)
    
    # Validate input file
    print(f"Input file: {input_rds}")
    if not os.path.exists(input_rds):
        print(f"❌ Error: Input file does not exist: {input_rds}")
        return
    
    file_size = os.path.getsize(input_rds) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.1f} MB")
    print("✅ Input file validation passed")
    print()
    
    # Check metadata
    metadata_info = check_seurat_metadata(input_rds)
    if not metadata_info:
        print("❌ Cannot get metadata information, skipping tests")
        return
    
    print()
    
    # Test 1: Cluster-based cell subsetting (using actual existing cluster values)
    print("Test 1: Cluster-based cell subsetting (keep first 2 clusters)")
    print("-" * 60)
    try:
        # Use first few clusters for testing
        result1 = run_subset_cells(
            input_rds=input_rds,
            subset_column="seurat_clusters",
            subset_values=["0", "1"],  # Usually clusters 0 and 1 exist
            invert=False
        )
        
        print(f"Status: {result1['status']}")
        if result1['status'] == 'success':
            print(f"✅ Cluster subsetting completed successfully")
            print(f"Output directory: {result1.get('output_directory', 'N/A')}")
            print(f"Generated files: {result1.get('file_count', 0)}")
            if result1.get('generated_files'):
                print("Generated files:")
                for file in result1['generated_files'][:5]:  # Show first 5 files
                    print(f"  - {file}")
        else:
            print(f"❌ Cluster subsetting failed: {result1.get('message', 'Unknown error')}")
            if 'stderr' in result1:
                print(f"Error details: {result1['stderr']}")
    except Exception as e:
        print(f"❌ Test 1 exception: {str(e)}")
    
    print()
    
    # Test 2: orig.ident-based cell subsetting (using actual value from metadata)
    print("Test 2: Sample identity-based cell subsetting")
    print("-" * 50)
    try:
        result2 = run_subset_cells(
            input_rds=input_rds,
            subset_column="orig.ident",  # Usually existing column
            subset_values=["your_project"],  # Actual value from metadata check
            invert=False
        )
        
        print(f"Status: {result2['status']}")
        if result2['status'] == 'success':
            print(f"✅ Sample subsetting completed successfully")
            print(f"Output directory: {result2.get('output_directory', 'N/A')}")
            print(f"Generated files: {result2.get('file_count', 0)}")
        else:
            print(f"⚠️ Sample subsetting failed: {result2.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"⚠️ Test 2 exception: {str(e)}")
    
    print()
    
    # Test 3: Inverse subsetting (exclude specific clusters)
    print("Test 3: Inverse subsetting (exclude non-existent cluster)")
    print("-" * 55)
    try:
        result3 = run_subset_cells(
            input_rds=input_rds,
            subset_column="seurat_clusters",
            subset_values=["999"],  # Non-existent cluster, inverse should keep all cells
            invert=True  # Inverse subsetting
        )
        
        print(f"Status: {result3['status']}")
        if result3['status'] == 'success':
            print(f"✅ Inverse subsetting completed successfully")
            print(f"Output directory: {result3.get('output_directory', 'N/A')}")
            print(f"Generated files: {result3.get('file_count', 0)}")
        else:
            print(f"❌ Inverse subsetting failed: {result3.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Test 3 exception: {str(e)}")
    
    print()
    
    # Test 4: Error handling test (non-existent column)
    print("Test 4: Error handling test (non-existent column)")
    print("-" * 50)
    try:
        result4 = run_subset_cells(
            input_rds=input_rds,
            subset_column="nonexistent_column",  # Non-existent column
            subset_values=["value1", "value2"],
            invert=False
        )
        
        print(f"Status: {result4['status']}")
        if result4['status'] == 'error':
            print(f"✅ Error handling correct: {result4.get('message', 'Unknown error')}")
        else:
            print(f"⚠️ Expected error but succeeded: {result4}")
    except Exception as e:
        print(f"✅ Correctly caught exception: {str(e)}")
    
    print()
    
    # Bioinformatics analysis guidance
    print("=" * 80)
    print("Bioinformatics Analysis Guidance")
    print("=" * 80)
    print("1. Cell Subsetting Strategies:")
    print("   - Cluster-based: Suitable for exploratory analysis, retain cell populations of interest")
    print("   - Cell type-based: Suitable for annotated data, specific cell type analysis")
    print("   - Sample-based: Suitable for multi-sample comparative analysis")
    print("   - Inverse subsetting: Used to exclude low-quality or irrelevant cell populations")
    print()
    print("2. Quality Control Recommendations:")
    print("   - Check cell count and gene expression distribution before subsetting")
    print("   - Ensure sufficient cell numbers are retained (recommend >100 cells)")
    print("   - Validate biological marker gene expression after subsetting")
    print("   - Check if post-subset cell type composition is reasonable")
    print()
    print("3. Downstream Analysis Considerations:")
    print("   - May need to re-normalize and re-reduce dimensions after subsetting")
    print("   - Re-evaluate cell cycle and batch effects")
    print("   - Update differential expression and pathway enrichment analyses")
    print("   - Consider whether re-clustering analysis is needed")
    print()
    print("4. Technical Considerations:")
    print("   - Extensive subsetting may affect statistical power")
    print("   - Maintain balance of biological replicates")
    print("   - Record subset parameters to ensure analysis reproducibility")
    print()

def main():
    """Main function"""
    print("Starting test of run_subset_cells method...")
    print()
    
    # Check configuration
    try:
        config = load_config()
        print(f"Configuration loaded successfully")
        print()
    except Exception as e:
        print(f"Configuration loading failed: {e}")
        return
    
    # Run tests
    test_subset_cells()
    
    print("Testing completed!")

if __name__ == "__main__":
    main()