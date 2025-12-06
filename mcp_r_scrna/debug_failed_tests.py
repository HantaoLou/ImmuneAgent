#!/usr/bin/env python3
"""
Debug script for failed subset tests

This script specifically debugs:
1. Test 2: Sample identity-based cell subsetting failure
2. Test 4: Error handling test details
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import server module
from scrna_mcp_server import run_subset_cells, load_config

def debug_test2_sample_subsetting():
    """Debug Test 2: Sample identity-based cell subsetting"""
    print("=" * 80)
    print("DEBUG: Test 2 - Sample identity-based cell subsetting")
    print("=" * 80)
    
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    # From metadata check, we know orig.ident has value "your_project", not "SeuratProject"
    print("Testing with correct orig.ident value: 'your_project'")
    
    try:
        result = run_subset_cells(
            input_rds=input_rds,
            subset_column="orig.ident",
            subset_values=["your_project"],  # Use actual value from metadata
            invert=False
        )
        
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"✅ Sample subsetting successful!")
            print(f"Output directory: {result.get('output_directory', 'N/A')}")
            print(f"Generated files: {result.get('file_count', 0)}")
            if result.get('stdout'):
                print("R script output:")
                print(result['stdout'][-500:])  # Last 500 chars
        else:
            print(f"❌ Sample subsetting failed: {result.get('message', 'Unknown error')}")
            if 'stderr' in result:
                print(f"Error details: {result['stderr']}")
            if 'stdout' in result:
                print(f"R script output: {result['stdout']}")
                
    except Exception as e:
        print(f"❌ Exception in Test 2: {str(e)}")
    
    print()

def debug_test4_error_handling():
    """Debug Test 4: Error handling test details"""
    print("=" * 80)
    print("DEBUG: Test 4 - Error handling test details")
    print("=" * 80)
    
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    print("Testing with non-existent column: 'nonexistent_column'")
    
    try:
        result = run_subset_cells(
            input_rds=input_rds,
            subset_column="nonexistent_column",
            subset_values=["value1", "value2"],
            invert=False
        )
        
        print(f"Status: {result['status']}")
        print(f"Message: {result.get('message', 'No message')}")
        
        if 'stderr' in result:
            print("Detailed error information:")
            print(result['stderr'])
        if 'stdout' in result:
            print("R script output:")
            print(result['stdout'])
            
        # This should be an error, so let's verify the error handling is working correctly
        if result['status'] == 'error':
            print("✅ Error handling is working correctly - detected non-existent column")
        else:
            print("⚠️ Unexpected: Should have failed but didn't")
            
    except Exception as e:
        print(f"Exception in Test 4: {str(e)}")
        print("✅ Exception correctly caught - this is expected behavior")
    
    print()

def test_additional_scenarios():
    """Test additional edge cases"""
    print("=" * 80)
    print("DEBUG: Additional edge case tests")
    print("=" * 80)
    
    input_rds = r"D:\data\test_data_20251001\Age_Bcells.rds"
    
    # Test with empty subset_values
    print("Test: Empty subset values")
    try:
        result = run_subset_cells(
            input_rds=input_rds,
            subset_column="seurat_clusters",
            subset_values=[],  # Empty list
            invert=False
        )
        print(f"Empty values result: {result['status']} - {result.get('message', 'No message')}")
    except Exception as e:
        print(f"Empty values exception: {str(e)}")
    
    print()
    
    # Test with non-existent cluster values
    print("Test: Non-existent cluster values")
    try:
        result = run_subset_cells(
            input_rds=input_rds,
            subset_column="seurat_clusters",
            subset_values=["999", "1000"],  # Non-existent clusters
            invert=False
        )
        print(f"Non-existent clusters result: {result['status']} - {result.get('message', 'No message')}")
        if 'stderr' in result:
            print(f"Error details: {result['stderr']}")
    except Exception as e:
        print(f"Non-existent clusters exception: {str(e)}")
    
    print()

def main():
    """Main debug function"""
    print("Starting debug analysis of failed tests...")
    print()
    
    # Check configuration
    try:
        config = load_config()
        print(f"Configuration loaded successfully")
        print()
    except Exception as e:
        print(f"Configuration loading failed: {e}")
        return
    
    # Debug each failed test
    debug_test2_sample_subsetting()
    debug_test4_error_handling()
    test_additional_scenarios()
    
    print("Debug analysis completed!")

if __name__ == "__main__":
    main()