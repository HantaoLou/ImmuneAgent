import subprocess
import os
import json
import datetime
from pathlib import Path

def generate_markdown_report(params, results):
    """
    Fills the English Markdown template with actual analysis results.
    """
    template_path = os.path.join(os.path.dirname(__file__),"report_template.md")
    
    if not os.path.exists(template_path):
        return "Warning: Report template not found."

    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Mapping Python variables to Markdown placeholders
    replacements = {
        "{{target_group}}": params['target_group'],
        "{{group_col}}": params.get('group_col', 'condition'),
        "{{label_col}}": params.get('label_col', 'celltype'),
        "{{root_type}}": params['root_type'],
        "{{status}}": results['status'].upper(),
        "{{completion_time}}": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "{{umap_plot_path}}": results['files']['plot_png'],
        "{{reduction}}": results['metadata'].get('reduction', 'UMAP'),
        "{{cds_rds_path}}": results['files']['data_rds'],
        "{{pdf_plot_path}}": results['files']['plot_pdf']
    }

    # Perform bulk replacement
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, str(value))

    report_path = os.path.join(params['output_path'], "Trajectory_Analysis_Report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return report_path

def run_trajectory_skill(params):
    # 1. Path Standardization
    rds_path = os.path.abspath(params['rds_path'])
    out_dir = os.path.abspath(params.get('output_path', './results'))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 2. Construct R Command
    script_path = os.path.join(os.path.dirname(__file__), "r_scripts", "trajectory_main.R")
    cmd = [
        "Rscript", script_path,
        "--input", rds_path,
        "--group_col", params.get('group_col', 'condition'),
        "--condition", params['target_group'],
        "--label_col", params.get('label_col', 'celltype'),
        "--root_type", params['root_type'],
        "--outdir", out_dir
    ]

    # 3. Execution
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 4. Prepare results metadata for the template
        analysis_results = {
            "status": "success",
            "files": {
                "plot_png": "pseudotime_umap.png", # Relative path for Markdown
                "plot_pdf": os.path.join(out_dir, "pseudotime_umap.pdf"),
                "data_rds": os.path.join(out_dir, "trajectory_result.rds")
            },
            "metadata": {
                "reduction": "UMAP" # This could be parsed from R stdout if needed
            }
        }

        # 5. Generate the Markdown Report
        report_file = generate_markdown_report(params, analysis_results)
        
        return {
            "status": "success",
            "report_path": report_file,
            "stdout": result.stdout
        }

    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": "R script execution failed.",
            "stderr": e.stderr
        }

if __name__ == "__main__":
    # Test Input
    test_input = {
        "rds_path": "data/your_data.rds",
        "target_group": "EP",
        "root_type": "Basal",
        "output_path": "./output_test"
    }
    print(run_trajectory_skill(test_input))