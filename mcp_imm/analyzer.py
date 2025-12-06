import os
import asyncio
import pandas as pd
from metagpt.logs import logger
from metagpt.roles.role import Role
from metagpt.actions.action import Action
from immagents.immroles import *
import argparse

IMMGPT_DATA_PATH = "/data/lht/immgpt/data"
ANALYZE_COLUMNS = {
    "sum_score": "The sum of the estimated neutralizing scores of the antibodies, higher is better.",
    "expasy": "The protein instability index test result for the antibodies, pass is better.",
    "leiden": "The group id number for Leiden clustering of the antibodies.",
    "total_energy": "The total energy of the antibodies, lower is better.",
    "H5N1_TE/H1N1_WI/H1N1_VI (bind)(experiment)": "The binding affinity of the antibodies with this antigen, higher is better."
}



def create_analyzing_prompt(
        analyze_column: dict=ANALYZE_COLUMNS,
        export_column: list[str]=None,
        action_prompt: str = "select the 5-top antibodies that you think have the highest potential to broadly against viruses?"
        ) -> str:
    analyzing_prompt = (
        f"Please provide a detailed analysis of the table with focusing on the following columns:\n"
        # f"In the file, please provide a simple analysis of the following columns:\n"
    )
    for key, value in analyze_column.items():
        analyzing_prompt += f"- {key}: {value}\n"
    analyzing_prompt += "\n"
    analyzing_prompt += (
        f"{action_prompt}?\n"
    )
    analyzing_prompt += "\n"
    if export_column:
        analyzing_prompt += f"and then export the selected rows with the columns: {export_column}\n"

    return analyzing_prompt

def analyze_table():
    parser = argparse.ArgumentParser(description='analyzer')
    parser.add_argument('--input_file', type=str, default="H5N1_first-batch/0307_first-batch_exp-results.xlsx")
    args = parser.parse_args()

    # fire.Fire(main)
    # action_prompt = "can you sort the antibodies from the table file that you think have the highest potential to broadly against the viruses?"
    action_prompt = "Can you select the 5-top antibodies that you think have the highest potential to broadly against viruses?"

    # result_file = "H5N1_first-batch/0307_first-batch_exp-results.xlsx"
    result_file = args.input_file
    print(f"result_file: {result_file}")

    result_name = os.path.basename(result_file).split(".")[0]
    analysis_file = result_name.replace("results", "analysis.txt")
    log_file = result_name.replace("results", "log.txt")
    selected_file = result_name.replace("results", "selected.csv")
    
    results_path = os.path.join(IMMGPT_DATA_PATH, result_file)
    analysis_path = os.path.join(os.path.dirname(results_path), "analysis")
    os.makedirs(analysis_path, exist_ok=True)
    selected_path = os.path.join(os.path.dirname(results_path), "selected")
    os.makedirs(selected_path, exist_ok=True)
    log_path = os.path.join(os.path.dirname(results_path), "log")
    os.makedirs(log_path, exist_ok=True)
    
    file_paths = {
        "results": results_path,
        "analysis": os.path.join(analysis_path, analysis_file),
        "selected": os.path.join(selected_path, selected_file),
        "log": os.path.join(log_path, log_file)
    }
    query_idea=""
    query_idea += create_analyzing_prompt(action_prompt=action_prompt)
    n_round = 5
    print(query_idea)


    analyzer = Analyzer()
    response=asyncio.run(analyzer.run(query_idea,file_paths))

    with open(file_paths['log'], 'w') as f:
        f.write(str(response))
        print(f"Log saved to {file_paths['log']}")

    return selected_file

if __name__ == "__main__":
    selected_file = analyze_table()