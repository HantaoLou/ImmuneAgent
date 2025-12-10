#!/bin/bash

root_dir="/data/lht/meta_bcr"
GPU_ID=5

antigen_name="rsv"
task_name="neu"
config_date="250225"
fdir_tst="Data/RSV_infer/HL_paired_clonefiltered_func-A.xlsx"
# fdir_tst="Data/RSV_infer/HL_paired_clonefiltered_func-B.xlsx"
output_dir="Data/RSV_infer"


# ==========================================================
cd $root_dir

export CUDA_VISIBLE_DEVICES=$GPU_ID

# conda init bash
# source ~/.bashrc
# source /home/lht/Miniconda/miniconda3/bin/activate metabcr
source /data_new/lht/Miniconda/miniconda3/bin/activate metabcr


# nohup python -u csv_run_af3.py > log_af3-GPU_$GPU_ID.txt 2>&1 &
python -u predict_metabcr.py --antigen_name=$antigen_name --task_name=$task_name --config_date=$config_date --fdir_tst=$fdir_tst --output_dir=$output_dir

