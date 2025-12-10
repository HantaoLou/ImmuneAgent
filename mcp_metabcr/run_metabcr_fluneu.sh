#!/bin/bash

root_dir="/data/lht/meta_bcr"
GPU_ID=1

# model settings
# antigen_name="flu"
# task_name="bind"
# config_date="250312"
antigen_name="flu"
task_name="neu"
config_date="240905"


# file to predict and the saved directory
fdir_tst="Data/FLU_infer/0322_ddg_datasets.csv"
output_dir="Data/FLU_infer"


# ==========================================================
cd $root_dir

export CUDA_VISIBLE_DEVICES=$GPU_ID

# conda init bash
# source ~/.bashrc
# source /home/lht/Miniconda/miniconda3/bin/activate metabcr
source /data_new/lht/Miniconda/miniconda3/bin/activate metabcr


# nohup python -u csv_run_af3.py > log_af3-GPU_$GPU_ID.txt 2>&1 &
python -u predict_metabcr.py --antigen_name=$antigen_name --task_name=$task_name --config_date=$config_date --fdir_tst=$fdir_tst --output_dir=$output_dir

