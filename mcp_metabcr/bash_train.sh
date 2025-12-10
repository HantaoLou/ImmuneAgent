
source activate /home/jachin/anaconda3/envs/leslie2
cd /home/jachin/data/Github/meta_bcr

export CUDA_VISIBLE_DEVICES=2
# # python -u OM_train.py -C Config/config_lct.yaml
# nohup python -u OM_train.py -C Config/config_lct.yaml > train_log.txt 2>&1 &

# python -u OM_train.py -C Config/config_om.yaml

# antigen_name=sars
# antigen_name=rsv
antigen_name=flu

task_name=bind
# task_name=neu

# config_date=2502b21  # 2817936
# config_date=250222  # 2809690
# config_date=250223  # 2621280
# config_date=250224  # 1413748/1407821
# config_date=250225  # 
config_date=250312  # 

# python -u train_meta_250312.py --antigen_name $antigen_name --task_name $task_name --config_date $config_date
nohup python -u train_meta_250312.py --antigen_name $antigen_name --task_name $task_name --config_date $config_date > Log/train_log-$task_name-$config_date.txt 2>&1 &

# python -u train_five_fold_rsv_meta_250219.py --task_name $task_name --config_date $config_date
# nohup python -u train_five_fold_rsv_meta_250219.py --task_name $task_name --config_date $config_date > Log/train_log-$task_name-$config_date.txt 2>&1 &