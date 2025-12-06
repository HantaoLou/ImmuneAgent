import json
import torch
import os
from types import SimpleNamespace

def get_config(config_path, print_info=True):
    # Load parameters from the JSON file
    try:
        with open(config_path, 'r') as file: config = json.load(file)
    except:
        print(f"ERROR [ Config.config.get_config ] : Cannot read JSON file from path <{config_path}>")
        raise ValueError
    # print out important info
    if print_info:
        print('INFO [ Config.config.get_config ] : Parameters from config file')
        for k, v in config.items(): print(f"    {k}: {v}")
    return SimpleNamespace(**config)


if __name__=='__main__':
    config = get_config('Config/config_five_fold_flu_bind_meta_240621_semi_supervise.json')
    # Access parameters
    device = torch.device(config['device'])
    root_dir = config['root_dir']
    pretrain_model_dir = config['pretrain_model_dir'].format(0)  # Example fold index
