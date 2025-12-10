'''
Mean Teacher for RSV
Jachin
2025.02.12
'''

import os
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader as DataLoader_n
import pandas as pd
import random
import glob

from Config.config import get_config
from MetaBCR.lm_gnn_model_jz import XBCR_ACNN_woBERT_meta
from MetaBCR.lm_gnn_model_jz0508_unfrozen import XBCR_ACNN_dense_meta
from MetaBCR.lm_gnn_model_jz import DeepBCR_ACEXN_protbert
from MetaBCR.lm_gnn_model_jz import Adaptive_Regulariz
from MetaBCR.dataset_rsv import Ab_Dataset, Ab_Dataset_mean_teacher
import MetaBCR.metrics as metrics
from MetaBCR.losses import *
from MetaBCR.lm_gnn_model_jz0508_unfrozen import Adaptive_Regulariz


def read_table(file):
    try:
        data = pd.read_csv(file)
    except:
        data = pd.read_excel(file)
    return data

def read_tables(files):
    if isinstance(files, str):
        data = read_table(files)
    else:
        data = pd.concat([read_table(f) for f in files], ignore_index=True)
    return data

def get_model(Model, _cfg_):
    return Model(extra_dense=True, block_num=8,
                 freeze_bert=_cfg_.freeze_bert,
                 ab_freeze_layer_count=_cfg_.freeze_layer_count,
                 bert=_cfg_.bert_name)

def implement(model, dataloader, _cfg_, wolabel=False):
    predictions_main_tr = []
    labels_main_tr = []
    lossweight_main_tr = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            input_ids_ab_v = data['input_ids_ab_v'].to(_cfg_.device)
            attention_mask_ab_v = data['attention_mask_ab_v'].to(_cfg_.device)
            input_ids_ab_l = data['input_ids_ab_l'].to(_cfg_.device)
            attention_mask_ab_l = data['attention_mask_ab_l'].to(_cfg_.device)
            input_ids_ag = data['input_ids_ag'].to(_cfg_.device)
            outputs = model(ag_x=input_ids_ag, ab_x=input_ids_ab_v,
                            attention_mask_ab_v=attention_mask_ab_v,
                            ab_l=input_ids_ab_l,
                            attention_mask_ab_l=attention_mask_ab_l)
            predictions_main_tr.extend(outputs[0].cpu().view(-1).tolist())
            if wolabel == False:
                labels_main_tr.extend(data['label'].view(-1).tolist())
                lossweight_main_tr.extend(data['loss_main'].view(-1).tolist())
    if wolabel: return predictions_main_tr
    return predictions_main_tr, labels_main_tr, lossweight_main_tr

def pred(_cfg_, antigen_name='rsv',epoch_num=None, fold=0,fdir_tst=None,output_dir=None):
    # _RESULT_DIR = f'Results/{_cfg_.train_mode}/rslt-meta_{_cfg_.model}_{_cfg_.date}_{_cfg_.train_mode}_{_cfg_.prop}_fold{fold}_meta{_cfg_.benchmark}-semi/'
    # _RESULT_DIR = f'/home/data/Github/meta_bcr/Results/{_cfg_.train_mode}/rslt-meta_{_cfg_.model}_{_cfg_.date}_{_cfg_.train_mode}_{_cfg_.prop}_fold{fold}_meta{_cfg_.benchmark}-semi/'
    # _RESULT_DIR = "/home/jachin/data/Github/meta_bcr/Results/rsv-bind/rslt-meta_XBCR_ACNN_250212-rsv_bind-updt_itr=5-epoch=100-f1-w_unsup_loss=1-ab-nopretrain-extradense-noclamp-acc-2201[0.5]-semi_rsv-bind_[2,2,0,1,0.5]_fold0_metaf1-semi"
    # _RESULT_DIR =os.path.dirname(_RESULT_DIR)
    _RESULT_DIR = f'Results/{_cfg_.train_mode}/rslt-*_fold{fold}*'

    if _cfg_.model == 'XBCR_ACNN':
        if _cfg_.use_onehot:
            Model = XBCR_ACNN_woBERT_meta
        else:
            Model = XBCR_ACNN_dense_meta
    elif _cfg_.model == 'DeepBCR_ACEXN_protbert':
        Model = DeepBCR_ACEXN_protbert
    else:
        print(f'ERROR [ test ] : Wrong model {_cfg_.model}')
        raise ValueError

    data_test = read_tables(fdir_tst)
    test_set = Ab_Dataset(datalist=[data_test], proportions=[None], sample_func=['sample'],
                          n_samples=data_test.shape[0], is_rand_sample=False,
                          onehot=_cfg_.use_onehot, rand_shift=False)
    test_loader = DataLoader_n(dataset=test_set, batch_size=_cfg_.batch_sz, num_workers=0, shuffle=False)

    model = get_model(Model=Model, _cfg_=_cfg_)
    model.to(_cfg_.device)

    model_dirs = glob.glob(os.path.join(_RESULT_DIR, "epoch_*.pth"))
    # model_dirs = glob.glob(os.path.join(_RESULT_DIR, "epoch_*.pth"))
    print(model_dirs)
    if epoch_num is None:
        print(f'INFO [ test ] : Loading the best model from {_RESULT_DIR}')
        epoch_nums = [int(os.path.basename(m).split('_')[-1].split('.')[0]) for m in model_dirs]
        epoch_num = max(epoch_nums)
    model_path = [m for m in model_dirs if f"epoch_{epoch_num}.pth" in m][0]
    _RESULT_DIR = os.path.dirname(model_path)
    print(model_path)


    model.load_state_dict(torch.load(model_path), strict=False)
    print(f'INFO [ test ] : Loaded model params from {model_path}')

    model.eval()
    predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader, _cfg_)
    data_test['output'] = np.around(np.array(predictions_tst)).tolist()
    data_test['predict'] = predictions_tst
    tst_basename = os.path.basename(fdir_tst).split(".")[0]
    if output_dir is None:
        output_dir = _RESULT_DIR
    data_test.to_excel(os.path.join(f"{output_dir}",f"test_results_{tst_basename}_fold{fold}.xlsx"), index=False, header=True)
    print(f'INFO [ test ] : Test results saved to {output_dir}/test_results_fold{fold}.xlsx')

if __name__ == '__main__':
    antigen_name = 'rsv'
    fold = 4  # Assuming fold 0 for testing
    # fdir_tst = f'Data/RSV/0212_RSV_bind_absplit_test.csv'
    fdir_tst = f'Data/RSV_infer/0214_libra-test.csv'
    output_dir = f'Data/RSV_infer/'
    configure = get_config(f"Config/config_five_fold_{antigen_name}_bind_meta_250212_semi_supervise.json")
    test(_cfg_=configure, antigen_name=antigen_name,fold=fold, fdir_tst=fdir_tst,output_dir=output_dir)
