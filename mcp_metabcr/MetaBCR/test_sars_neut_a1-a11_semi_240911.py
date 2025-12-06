'''
Mean Teacher
Yu Chen
2024.06.21
'''
import config_five_fold_sars_neu_small_meta_240905_unfrozen_semi_supervise as _cfg_

if _cfg_.model == 'XBCR_ACNN':
    if _cfg_.use_onehot:
        from MetaBCR.lm_gnn_model_jz import XBCR_ACNN_woBERT_meta as Model
    else:
        # from lm_gnn_model_jz import XBCR_ACNN_meta as Model
        from MetaBCR.lm_gnn_model_jz0508_unfrozen import XBCR_ACNN_dense_meta as Model
elif _cfg_.model == 'DeepBCR_ACEXN_protbert':
    from MetaBCR.lm_gnn_model_jz import DeepBCR_ACEXN_protbert as Model
else:
    print('Wrong model {}'.format(_cfg_.model))
    raise ValueError

import os
from torch.utils.data import DataLoader as DataLoader_n
import pandas as pd

from MetaBCR.dataset_sars import Ab_Dataset
from MetaBCR.losses import *

# from lm_gnn_model_jz0508 import get_frozen_bert, get_unfrozen_bert

# HYPER PARAM
_device = torch.device('cuda')  # train params
_batch_sz = 4096
_use_onehot = False
# _use_onehot = True
_lr = 0.00001
_print_step = 20
_regul_step = 100
_regul_v = [0.02, 0.001]
_regul_tgt_dev_rat = 0.08
_best_val_epoch = 125
_rand_seed = 2023

_root_dir = './'

_RESULT_DIR = _root_dir + '0905-small_rbd_neu_results_semi/'  # output dir
if not os.path.exists(_RESULT_DIR):
    print('Cannot find [RESULT DIR], created a new one.')
    os.makedirs(_RESULT_DIR)
print('[RESULT DIR]: {}'.format(_RESULT_DIR))

def implement(model, dataloader, wolabel=False):
    '''
    Model implement function.
        input: model & dataloader
        output: prediction, labels, and loss weights
    '''
    predictions_main_tr = []
    labels_main_tr = []
    lossweight_main_tr = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            # data pre-process
            input_ids_ab_v = data['input_ids_ab_v'].to(_cfg_.device)
            attention_mask_ab_v = data['attention_mask_ab_v'].to(_cfg_.device)
            input_ids_ab_l = data['input_ids_ab_l'].to(_cfg_.device)
            attention_mask_ab_l = data['attention_mask_ab_l'].to(_cfg_.device)
            input_ids_ag = data['input_ids_ag'].to(_cfg_.device)
            # prediction
            outputs = model(ag_x=input_ids_ag, ab_x=input_ids_ab_v,
                            attention_mask_ab_v=attention_mask_ab_v,
                            ab_l=input_ids_ab_l,
                            attention_mask_ab_l=attention_mask_ab_l)
            # loss
            predictions_main_tr.extend(outputs[0].cpu().view(-1).tolist())
            if wolabel == False:
                labels_main_tr.extend(data['label'].view(-1).tolist())
                lossweight_main_tr.extend(data['loss_main'].view(-1).tolist())
    if wolabel: return predictions_main_tr
    return predictions_main_tr, labels_main_tr, lossweight_main_tr

def font(out, gt):
    # this is for visualizing the correctness of predictions,
    # 'o' for correct predictions, 'x' for false predictions.
    ff = []
    for i in range(len(out)):
        fff = 'o' if out[i] == gt[i] else 'x'
        ff.append(fff)
    return ff


def get_model():
    return Model(extra_dense=True, block_num=8,
                 freeze_bert=_cfg_.freeze_bert,
                 ab_freeze_layer_count=_cfg_.freeze_layer_count,
                 bert=_cfg_.bert_name)

# model = nn.DataParallel(model)
model = get_model()
model.to(_cfg_.device)

## BCR
fdir_tst_neu_0104 = _cfg_.root_dir + f'Benchmark_rbd_neu_0104.xlsx'
data_test_neu_0104 = pd.read_excel(fdir_tst_neu_0104)
test_name_0104 = 'TEST'
test_set_neu_0104 = Ab_Dataset(datalist=[data_test_neu_0104], proportions=[None], sample_func=['sample'],
                                n_samples=data_test_neu_0104.shape[0], is_rand_sample=False,
                                onehot=_cfg_.use_onehot, rand_shift=False)
test_loader_neu_0104 = DataLoader_n(dataset=test_set_neu_0104, batch_size=_cfg_.batch_sz,
                                     num_workers=0, shuffle=False)
model_dirs = [
'rslt-meta_XBCR_ACNN_0905-small-abag3-acc-[1, 2, 3, 1, 1.2]-semi_sars-neu_[1, 2, 3, 1, 1.2]_fold0_metaacc-semi/epoch_38.pth',
'rslt-meta_XBCR_ACNN_0905-small-abag3-acc-[1, 2, 3, 1, 1.2]-semi_sars-neu_[1, 2, 3, 1, 1.2]_fold1_metaacc-semi/epoch_39.pth',
'rslt-meta_XBCR_ACNN_0905-small-abag3-acc-[1, 2, 3, 1, 1.2]-semi_sars-neu_[1, 2, 3, 1, 1.2]_fold2_metaacc-semi/epoch_41.pth',
'rslt-meta_XBCR_ACNN_0905-small-abag3-acc-[1, 2, 3, 1, 1.2]-semi_sars-neu_[1, 2, 3, 1, 1.2]_fold3_metaacc-semi/epoch_39.pth',
'rslt-meta_XBCR_ACNN_0905-small-abag3-acc-[1, 2, 3, 1, 1.2]-semi_sars-neu_[1, 2, 3, 1, 1.2]_fold4_metaacc-semi/epoch_41.pth',
            ]
model_names = [
'0905-small-semi_sars-neu_fold0_38',
'0905-small-semi_sars-neu_fold1_39',
'0905-small-semi_sars-neu_fold2_41',
'0905-small-semi_sars-neu_fold3_39',
'0905-small-semi_sars-neu_fold4_41',

               ]

for model_name, model_dir in zip(model_names, model_dirs):
    print('Testing `{}`'.format(model_dir))
    # test implement
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader_neu_0104, )
    #model.train()

    data_test_neu_0104['output'] = np.around(np.array(predictions_tst)).tolist()
    data_test_neu_0104['predict'] = predictions_tst
    data_test_neu_0104.to_excel(
        _cfg_.root_dir + f"{_RESULT_DIR}/{test_name_0104}_{model_name}_rbd_neu_0104_test.xlsx",
        index=False,
        header=True)