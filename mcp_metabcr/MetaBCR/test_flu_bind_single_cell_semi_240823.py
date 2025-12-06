import os
from torch.utils.data import DataLoader as DataLoader_n
import pandas as pd
import argparse
from dataset_flu import Ab_Dataset
from MetaBCR.losses import *

import config_five_fold_flu_bind_meta_240613_unfrozen_semi_supervise as _cfg_

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

# from lm_gnn_model_jz0508 import get_frozen_bert, get_unfrozen_bert

parser = argparse.ArgumentParser()

# HYPER PARAM
_device = torch.device('cuda')  # train params
_batch_sz = 2048
_use_onehot = False
# _use_onehot = True
_lr = 0.00001
_print_step = 20
_regul_step = 100
_regul_v = [0.02, 0.001]
_regul_tgt_dev_rat = 0.08
_best_val_epoch = 125
_rand_seed = 2023

# _root_dir = './'
_root_dir = '/home/jachin/data/jzheng/bcr_semi_supervise_alpha/'

_RESULT_DIR = _root_dir + 'H5N1_test_with_flu_bind/'  # output dir
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
            print(f'Implementation - Processing Batch {i+1} / {len(dataloader)}')
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


bert_name = 'prot_bert'
model = Model(extra_dense=True, block_num=8,
                     freeze_bert=True, ab_freeze_layer_count=30,
                     bert=bert_name
                     )

# model = nn.DataParallel(model)
model.to(_device)

## BCR
# _fdir_tst_flu = _root_dir + "1025_all_merged_QIV_bcrs.xlsx"
_fdir_tst_flu = _root_dir + "H5N1_data/H5N1_potential_Ab(super)_ANARCI(1).csv"
test_name = 'BCR'
# _fdir_tst_flu = _root_dir + "data/Benchmark_flu_bind_240621_clone.xlsx"
# test_name = 'TEST-clone'
data_test_flu = pd.DataFrame()
data_test = pd.read_csv(_fdir_tst_flu)
ags = pd.read_excel(_root_dir + 'data/flu-antigen_to_predict1.xlsx')
# ags = pd.read_excel('./flu-antigen_to_predict_clone.xlsx')
for i in ags['variant_name'].values:
    data_test['Antig Name'] = i
    data_test['variant_seq'] = ags.loc[ags['variant_name'] == i, 'variant_seq'].values[0]
    data_test_flu = pd.concat([data_test_flu, data_test], ignore_index=True)
data_test_flu['rbd'] = 0


test_set_flu = Ab_Dataset(datalist=[data_test_flu], proportions=[None], sample_func=['sample'],
                             n_samples=data_test_flu.shape[0], is_rand_sample=False, onehot=_cfg_.use_onehot,
                             rand_shift=False)
test_loader_flu = DataLoader_n(dataset=test_set_flu, batch_size=_batch_sz, shuffle=False)
model_dirs = [#'rslt-meta_XBCR_ACNN_0818-updt_itr=10-epoch=200-f1-w_unsup_loss=1-abag3-4300(1.4)-semi_flu-bind_[4, 3, 0, 0, 1.4]_fold3_metaf1-semi/epoch_77.pth'
              #'rslt-meta_XBCR_ACNN_0818-updt_itr=10-epoch=200-f1-w_unsup_loss=1-abag3-4300(1.4)-semi_flu-bind_[4, 3, 0, 0, 1.4]_fold0_metaf1-semi/epoch_73.pth'
                # 'rslt-meta_XBCR_ACNN_0818-updt_itr=10-epoch=200-f1-w_unsup_loss=1-abag3-4300(1.4)-semi_flu-bind_[4, 3, 0, 0, 1.4]_fold2_metaf1-semi/epoch_113.pth'
    # 'rslt-meta_XBCR_ACNN_0805-updt_itr=10-epoch=200-f1-w_unsup_loss=10-abag3-4300(1.4)-semi_flu-bind_[4, 3, 0, 0, 1.4]_fold3_metaf1-semi/epoch_160.pth'
    # 'rslt-meta_XBCR_ACNN_0822-bigdata-updt_itr=5-epoch=100-f1-w_unsup_loss=1-abag3-nopretrain-extradense-noclamp-acc-2211(1.2)-semi_flu-bind_[2, 2, 1, 1, 1.2]_fold0_metaacc-semi/epoch_93.pth'
# 'rslt-meta_XBCR_ACNN_0822-bigdata-updt_itr=5-epoch=100-f1-w_unsup_loss=1-abag3-nopretrain-extradense-noclamp-acc-2211(1.2)-semi_flu-bind_[2, 2, 1, 1, 1.2]_fold1_metaacc-semi/epoch_77.pth'
# 'rslt-meta_XBCR_ACNN_0822-bigdata-updt_itr=5-epoch=100-f1-w_unsup_loss=1-abag3-nopretrain-extradense-noclamp-acc-2211(1.2)-semi_flu-bind_[2, 2, 1, 1, 1.2]_fold2_metaacc-semi/epoch_9.pth'
'rslt-meta_XBCR_ACNN_0808-updt_itr=10-epoch=200-f1-w_unsup_loss=10-abag3-4300(1.4)-semi_flu-bind_[4, 3, 0, 0, 1.4]_fold3_metaf1-semi/epoch_174.pth'
            ]
model_names = [
#"0818semi_fold3_77"
   # '0818semi_fold0_73'
# '0818semi_fold2_113'
# '0805semi_fold3_160'
# '0822semi-bigdata_fold0_93'
# '0822semi-bigdata_fold1_77'
# '0822semi-bigdata_fold2_9'
'test_H5N1'
               ]


for model_name, model_dir in zip(model_names, model_dirs):
    print('Testing `{}`'.format(model_dir))
    # test implement
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader_flu, )

    data_test_flu['output'] = np.around(np.array(predictions_tst)).tolist()
    data_test_flu['predict'] = predictions_tst
    data_test_flu.to_csv(f"{_RESULT_DIR}{test_name}_{model_name}_flu_binding_test.csv")