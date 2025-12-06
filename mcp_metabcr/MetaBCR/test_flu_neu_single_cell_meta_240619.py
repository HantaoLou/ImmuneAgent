import os
from torch.utils.data import DataLoader as DataLoader_n
import pandas as pd
from MetaBCR.lm_gnn_model_jz0508 import XBCR_ACNN_dense_meta as Model
from MetaBCR.lm_gnn_model_jz0508 import get_frozen_bert
import argparse

from dataset import Ab_Dataset  # Ab_Dataset_augment, Ab_Dataset_augment_cross, Ab_Dataset_wo_label
from MetaBCR.losses import *

parser = argparse.ArgumentParser()

# HYPER PARAM
_train_mode = 'neu'  # 'sars+neu', 'sars', 'neu'
_model = 'XBCR_ACNN'  # 'XBCR_ACNN', 'DeepBCR_ACEXN_protbert'
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

_RESULT_DIR = _root_dir + '240619_flu_neu_results/'  # output dir
if not os.path.exists(_RESULT_DIR):
    print('Cannot find [RESULT DIR], created a new one.')
    os.makedirs(_RESULT_DIR)
print('[RESULT DIR]: {}'.format(_RESULT_DIR))


def implement(model, dataloader, wolabel=False, ab_bert=None):
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
            input_ids_ab_v = data['input_ids_ab_v'].to(_device)
            attention_mask_ab_v = data['attention_mask_ab_v'].to(_device)
            input_ids_ab_l = data['input_ids_ab_l'].to(_device)
            attention_mask_ab_l = data['attention_mask_ab_l'].to(_device)
            input_ids_ag = data['input_ids_ag'].to(_device)
            # prediction
            outputs = model(ag_x=input_ids_ag, ab_x=input_ids_ab_v,
                            attention_mask_ab_v=attention_mask_ab_v,
                            ab_l=input_ids_ab_l,
                            attention_mask_ab_l=attention_mask_ab_l, ab_bert=ab_bert)
            # loss
            predictions_main_tr.extend(outputs[0].cpu().view(-1).tolist())
            if wolabel == False:
                labels_main_tr.extend(data['label'].view(-1).tolist())
                lossweight_main_tr.extend(data['loss_main'].view(-1).tolist())
    if wolabel:
        return predictions_main_tr
    return predictions_main_tr, labels_main_tr, lossweight_main_tr


def font(out, gt):
    # this is for visualizing the correctness of predictions,
    # 'o' for correct predictions, 'x' for false predictions.
    ff = []
    for i in range(len(out)):
        fff = 'o' if out[i] == gt[i] else 'x'
        ff.append(fff)
    return ff


ab_bert = get_frozen_bert("prot_bert")
ab_bert.to(_device)

model = Model(extra_dense=True, block_num=8, freeze_bert=True, ab_freeze_layer_count=True)
# model = nn.DataParallel(model)
model.to(_device)

## BCR
# _fdir_tst_flu = _root_dir + "1025_all_merged_QIV_bcrs.xlsx"
# test_name = 'BCR'
_fdir_tst_flu = _root_dir + "data/Benchmark_flu_bind_240621_clone.xlsx"
test_name = 'TEST-clone'
data_test_flu = pd.DataFrame()
data_test = pd.read_excel(_fdir_tst_flu)
# ags = pd.read_excel('./flu-antigen_to_predict1.xlsx')
ags = pd.read_excel('./flu-antigen_to_predict_clone.xlsx')

for i in ags['variant_name'].values:
    data_test['Antig Name'] = i
    data_test['variant_seq'] = ags.loc[ags['variant_name'] == i, 'variant_seq'].values[0]
    data_test_flu = pd.concat([data_test_flu, data_test], ignore_index=True)
data_test_flu['rbd'] = 0
test_set_flu = Ab_Dataset(datalist=[data_test_flu], proportions=[None], sample_func=['sample'],
                          n_samples=data_test_flu.shape[0], is_rand_sample=False,
                          onehot=_use_onehot, rand_shift=False)
test_loader_flu = DataLoader_n(dataset=test_set_flu, batch_size=_batch_sz, num_workers=0, shuffle=False)
model_dirs = [#"rslt-meta_XBCR_ACNN_0530-abag3-nopretrain-noclamp-extradense-acc_flu-neu_[22, 13, 0, 0]_fold0_metaacc/epoch_24.pth",
    "rslt-meta_XBCR_ACNN_0620-abag3-nopretrain-noclamp-extradense-acc2131_flu-neu_[2, 1, 3, 1]_fold2_metaacc/epoch_65.pth",
    "rslt-meta_XBCR_ACNN_0620-abag3-nopretrain-noclamp-extradense-acc2131_flu-neu_[2, 1, 3, 1]_fold3_metaacc/epoch_76.pth",
    "rslt-meta_XBCR_ACNN_0620-abag3-nopretrain-noclamp-extradense-acc2131_flu-neu_[2, 1, 3, 1]_fold4_metaacc/epoch_86.pth",


"rslt-meta_XBCR_ACNN_0621-abag3-finetuned-extradense-noclamp-acc-2211_flu-bind_[2, 2, 1, 1]_fold1_metaacc/epoch_69.pth",

            ]
model_names = [#'0530-nopretrain-221300_24_fold0'
               '0620-nopretrain-2131_65_fold2-neu',
'0620-nopretrain-2131_76_fold3-neu',
'0620-nopretrain-2131_86_fold4-neu',

"0621MetaBCR-finetuned-2211_fold1-bind"
               ]

for model_name, model_dir in zip(model_names, model_dirs):
    print('Testing `{}`'.format(model_dir))
    # test implement
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader_flu, ab_bert=ab_bert)

    data_test_flu['output'] = np.around(np.array(predictions_tst)).tolist()
    data_test_flu['predict'] = predictions_tst
    data_test_flu.to_csv(f"{_RESULT_DIR}{test_name}_{model_name}_flu_neut_test.csv")