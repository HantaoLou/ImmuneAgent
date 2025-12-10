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

from Config.config import get_config
from MetaBCR.lm_gnn_model_jz import XBCR_ACNN_woBERT_meta
from MetaBCR.lm_gnn_model_jz0508_unfrozen import XBCR_ACNN_dense_meta
from MetaBCR.lm_gnn_model_jz import DeepBCR_ACEXN_protbert
from MetaBCR.lm_gnn_model_jz import Adaptive_Regulariz
from MetaBCR.dataset_rsv import Ab_Dataset, Ab_Dataset_mean_teacher
# from MetaBCR.dataset_flu import Ab_Dataset, Ab_Dataset_mean_teacher
import MetaBCR.metrics as metrics
from MetaBCR.losses import *
from MetaBCR.lm_gnn_model_jz0508_unfrozen import Adaptive_Regulariz
import json


#################### put implement funcs here ####################

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


def init_weights(m):
    print('Initializing model weight ...')
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)  # he ini
        # init.orthogonal_(m.weight.data)  # orthogonal inic
        # init.orthogonal_(m.weight.data)  # xaiver / glorot ini
        # init.normal_(m.weight.data, mean=0, std=0.01)  # normal distribution ini
        if m.bias is not None: init.constant_(m.bias.data, 0.01)  # preventing zero bias


def font(out, gt):
    # this is for visualizing the correctness of predictions,
    # 'o' for correct predictions, 'x' for false predictions.
    ff = []
    for i in range(len(out)):
        fff = 'o' if out[i] == gt[i] else 'x'
        ff.append(fff)
    return ff


def implement(model, dataloader, _cfg_, wolabel=False):
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
            # print(f'INFO [ implement ] : label: {data["label"].view(-1).tolist()}')
    if wolabel: return predictions_main_tr
    return predictions_main_tr, labels_main_tr, lossweight_main_tr


def get_optimizer(net, _cfg_, state=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=_cfg_.lr, betas=(0, 0.999))
    if state is not None: optimizer.load_state_dict(state)
    return optimizer


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def model_deep_duplicate(model, Model, _cfg_):
    dup_model = get_model(Model=Model, _cfg_=_cfg_)
    dup_model.to(_cfg_.device)
    dup_model.load_state_dict(model.state_dict())
    return dup_model


def fast_learn(model, sup_criterion, fast_opt, _cfg_, teacher_model=None, unsup_criterion=None, X=None, Y=None,
               mask=None):
    model.train()
    fast_opt.zero_grad()
    outputs = model(ag_x=X[4], ab_x=X[0], attention_mask_ab_v=X[1],
                    ab_l=X[2], attention_mask_ab_l=X[3])
    intermediate_loss = sup_criterion(outputs[0], Y.view(-1, 1).float())

    if mask is None:
        loss = torch.mean(intermediate_loss)
    else:
        supervised_loss = mask * intermediate_loss
        inverse_mask = torch.ones_like(mask).to(_cfg_.device) - mask

        teacher_model.eval()
        teacher_outputs = teacher_model(ag_x=X[4], ab_x=X[5], attention_mask_ab_v=X[1],
                                        ab_l=X[6], attention_mask_ab_l=X[3])
        teacher_model.train()

        unsupervised_loss = inverse_mask * unsup_criterion(outputs[0], teacher_outputs[0])

        # print('sup: ', torch.mean(supervised_loss).item(), ', unsup: ', torch.mean(unsupervised_loss).item())

        loss = torch.mean(supervised_loss + _cfg_.unsup_loss_weight * unsupervised_loss)
        # print(supervised_loss, torch.mean(supervised_loss))
        # print(unsupervised_loss)
        # print(loss)
        # quit()

    # # regularize
    # loss_regulariz = model.module.get_variables()
    # loss += sum([w * x for w, x in zip(weight_regulariz, loss_regulariz)])
    # backward and optimize
    loss.backward()
    fast_opt.step()
    # state = fast_opt.state_dict()  # save fast optimizer state

    return loss


##################################################################

def train(num_fold=None, _cfg_=None, antigen_name='rsv',task_name='bind'):
    if num_fold == None:
        all_folds = [0, 1, 2, 3, 4]
        # all_folds = [2, 3, 4]
        print("INFO [ train ] : Got num_folds=None, will train all folds")
    else:
        all_folds = [num_fold]
        print(f"INFO [ train ] : Train fold #{num_fold}")

    if _cfg_.model == 'XBCR_ACNN':
        if _cfg_.use_onehot:
            Model = XBCR_ACNN_woBERT_meta
        else:
            Model = XBCR_ACNN_dense_meta
    elif _cfg_.model == 'DeepBCR_ACEXN_protbert':
        Model = DeepBCR_ACEXN_protbert
    else:
        print(f'ERROR [ train ] : Wrong model {_cfg_.model}')
        raise ValueError

    for fold in all_folds:
        excluded_folds = [i for i in all_folds if i != fold]
        #################### data path #####################

        # fdir_train_1 = [f'Data/RSV/0212_RSV_bind_absplit_pos_fold{f}.csv' for f in excluded_folds]
        # fdir_train_0 = [f'Data/RSV/0212_RSV_bind_absplit_neg_fold{f}.csv' for f in excluded_folds]
        fdir_train_1 = [f'{_cfg_.train_data}_pos_fold{f}.csv' for f in excluded_folds]
        fdir_train_0 = [f'{_cfg_.train_data}_neg_fold{f}.csv' for f in excluded_folds]

        fdir_train_nolabel = 'Data/1025_all_merged_QIV_bcrs.xlsx'

        fdir_val_1 = f'{_cfg_.train_data}_pos_fold{fold}.csv'
        fdir_val_0 = [f'{_cfg_.train_data}_neg_fold{fold}.csv',f'{_cfg_.valid_data}']

        # fdir_train_non_experiment = "Data/240408_nega_all_processed_data_for_train.xlsx"  ### 240316
        fdir_train_non_experiment = "Data/RSV/20250215_nega_all_processed_data_for_RSV.xlsx"  ### 250215
        
        fdir_train_non_experiment_sars = "Data/240314_neg_data_for_sars.xlsx"  ### 240316
        fdir_train_non_experiment_hiv = "Data/240314_neg_data_for_hiv.xlsx"  ### 240316
        fdir_train_non_experiment_flu = "Data/240314_neg_data_for_flu.xlsx"  ### 240316
        fdir_train_non_experiment_cross = [fdir_train_non_experiment_flu, fdir_train_non_experiment_sars, fdir_train_non_experiment_hiv]
        # fdir_tst = f'Data/20240510_rbd_flu_hiv_test_data/flu_unique_test_randomseed-{_cfg_.hiv_split_seed}.xlsx'
        fdir_tst = f'{_cfg_.test_data}'
        
        _fdir_tst_bind_BNT = "Data/Benchmark_flu_bind_0612.xlsx"
        _fdir_tst_bind_clone = "Data/Benchmark_flu_bind_240621_clone.xlsx"

        _RESULT_DIR = f'Results/{_cfg_.train_mode}/rslt-meta_{_cfg_.model}_{_cfg_.date}_{_cfg_.train_mode}_{_cfg_.prop}_fold{fold}_meta{_cfg_.benchmark}-semi/'


        if not os.path.exists(_RESULT_DIR):
            print('INFO [ train ] : Cannot find <RESULT DIR>, created a new one.')
            os.makedirs(_RESULT_DIR)
        print('INFO [ train ] : <RESULT DIR>: {}'.format(_RESULT_DIR))
        
        #################### config ####################
        # Save a copy of _cfg_ in _RESULT_DIR in json format
        cfg_dict = {k: v for k, v in _cfg_.__dict__.items() if not k.startswith('__') and not callable(v)}
        cfg_path = os.path.join(_RESULT_DIR, 'config.json')
        with open(cfg_path, 'w') as f:
            json.dump(cfg_dict, f, indent=4)
        #################### config ####################

        random.seed(_cfg_.rand_seed)

        #################### dataloader ####################

        data_train_1 = read_tables(fdir_train_1)
        data_train_0 = read_tables(fdir_train_0)

        data_train_non_experiment = read_tables(fdir_train_non_experiment)
        data_train_non_experiment_sars = read_tables(fdir_train_non_experiment_sars)
        data_train_non_experiment_flu = read_tables(fdir_train_non_experiment_flu)
        data_train_non_experiment_hiv = read_tables(fdir_train_non_experiment_hiv)
        data_train_non_experiment_cross = read_tables(fdir_train_non_experiment_cross)

        data_train_nolabel = read_tables(fdir_train_nolabel)

        train_set = Ab_Dataset_mean_teacher(datalist=[data_train_1,
                                                      data_train_0,
                                                      [data_train_0, data_train_non_experiment],
                                                      [data_train_0, data_train_non_experiment_cross],
                                                      [data_train_nolabel, data_train_1]
                                                      ],
                                            proportions=_cfg_.prop,
                                            sample_func=['rand_sample',
                                                         'rand_sample',
                                                         'rand_sample_rand_combine',
                                                         'rand_sample_rand_combine',
                                                         'no_label'
                                                         ],
                                            n_samples=max(data_train_1.shape[0] + data_train_0.shape[0], 1024),
                                            is_rand_sample=True, onehot=_cfg_.use_onehot, rand_shift=True,
                                            label_str=f"{task_name}_value"
                                            )

        # train_loader = DataLoader_n(dataset=train_set, batch_size=_batch_sz, num_workers=0, shuffle=False)
        train_loader = DataLoader_n(dataset=train_set, batch_size=_cfg_.batch_sz, shuffle=False)

        data_val_1 = read_tables(fdir_val_1)
        data_val_0 = read_tables(fdir_val_0)

        data_val = pd.concat([data_val_1, data_val_0], ignore_index=True)

        val_set = Ab_Dataset(datalist=[data_val], proportions=[None], sample_func=['sample'],
                             n_samples=data_val.shape[0], is_rand_sample=False, onehot=_cfg_.use_onehot,
                             rand_shift=False,label_str=f"{task_name}_value")
        val_loader = DataLoader_n(dataset=val_set, batch_size=_cfg_.batch_sz, shuffle=False)
        meta_loader = DataLoader_n(dataset=val_set, batch_size=_cfg_.batch_sz, shuffle=False)

        data_test = read_tables(fdir_tst)
        test_name = 'testset'
        test_set = Ab_Dataset(datalist=[data_test], proportions=[None], sample_func=['sample'],
                                  n_samples=data_test.shape[0], is_rand_sample=False,
                                  onehot=_cfg_.use_onehot, rand_shift=False,label_str=f"{task_name}_value")
        test_loader = DataLoader_n(dataset=test_set, batch_size=_cfg_.batch_sz, num_workers=0, shuffle=False)

        # data_test_bind_BNT = read_tables(_fdir_tst_bind_BNT)
        # test_name_BNT = 'TEST'
        # test_set_bind_BNT = Ab_Dataset(datalist=[data_test_bind_BNT], proportions=[None], sample_func=['sample'],
        #                                n_samples=data_test_bind_BNT.shape[0], is_rand_sample=False,
        #                                onehot=_cfg_.use_onehot, rand_shift=False)

        # test_loader_bind_BNT = DataLoader_n(dataset=test_set_bind_BNT, batch_size=_cfg_.batch_sz, num_workers=0,
        #                                     shuffle=False)
        # data_test_bind_clone = read_table(_fdir_tst_bind_clone)
        # test_name_clone = 'TEST-clone'
        # test_set_bind_clone = Ab_Dataset(datalist=[data_test_bind_clone], proportions=[None], sample_func=['sample'],
        #                                  n_samples=data_test_bind_clone.shape[0], is_rand_sample=False,
        #                                  onehot=_cfg_.use_onehot, rand_shift=False)

        # test_loader_bind_clone = DataLoader_n(dataset=test_set_bind_clone, batch_size=_cfg_.batch_sz, num_workers=0,
        #                                       shuffle=False)

        #################### train ####################

        print('INFO [ train ] : Start training ...')
        model = get_model(Model=Model, _cfg_=_cfg_)
        # model = nn.DataParallel(model)
        model.to(_cfg_.device)

        # load state dict
        if _cfg_.pretrain_model_dir is not None:
            pretrain_model_dir = os.path.join(_cfg_.pretrain_model_dir, f'fold{fold}.pth')
            model.load_state_dict(torch.load(pretrain_model_dir), strict=False)  # 0107
            print(f'INFO [ train ] : Loaded pretrain model params from {pretrain_model_dir}')
        else:
            # model.apply(init_weights)  # Yu 2024/1/4
            print('INFO [ train ] : No pretrained models, initialize model weights')

        supervise_criterion = nn.BCELoss(reduction='none')
        val_criterion = nn.BCELoss()
        unsupervise_criterion = nn.MSELoss(reduction='none')
        optimizer = torch.optim.SGD(model.parameters(), lr=_cfg_.lr)
        # adaptive regularize
        adaptive_regular = Adaptive_Regulariz(velocity=_cfg_.regul_v,
                                              target_deviation_ratio=_cfg_.regul_tgt_dev_rat)  # velocity=[0.02,0.001], target_deviation_ratio=0.08
        weight_regulariz_neu = adaptive_regular.weight

        # train the network
        if _cfg_.train_mode == 'sars+flu':
            _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = {'sars': 0, 'flu': 0}, {'sars': 0, 'flu': 0}, {
                'sars': 0, 'flu': 0}, {'sars': 0, 'flu': 0}
        elif _cfg_.train_mode == 'sars+flu+hiv':  ### 240314
            _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = {'sars': 0, 'flu': 0, 'hiv': 0}, {'sars': 0,
                                                                                                          'flu': 0,
                                                                                                          'hiv': 0}, {
                                                                            'sars': 0, 'flu': 0, 'hiv': 0}, {'sars': 0,
                                                                                                             'flu': 0,
                                                                                                             'hiv': 0}
        else:
            _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = 0, 0, 0, 0
        global_iter = 0

        # fast model
        # fast_model = model_deep_duplicate(model)
        fast_model = model.clone_meta(device=_cfg_.device)

        fast_optimizer = get_optimizer(fast_model, _cfg_, None)
        # fast_optimizer = torch.optim.Adam(fast_model.parameters(), lr=_lr, betas=(0, 0.999))
        # fast_optimizer = torch.optim.SGD(fast_model.parameters(), lr=_lr)
        train_acc = []
        train_loss = []
        train_epoch = []
        train_precision = []
        train_recall = []

        val_loss = []
        val_acc = []
        val_precision = []
        val_recall = []
        for epoch in range(_cfg_.num_epochs):
            train_running_loss = 0.0
            running_loss = 0.0  # for loss printing
            predictions_tr, labels_tr, has_label_mask_tr = [], [], []
            weight_regulariz = [weight_regulariz_neu, 0.]

            # meta_lr = _lr * 1e5 * (1.01 - epoch / float(_num_epochs))
            meta_lr = _cfg_.lr * 1e4 * (1.01 - epoch / float(_cfg_.num_epochs))
            set_learning_rate(optimizer, meta_lr)

            for i, data in enumerate(train_loader, 0):
                global_iter += 1
                
                # data pre-process
                input_ids_ab_v = data['input_ids_ab_v'].to(_cfg_.device)
                attention_mask_ab_v = data['attention_mask_ab_v'].to(_cfg_.device)
                input_ids_ab_l = data['input_ids_ab_l'].to(_cfg_.device)
                attention_mask_ab_l = data['attention_mask_ab_l'].to(_cfg_.device)
                input_ids_ag = data['input_ids_ag'].to(_cfg_.device)
                labels = data['label'].to(_cfg_.device)
                has_label_mask = data['has_label'].unsqueeze(1).to(_cfg_.device)
                input_ids_ab_v_origin = data['input_ids_ab_v_origin'].to(_cfg_.device)
                attention_mask_ab_v_origin = data['attention_mask_ab_v_origin'].to(_cfg_.device)
                input_ids_ab_l_origin = data['input_ids_ab_l_origin'].to(_cfg_.device)
                attention_mask_ab_l_origin = data['attention_mask_ab_l_origin'].to(_cfg_.device)

                # fast learning
                fast_loss = fast_learn(fast_model, supervise_criterion, fast_optimizer, _cfg_,
                                       model, unsupervise_criterion,
                                       X=[input_ids_ab_v, attention_mask_ab_v,
                                          input_ids_ab_l, attention_mask_ab_l,
                                          input_ids_ag,
                                          input_ids_ab_v_origin, attention_mask_ab_v_origin,
                                          input_ids_ab_l_origin, attention_mask_ab_l_origin],
                                       Y=labels, mask=has_label_mask)
                # update slow model
                if i % _cfg_.meta_update_iter == 0:
                    # zero the meta-parameter gradients
                    optimizer.zero_grad()

                    # # forward
                    outputs = model(ag_x=input_ids_ag, ab_x=input_ids_ab_v, attention_mask_ab_v=attention_mask_ab_v,
                                    ab_l=input_ids_ab_l, attention_mask_ab_l=attention_mask_ab_l)

                    # loss = criterion(outputs[0], labels.view(-1, 1).float())

                    # model.module.point_grad_to(fast_model)
                    # loss = 0.
                    # regularize
                    loss_regulariz = model.get_variables()
                    loss = sum([w * x for w, x in zip(weight_regulariz, loss_regulariz)])
                    # backward and optimize
                    loss.backward()
                    optimizer.step()
                    # fast MAML
                    optimizer.zero_grad()
                    model.point_grad_to(fast_model, _cfg_.device)  ####
                    optimizer.step()

                    # initialize fast model and fast optimizer
                    # # 0108b
                    # fast_model = model_deep_duplicate(model)
                    # fast_optimizer = torch.optim.Adam(fast_model.parameters(), lr=_lr, betas=(0, 0.999))
                    # # 0108c
                    # fast_model = model.clone_meta()
                    # fast_optimizer = torch.optim.SGD(fast_model.parameters(), lr=_lr)
                    # 0108d
                    fast_model.load_state_dict(model.state_dict())

                    # model = fast_model  ####
                    # fast_model = model.clone()  ####
                    # fast_model = model.module.clone()
                    # fast_model = copy.deepcopy(model.module)

                    # fast_optimizer = torch.optim.SGD(fast_model.parameters(), lr=_lr)
                    fast_state = fast_optimizer.state_dict()  # save fast optimizer state
                    fast_optimizer = get_optimizer(fast_model, _cfg_, state=fast_state)  ####

                    # store predictions and labels
                    predictions_tr.extend(outputs[0].cpu().view(-1).tolist())
                    labels_tr.extend(labels.view(-1).tolist())
                    has_label_mask_tr.extend(has_label_mask.view(-1).tolist())
                    # print statistics
                    running_loss += loss.item()

                running_loss += fast_loss.item()
                train_running_loss += fast_loss.item()
                if i % _cfg_.print_step == (_cfg_.print_step - 1):
                    print(
                        f'INFO [ train ] : Training '
                        f'Epoch {epoch} Iter {i + 1:5d}, '
                        f'loss: {running_loss / _cfg_.print_step:.6f}, '
                        f'wregul: {weight_regulariz_neu:.3f}')
                    running_loss = 0.0

                # meta
                if global_iter % _cfg_.regul_step == (_cfg_.regul_step - 1):
                    # get train confusion matrix
                    confusion_mat_tr = metrics.get_confusion_mat(predictions_tr, labels_tr, has_label_mask_tr)
                    eval_confusion_tr = metrics.eval_confusion(confusion_mat_tr)
                    # validation
                    model.eval()
                    predictions_val, labels_val, lossweight_val = implement(model, meta_loader, _cfg_)
                    model.train()
                    # get validate confusion matrix
                    confusion_mat_val = metrics.get_confusion_mat(
                        [predictions_val[i] for (i, v) in enumerate(lossweight_val) if v == 1],
                        [labels_val[i] for (i, v) in enumerate(lossweight_val) if v == 1])
                    eval_confusion_val = metrics.eval_confusion(confusion_mat_val)
                    # update regularize weight
                    # w_reg_neu_acc = adaptive_regular.update_weight(-eval_confusion_tr[0], -eval_confusion_val[0])
                    # w_reg_neu_ppv = adaptive_regular.update_weight(-eval_confusion_tr[3], -eval_confusion_val[3])
                    # weight_regulariz_neu = 0.5*w_reg_neu_acc + 0.5*w_reg_neu_ppv
                    if _cfg_.benchmark == 'acc':
                        weight_regulariz_neu = adaptive_regular.update_weight(-eval_confusion_tr[0],
                                                                              -eval_confusion_val[0])  ### 240314
                    else:
                        weight_regulariz_neu = adaptive_regular.update_weight(-eval_confusion_tr[6],
                                                                              -eval_confusion_val[6])  ### 240314

                    # weight_regulariz_neu = adaptive_regular.update_weight(-eval_confusion_tr[6], -eval_confusion_val[6])

            # get train confusion matrix
            confusion_mat_tr = metrics.get_confusion_mat(predictions_tr, labels_tr, has_label_mask_tr)
            eval_confusion_tr = metrics.eval_confusion(confusion_mat_tr)
            train_loss.append(train_running_loss / len(train_loader))
            train_epoch.append(epoch)
            train_acc.append(eval_confusion_tr[0])
            train_precision.append(eval_confusion_tr[3])
            train_recall.append(eval_confusion_tr[1])
            # validation
            model.eval()
            predictions_val, labels_val, lossweight_val = implement(model, val_loader, _cfg_)
            model.train()
            # get validate confusion matrix
            confusion_mat_val = metrics.get_confusion_mat(
                [predictions_val[i] for (i, v) in enumerate(lossweight_val) if v == 1],
                [labels_val[i] for (i, v) in enumerate(lossweight_val) if v == 1])
            eval_confusion_val = metrics.eval_confusion(confusion_mat_val)

            val_loss.append(
                val_criterion(torch.Tensor(predictions_val), torch.Tensor(labels_val)).item())
            val_acc.append(eval_confusion_val[0])
            val_precision.append(eval_confusion_val[3])
            val_recall.append(eval_confusion_val[1])

            print(f"INFO [ train ] : Validation "
                  f"Epoch {epoch}, acc: {eval_confusion_val[0]:.2f}, "
                  f"ppv: {eval_confusion_val[3]:.2f}, sns: {eval_confusion_val[1]:.2f}")
            if _cfg_.benchmark == 'acc':
                if (epoch >= _cfg_.saveaft or epoch == 0) and (eval_confusion_val[0] >= best_val_acc):
                    _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = epoch, eval_confusion_val[0], \
                                                                                eval_confusion_val[3], \
                                                                                eval_confusion_val[6]
                    torch.save(model.state_dict(), f'{_RESULT_DIR}epoch_{epoch}.pth')

                    model_name = f'{_cfg_.date}_{epoch}_fold{fold}-maml'
                    model.eval()
                    # predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader_bind_BNT, _cfg_)
                    predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader, _cfg_)
                    model.train()
                    data_test['output'] = np.around(np.array(predictions_tst)).tolist()
                    data_test['predict'] = predictions_tst
                    data_test.to_excel(
                        f"{_RESULT_DIR}{test_name}_{model_name}_{antigen_name}_{task_name}_tst.xlsx",
                        index=False, header=True)

                    # model.eval()
                    # predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader_bind_clone, _cfg_)
                    # model.train()
                    # data_test_bind_clone['output'] = np.around(np.array(predictions_tst)).tolist()
                    # data_test_bind_clone['predict'] = predictions_tst
                    # data_test_bind_clone.to_excel(
                    #     f"{_RESULT_DIR}{test_name_clone}_{model_name}_{antigen_name}_binding_test.xlsx",
                    #     index=False, header=True)
            if _cfg_.benchmark == 'f1':
                if (epoch >= _cfg_.saveaft or epoch == 0) and (eval_confusion_val[6] >= best_val_f1):
                    _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = epoch, eval_confusion_val[0], \
                                                                                eval_confusion_val[3], \
                                                                                eval_confusion_val[6]
                    torch.save(model.state_dict(), f'{_RESULT_DIR}epoch_{epoch}.pth')

                    model_name = f'{_cfg_.date}_{epoch}_fold{fold}-maml'
                    model.eval()
                    predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader, _cfg_)
                    model.train()
                    data_test['output'] = np.around(np.array(predictions_tst)).tolist()
                    data_test['predict'] = predictions_tst
                    data_test.to_excel(
                        f"{_RESULT_DIR}{test_name}_{model_name}_{antigen_name}_{task_name}_tst.xlsx",
                        index=False, header=True)
            else:
                if (epoch >= _cfg_.saveaft or epoch == 0) and (eval_confusion_val[-1] > best_val_f1):
                    _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = epoch, eval_confusion_val[0], \
                                                                                eval_confusion_val[3], \
                                                                                eval_confusion_val[6]
                    torch.save(model.state_dict(), f'{_RESULT_DIR}epoch_{epoch}.pth')

            train_file = pd.DataFrame({'epoch': train_epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                                       'train_precision': train_precision, 'train_recall': train_recall,
                                       'val_loss': val_loss, 'val_acc': val_acc,
                                       'val_precision': val_precision, 'val_recall': val_recall, })
            train_file.to_excel(f'{_RESULT_DIR}{model_name}_train_file.xlsx', index=False)
        print(f'INFO [ train ] : Finished Training, '
              f'best model from epoch {_best_val_epoch}, '
              f'f1 {best_val_f1:.2f}, acc {best_val_acc:.2f}, ppv {best_val_prec:.2f}')

        #################### test ####################


if __name__ == '__main__':
    antigen_name = 'rsv'
    task_name = 'bind'
    # task_name = 'neu'
    # config_date = '250212'
    # config_date = '250215'
    # config_date = '250216'
    # config_date = '250217'
    # config_date = '250218'
    config_date = '2502s18'
    configure = get_config(f"Config/config_five_fold_{antigen_name}_{task_name}_meta_{config_date}_semi_supervise.json")

    train(num_fold=None, _cfg_=configure,antigen_name=antigen_name,task_name=task_name)
