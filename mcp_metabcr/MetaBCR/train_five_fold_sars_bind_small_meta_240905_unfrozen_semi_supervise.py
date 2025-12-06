'''
Mean Teacher
Yu Chen
2024.06.21
'''
import argparse
import config_five_fold_sars_bind_small_meta_240905_unfrozen_semi_supervise as _cfg_

if _cfg_.model == 'XBCR_ACNN':
    if _cfg_.use_onehot:
        from MetaBCR.lm_gnn_model_jz import XBCR_ACNN_woBERT_meta as Model
    else:
        # from lm_gnn_model_jz import XBCR_ACNN_meta as Model
        from MetaBCR.lm_gnn_model_jz0508_unfrozen import XBCR_ACNN_dense_meta as Model
        from MetaBCR.lm_gnn_model_jz0508_unfrozen import Adaptive_Regulariz
elif _cfg_.model == 'DeepBCR_ACEXN_protbert':
    from MetaBCR.lm_gnn_model_jz import DeepBCR_ACEXN_protbert as Model
else:
    print('Wrong model {}'.format(_cfg_.model))
    raise ValueError

import os
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader as DataLoader_n
import pandas as pd
import random

from MetaBCR.lm_gnn_model_jz import Adaptive_Regulariz
from MetaBCR.dataset_sars import Ab_Dataset, Ab_Dataset_mean_teacher
from MetaBCR.metrics import *
from MetaBCR.losses import *

# from lm_gnn_model_jz0508 import get_frozen_bert, get_unfrozen_bert
from MetaBCR.lm_gnn_model_jz0508_unfrozen import Adaptive_Regulariz


#################### put implement funcs here ####################

def read_table(file):
    try:
        data = pd.read_csv(file)
    except:
        data = pd.read_excel(file)
    return data


def get_model():
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


def get_optimizer(net, state=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=_cfg_.lr, betas=(0, 0.999))
    if state is not None: optimizer.load_state_dict(state)
    return optimizer


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def model_deep_duplicate(model):
    dup_model = get_model()
    dup_model.to(_cfg_.device)
    dup_model.load_state_dict(model.state_dict())
    return dup_model


def fast_learn(model, sup_criterion, fast_opt, teacher_model=None, unsup_criterion=None, X=None, Y=None, mask=None):
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
        teacher_outputs = teacher_model(ag_x=X[4], ab_x=X[5], attention_mask_ab_v=X[6],
                                        ab_l=X[7], attention_mask_ab_l=X[8])
        teacher_model.train()

        unsupervised_loss = inverse_mask * unsup_criterion(outputs[0], teacher_outputs[0])

        # print('sup: ', torch.mean(supervised_loss).item(), ', unsup: ', torch.mean(unsupervised_loss).item())

        loss = torch.mean(supervised_loss + _cfg_.unsup_loss_weight*unsupervised_loss)
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

    return loss, torch.mean(supervised_loss), torch.mean(unsupervised_loss)


##################################################################

def train(num_fold=None):
    all_folds = [0, 2, 3, 1, 4] if num_fold==None else [num_fold]

    print('Start training fold # {} on device {}'.format(all_folds, _cfg_.device))

    for fold in all_folds:

        #################### data path #####################

        fdir_train_sars_1 = _cfg_.root_dir + f'data/20240611_rbd_flu_hiv_trainmeta_data/20240612-abag_sars-bind_trainmeta-A1A11_pos.xlsx'
        fdir_train_sars_0 = _cfg_.root_dir + f'data/20240611_rbd_flu_hiv_trainmeta_data/20240612-abag_sars-bind_trainmeta-A1A11_neg.xlsx'

        fdir_train_nolabel = _cfg_.root_dir + 'data/1222_merged_A1-A11_BCR.xlsx'

        fdir_val_sars_0 = fdir_train_sars_0
        fdir_val_sars_1 = fdir_train_sars_1

        fdir_train_non_experiment = _cfg_.root_dir + "data/240408_nega_all_processed_data_for_train.xlsx"  ### 240316
        fdir_train_non_experiment_sars = _cfg_.root_dir + "data/240314_neg_data_for_sars.xlsx"  ### 240316
        fdir_train_non_experiment_hiv = _cfg_.root_dir + "data/240314_neg_data_for_hiv.xlsx"  ### 240316
        fdir_train_non_experiment_flu = _cfg_.root_dir + "data/240314_neg_data_for_flu.xlsx"  ### 240316

        fdir_tst_sars = _cfg_.root_dir + f'data/20240510_rbd_flu_hiv_test_data/sars_unique_test_randomseed-{_cfg_.hiv_split_seed}.xlsx'
        fdir_tst_bind_0611 = _cfg_.root_dir + "data/Benchmark_rbd_bind_0611.xlsx"

        _RESULT_DIR = _cfg_.root_dir + 'rslt-meta_{}_{}_{}_{}_fold{}_meta{}-semi/'.format(_cfg_.model,
                                                                                     _cfg_.date,
                                                                                     _cfg_.train_mode,
                                                                                     _cfg_.prop,
                                                                                     fold,
                                                                                     _cfg_.benchmark)  # output dir

        if not os.path.exists(_RESULT_DIR):
            print('Cannot find [RESULT DIR], created a new one.')
            os.makedirs(_RESULT_DIR)
        print('[RESULT DIR]: {}'.format(_RESULT_DIR))

        random.seed(_cfg_.rand_seed)

        #################### dataloader ####################

        data_train_sars_1 = read_table(fdir_train_sars_1)
        data_train_sars_0 = read_table(fdir_train_sars_0)

        data_train_non_experiment = read_table(fdir_train_non_experiment)
        data_train_non_experiment_sars = read_table(fdir_train_non_experiment_sars)
        data_train_non_experiment_flu = read_table(fdir_train_non_experiment_flu)
        data_train_non_experiment_hiv = read_table(fdir_train_non_experiment_hiv)

        data_train_nolabel = read_table(fdir_train_nolabel)

        train_set = Ab_Dataset_mean_teacher(datalist=[data_train_sars_1,
                                                      data_train_sars_0,
                                                      [data_train_sars_1, data_train_non_experiment],
                                                      [data_train_sars_1, data_train_non_experiment_sars],
                                                      [data_train_nolabel, data_train_sars_1]
                                                      ],
                                            proportions=_cfg_.prop,
                                            sample_func=['rand_sample',
                                                         'rand_sample',
                                                         'rand_sample_rand_combine',
                                                         'rand_sample_rand_combine',
                                                         'no_label'
                                                         ],
                                            n_samples=max(data_train_sars_1.shape[0] + data_train_sars_0.shape[0],1024),
                                            is_rand_sample=True, onehot=_cfg_.use_onehot, rand_shift=True)

        # train_loader = DataLoader_n(dataset=train_set, batch_size=_batch_sz, num_workers=0, shuffle=False)
        train_loader = DataLoader_n(dataset=train_set, batch_size=_cfg_.batch_sz, shuffle=False)

        data_val_hiv_1 = read_table(fdir_val_sars_1)
        data_val_hiv_0 = read_table(fdir_val_sars_0)

        data_val = pd.concat([data_val_hiv_1, data_val_hiv_0], ignore_index=True)

        val_set = Ab_Dataset(datalist=[data_val], proportions=[None], sample_func=['sample'],
                             n_samples=data_val.shape[0], is_rand_sample=False, onehot=_cfg_.use_onehot,
                             rand_shift=False)
        # val_loader = DataLoader_n(dataset=val_set, batch_size=_batch_sz, num_workers=0, shuffle=False)
        val_loader = DataLoader_n(dataset=val_set, batch_size=_cfg_.batch_sz, shuffle=False)
        meta_loader = DataLoader_n(dataset=val_set, batch_size=_cfg_.batch_sz, shuffle=False)

        data_test_bind_0611 = pd.read_excel(fdir_tst_bind_0611)
        test_name_0611 = 'TEST_BIND_0611'
        test_set_bind_0611 = Ab_Dataset(datalist=[data_test_bind_0611], proportions=[None], sample_func=['sample'],
                                        n_samples=data_test_bind_0611.shape[0], is_rand_sample=False,
                                        onehot=_cfg_.use_onehot, rand_shift=False)
        test_loader_bind_0611 = DataLoader_n(dataset=test_set_bind_0611, batch_size=_cfg_.batch_sz,
                                             num_workers=0, shuffle=False)
        
        data_test_rbd = read_table(fdir_tst_sars)
        test_name = 'TEST_RBD'
        test_set_rbd = Ab_Dataset(datalist=[data_test_rbd], proportions=[None], sample_func=['sample'],
                                  n_samples=data_test_rbd.shape[0], is_rand_sample=False,
                                  onehot=_cfg_.use_onehot, rand_shift=False)
        test_loader_rbd = DataLoader_n(dataset=test_set_rbd, batch_size=_cfg_.batch_sz, num_workers=0, shuffle=False)

        #################### train ####################

        print('\n\n Start training...')
        model = get_model()
        # model = nn.DataParallel(model)
        model.to(_cfg_.device)

        # load state dict
        if _cfg_.pretrain_model_dir is not None:
            # model.load_state_dict(torch.load(_pretrain_model_dir))
            model.load_state_dict(torch.load(_cfg_.pretrain_model_dir.format(fold)), strict=False)  # 0107
        else:
            model.apply(init_weights)  # Yu 2024/1/4
        # else:
        #     torch.nn.init.normal_(model.weight,0.00001,0.1**7)
        # define a loss function and an optimizer
        supervise_criterion = nn.BCELoss(reduction='none')
        val_criterion = nn.BCELoss()
        unsupervise_criterion = nn.MSELoss(reduction='none')
        # optimizer = optim.Adam(model.parameters(), lr=_lr)
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

        fast_optimizer = get_optimizer(fast_model, None)
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
            running_loss = 0.0  # for loss printing
            predictions_tr, labels_tr = [], []
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
                fast_loss, fast_suploss, fast_unsuploss = fast_learn(fast_model, supervise_criterion, fast_optimizer,
                                                                       model, unsupervise_criterion,
                                                                       X=[input_ids_ab_v, attention_mask_ab_v,
                                                                          input_ids_ab_l, attention_mask_ab_l,
                                                                          input_ids_ag,
                                                                          input_ids_ab_v_origin, attention_mask_ab_v_origin, 
                                                                          input_ids_ab_l_origin, attention_mask_ab_l_origin],
                                                                       Y=labels, mask=has_label_mask)
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
                    fast_optimizer = get_optimizer(fast_model, state=fast_state)  ####

                    # store predictions and labels
                    predictions_tr.extend(outputs[0].cpu().view(-1).tolist())
                    labels_tr.extend(labels.view(-1).tolist())
                    # print statistics
                    running_loss += loss.item()

                running_loss += fast_loss.item()

                if i % _cfg_.print_step == (_cfg_.print_step - 1):
                    print(
                        f'[{epoch}, {i + 1:5d}] loss: {running_loss / _cfg_.print_step:.6f} wregul: {weight_regulariz_neu:.3f} | '
                        f'suploss (this iter): {fast_suploss.item():4f} unsuploss (this iter): {fast_unsuploss.item():4f}')
                    running_loss = 0.0

                # meta
                if global_iter % _cfg_.regul_step == (_cfg_.regul_step - 1):
                    # get train confusion matrix
                    confusion_mat_tr = get_confusion_mat(predictions_tr, labels_tr)
                    eval_confusion_tr = eval_confusion(confusion_mat_tr)
                    # validation
                    model.eval()
                    predictions_val, labels_val, lossweight_val = implement(model, meta_loader)
                    model.train()
                    # get validate confusion matrix
                    confusion_mat_val = get_confusion_mat(
                        [predictions_val[i] for (i, v) in enumerate(lossweight_val) if v == 1],
                        [labels_val[i] for (i, v) in enumerate(lossweight_val) if v == 1])
                    eval_confusion_val = eval_confusion(confusion_mat_val)
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
            confusion_mat_tr = get_confusion_mat(predictions_tr, labels_tr)
            eval_confusion_tr = eval_confusion(confusion_mat_tr)
            train_loss.append(running_loss / len(train_loader))
            train_epoch.append(epoch)
            train_acc.append(eval_confusion_tr[0])
            train_precision.append(eval_confusion_tr[3])
            train_recall.append(eval_confusion_tr[1])
            # validation
            model.eval()
            predictions_val, labels_val, lossweight_val = implement(model, val_loader)
            model.train()
            # get validate confusion matrix
            confusion_mat_val = get_confusion_mat(
                [predictions_val[i] for (i, v) in enumerate(lossweight_val) if v == 1],
                [labels_val[i] for (i, v) in enumerate(lossweight_val) if v == 1])
            eval_confusion_val = eval_confusion(confusion_mat_val)

            val_loss.append(
                val_criterion(torch.Tensor(predictions_val), torch.Tensor(labels_val)).item())
            val_acc.append(eval_confusion_val[0])
            val_precision.append(eval_confusion_val[3])
            val_recall.append(eval_confusion_val[1])

            print(f"[Val] epoch{epoch}, acc: {eval_confusion_val[0]:.2f}, "
                  f"ppv: {eval_confusion_val[3]:.2f}, sns: {eval_confusion_val[1]:.2f}")
            if _cfg_.benchmark == 'acc':
                if (epoch >= _cfg_.saveaft or epoch == 0) and (eval_confusion_val[0] > best_val_acc):
                    _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = epoch, eval_confusion_val[0], \
                                                                                eval_confusion_val[3], \
                                                                                eval_confusion_val[6]
                    torch.save(model.state_dict(), f'{_RESULT_DIR}epoch_{epoch}.pth')

                    ### 测试集unseen
                    # '0530-abag3-nopretrain-noclamp-extradense-unfrozenbert-acc'
                    model_name = f'{_cfg_.date}_{epoch}_fold{fold}-maml'

                    model.eval()
                    predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader_bind_0611)
                    model.train()

                    data_test_bind_0611['output'] = np.around(np.array(predictions_tst)).tolist()
                    data_test_bind_0611['predict'] = predictions_tst
                    data_test_bind_0611.to_excel(
                        _cfg_.root_dir + f"{_cfg_.date_step}_rbd_bind_results_semi/{test_name_0611}_{model_name}_rbd_binding_test.xlsx",
                        index=False,
                        header=True)

                    model.eval()
                    predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader_rbd, )
                    model.train()

                    data_test_rbd['output'] = np.around(np.array(predictions_tst)).tolist()
                    data_test_rbd['predict'] = predictions_tst
                    data_test_rbd.to_excel(f"{_cfg_.date_step}_rbd_bind_results_semi/{test_name}_{model_name}_rbd_binding_test.xlsx",
                                           index=False, header=True)
            if _cfg_.benchmark == 'f1':
                if (epoch >= _cfg_.saveaft or epoch == 0) and (eval_confusion_val[6] > best_val_f1):
                    _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = epoch, eval_confusion_val[0], \
                                                                                eval_confusion_val[3], \
                                                                                eval_confusion_val[6]
                    torch.save(model.state_dict(), f'{_RESULT_DIR}epoch_{epoch}.pth')

                    model_name = f'{_cfg_.date}_{epoch}_fold{fold}-maml'
                    model.eval()
                    predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader_bind_0611)
                    model.train()

                    data_test_bind_0611['output'] = np.around(np.array(predictions_tst)).tolist()
                    data_test_bind_0611['predict'] = predictions_tst
                    data_test_bind_0611.to_excel(
                        _cfg_.root_dir + f"{_cfg_.date_step}_rbd_bind_results_semi/{test_name_0611}_{model_name}_rbd_binding_test.xlsx",
                        index=False,
                        header=True)

                    model.eval()
                    predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader_rbd, )
                    model.train()

                    data_test_rbd['output'] = np.around(np.array(predictions_tst)).tolist()
                    data_test_rbd['predict'] = predictions_tst
                    data_test_rbd.to_excel(
                        f"{_cfg_.date_step}_rbd_bind_results_semi/{test_name}_{model_name}_rbd_binding_test.xlsx",
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
            train_file.to_excel(f'{_RESULT_DIR}train_file.xlsx', index=False)
        print(f'Finished Training, best model from epoch {_best_val_epoch}, '
              f'f1 {best_val_f1:.2f}, acc {best_val_acc:.2f}, ppv {best_val_prec:.2f}.')
        print('... ...')

        #################### test ####################


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, help="# of fold")
    args = parser.parse_args()

    train(args.fold)
