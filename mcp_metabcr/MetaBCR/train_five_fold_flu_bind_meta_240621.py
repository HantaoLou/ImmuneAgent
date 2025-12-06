'''
XBCR + meta learning
Yu Chen
2023.09.05
'''
import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader as DataLoader_n
import pandas as pd
import random

# date = '0614-abag3-finetunedbatch-extradense-noclamp-acc-unfrozenbert-1221'
# date = '0614-abag3-finetunedbatch-extradense-noclamp-acc-unfrozenbert-Influenza-1221'
# date = '0620-abag3-finetunedbatch-extradense-noclamp-acc-unfrozenbert30-4400' #-newneg
# date = '0621-abag3-finetuned-extradense-noclamp-acc-2211'
date = '0621-abag3-nopretrain-extradense-noclamp-acc-2211'
_prop = [ 2, 2, 1, 1,]
if 'unfrozenbert' in date:
    _batch_sz = 64
    _freeze_bert = False
    # _freeze_layer_count = 20
    _freeze_layer_count = 30
else:
    _batch_sz = 128
    _freeze_bert = True
    _freeze_layer_count = 100
_num_epochs = 100
# _num_epochs = 50
# DATA DIR
# _fold = 0
# hiv_split_seed = 339
_benchmark = 'acc'
hiv_split_seed = date.split('abag')[1].split('-')[0]
# meta_split_seed = date.split('trainmeta')[1].split('-')[0]
_root_dir = './'

# HYPER PARAM
# _train_mode = 'sars+flu'  # 'flu'  # , 'sars', 'flu'
_train_mode = 'flu-bind'  # 'flu'  # , 'sars', 'flu'        #0330 by jzheng
_model = 'XBCR_ACNN'  # 'XBCR_ACNN', 'DeepBCR_ACEXN_protbert'
_device = torch.device('cuda')  # train params
_use_onehot = False
# _use_onehot = True
if 'nopretrain' in date:
    # _lr = 0.0001
    _lr = 0.00001
    _saveaft = 5
else:
    _lr = 0.00001
    _saveaft = 0
# _lr = 0.000001
_print_step = 20
_regul_step = 100
_regul_v = [0.02, 0.001]
# _regul_v = [0.0002, 0.00001]
_regul_tgt_dev_rat = 0.08
_best_val_epoch = 125
_rand_seed = 2023

if _model == 'XBCR_ACNN':
    if _use_onehot:
        from MetaBCR.lm_gnn_model_jz import XBCR_ACNN_woBERT_meta as Model
    else:
        # from lm_gnn_model_jz import XBCR_ACNN_meta as Model
        from MetaBCR.lm_gnn_model_jz0508_unfrozen import XBCR_ACNN_dense_meta as Model
        from MetaBCR.lm_gnn_model_jz0508_unfrozen import Adaptive_Regulariz
elif _model == 'DeepBCR_ACEXN_protbert':
    from MetaBCR.lm_gnn_model_jz import DeepBCR_ACEXN_protbert as Model
else:
    print('Wrong model {}'.format(_model))
    raise ValueError
from dataset import Ab_Dataset  # Ab_Dataset_augment, Ab_Dataset_augment_cross, Ab_Dataset_wo_label
from MetaBCR.metrics import *
from MetaBCR.losses import *

# ARGUMENTS




# for _fold in [4,]:
for _fold in [ 0,]:
    if 'nopretrain' in date:
        _pretrain_model_dir = None
    else:
        _pretrain_model_dir = f'0612-flu-bind/fold{_fold}.pth'

    _fdir_train_flu_1 = f'data/20240611_rbd_flu_hiv_trainmeta_data/20240621-abag_flu_trainmeta_pos_fold{_fold}_randomseed-3.xlsx'
    _fdir_train_flu_0 = f'data/20240611_rbd_flu_hiv_trainmeta_data/20240621-abag_flu_trainmeta_neg_fold{_fold}_randomseed-3.xlsx'

    _fdir_val_flu_0 = f'data/20240611_rbd_flu_hiv_trainmeta_data/20240621-abag_flu_valmeta_neg_fold{_fold}_randomseed-3.xlsx'
    _fdir_val_flu_1 = f'data/20240611_rbd_flu_hiv_trainmeta_data/20240621-abag_flu_valmeta_pos_fold{_fold}_randomseed-3.xlsx'

    # _fdir_train_non_experiment = _root_dir + "data/240408_nega_all_processed_data_for_train.xlsx"  ### 240316
    _fdir_train_non_experiment = _root_dir + "data/240618_nega_forflu_processed_data.xlsx"  ### 240316
    _fdir_train_non_experiment_sars = _root_dir + "data/240314_neg_data_for_sars.xlsx" ### 240316
    _fdir_train_non_experiment_hiv = _root_dir + "data/240314_neg_data_for_hiv.xlsx" ### 240316
    _fdir_train_non_experiment_flu = _root_dir + "data/240314_neg_data_for_flu.xlsx" ### 240316
    _fdir_tst_flu = _root_dir + f'data/20240510_rbd_flu_hiv_test_data/flu_unique_test_randomseed-{hiv_split_seed}.xlsx'

    #


    _RESULT_DIR = _root_dir + 'rslt-meta_{}_{}_{}_{}_fold{}_meta{}/'.format(_model, date,_train_mode, _prop, _fold,_benchmark)  # output dir
    if not os.path.exists(_RESULT_DIR):
        print('Cannot find [RESULT DIR], created a new one.')
        os.makedirs(_RESULT_DIR)
    print('[RESULT DIR]: {}'.format(_RESULT_DIR))

    random.seed(_rand_seed)


    #################### dataloader ####################

    def read_table(file):
        try:
            data = pd.read_csv(file)
        except:
            data = pd.read_excel(file)
        return data


    data_train_flu_1 = read_table(_fdir_train_flu_1)
    data_train_flu_0 = read_table(_fdir_train_flu_0)

    data_train_non_experiment = read_table(_fdir_train_non_experiment)
    data_train_non_experiment_sars = read_table(_fdir_train_non_experiment_sars)
    data_train_non_experiment_flu = read_table(_fdir_train_non_experiment_flu)
    data_train_non_experiment_hiv = read_table(_fdir_train_non_experiment_hiv)

    train_set = Ab_Dataset(datalist=[data_train_flu_1,
                                     data_train_flu_0,
                                     [data_train_flu_1, data_train_non_experiment],
                                     [data_train_flu_1, data_train_non_experiment_flu],
                                     ],
                           proportions=_prop,
                           sample_func=['rand_sample',
                                        'rand_sample',
                                        'rand_sample_rand_combine',
                                        'rand_sample_rand_combine',
                                        ],
                           n_samples=max(data_train_flu_1.shape[0] + data_train_flu_0.shape[0],1024),
                           is_rand_sample=True, onehot=_use_onehot, rand_shift=True)


    # train_loader = DataLoader_n(dataset=train_set, batch_size=_batch_sz, num_workers=0, shuffle=False)
    train_loader = DataLoader_n(dataset=train_set, batch_size=_batch_sz, shuffle=False)

    data_val_flu_1 = read_table(_fdir_val_flu_1)
    data_val_flu_0 = read_table(_fdir_val_flu_0)

    data_val = pd.concat([data_val_flu_1, data_val_flu_0], ignore_index=True)

    val_set = Ab_Dataset(datalist=[data_val], proportions=[None], sample_func=['sample'],
                         n_samples=data_val.shape[0], is_rand_sample=False, onehot=_use_onehot, rand_shift=False)
    # val_loader = DataLoader_n(dataset=val_set, batch_size=_batch_sz, num_workers=0, shuffle=False)
    val_loader = DataLoader_n(dataset=val_set, batch_size=_batch_sz, shuffle=False)
    meta_loader = DataLoader_n(dataset=val_set, batch_size=_batch_sz, shuffle=False)

    data_test_flu = read_table(_fdir_tst_flu)
    test_name = 'testset'
    test_set_flu = Ab_Dataset(datalist=[data_test_flu], proportions=[None], sample_func=['sample'],
                              n_samples=data_test_flu.shape[0], is_rand_sample=False,
                              onehot=_use_onehot, rand_shift=False)
    test_loader_flu = DataLoader_n(dataset=test_set_flu, batch_size=_batch_sz, num_workers=0, shuffle=False)

    _fdir_tst_bind_BNT = _root_dir + "data/Benchmark_flu_bind_0612.xlsx"
    data_test_bind_BNT = read_table(_fdir_tst_bind_BNT)
    test_name_BNT = 'TEST'
    test_set_bind_BNT = Ab_Dataset(datalist=[data_test_bind_BNT], proportions=[None], sample_func=['sample'],
                                   n_samples=data_test_bind_BNT.shape[0], is_rand_sample=False,
                                   onehot=_use_onehot, rand_shift=False)

    test_loader_bind_BNT = DataLoader_n(dataset=test_set_bind_BNT, batch_size=_batch_sz, num_workers=0, shuffle=False)

    _fdir_tst_bind_clone = _root_dir + "data/Benchmark_flu_bind_240621_clone.xlsx"
    data_test_bind_clone = read_table(_fdir_tst_bind_clone)
    test_name_clone = 'TEST-clone'
    test_set_bind_clone = Ab_Dataset(datalist=[data_test_bind_clone], proportions=[None], sample_func=['sample'],
                                   n_samples=data_test_bind_clone.shape[0], is_rand_sample=False,
                                   onehot=_use_onehot, rand_shift=False)

    test_loader_bind_clone = DataLoader_n(dataset=test_set_bind_clone, batch_size=_batch_sz, num_workers=0, shuffle=False)


    #################### put implement funcs here ####################

    def get_model():
        return Model(extra_dense=True, block_num=8,
                     freeze_bert=_freeze_bert, ab_freeze_layer_count=_freeze_layer_count,
                     bert=bert_name
                     )

    def init_weights(m):
        print('Initializing model weight ...')
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)  # he ini
            # init.orthogonal_(m.weight.data)  # orthogonal inic
            # init.orthogonal_(m.weight.data)  # xaiver / glorot ini
            # init.normal_(m.weight.data, mean=0, std=0.01)  # normal distribution ini
            if m.bias is not None:
                init.constant_(m.bias.data, 0.01)  # preventing zero bias


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
                input_ids_ab_v = data['input_ids_ab_v'].to(_device)
                attention_mask_ab_v = data['attention_mask_ab_v'].to(_device)
                input_ids_ab_l = data['input_ids_ab_l'].to(_device)
                attention_mask_ab_l = data['attention_mask_ab_l'].to(_device)
                input_ids_ag = data['input_ids_ag'].to(_device)
                # prediction
                outputs = model(ag_x=input_ids_ag, ab_x=input_ids_ab_v,
                                attention_mask_ab_v=attention_mask_ab_v,
                                ab_l=input_ids_ab_l,
                                attention_mask_ab_l=attention_mask_ab_l,)
                # loss
                predictions_main_tr.extend(outputs[0].cpu().view(-1).tolist())
                if wolabel == False:
                    labels_main_tr.extend(data['label'].view(-1).tolist())
                    lossweight_main_tr.extend(data['loss_main'].view(-1).tolist())
        if wolabel:
            return predictions_main_tr
        return predictions_main_tr, labels_main_tr, lossweight_main_tr


    def get_optimizer(net, state=None):
        optimizer = torch.optim.Adam(net.parameters(), lr=_lr, betas=(0, 0.999))
        if state is not None:
            optimizer.load_state_dict(state)
        return optimizer


    def set_learning_rate(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def model_deep_duplicate(model):
        dup_model = get_model()
        dup_model.to(_device)
        dup_model.load_state_dict(model.state_dict())
        return dup_model


    def fast_learn(model, fast_opt, X=None, Y=None, ):
        model.train()
        fast_opt.zero_grad()
        outputs = model(ag_x=X[4], ab_x=X[0], attention_mask_ab_v=X[1],
                        ab_l=X[2], attention_mask_ab_l=X[3],)
        loss = criterion(outputs[0], Y.view(-1, 1).float())
        # # regularize
        # loss_regulariz = model.module.get_variables()
        # loss += sum([w * x for w, x in zip(weight_regulariz, loss_regulariz)])
        # backward and optimize
        loss.backward()
        fast_opt.step()
        # state = fast_opt.state_dict()  # save fast optimizer state
        return loss


    meta_update_iter = 5

    #################### train ####################
    print('\n\n Start training...')
    # define a model
    # ab_bert = BertModel.from_pretrained("train_2_193999")
    #
    if 'Influenza' in date:
        bert_name = '240612_Influenza_epoch10'

    else:
        bert_name = 'prot_bert'


    model = get_model()
    # model = nn.DataParallel(model)
    model.to(_device)

    # load state dict
    if _pretrain_model_dir is not None:
        # model.load_state_dict(torch.load(_pretrain_model_dir))
        model.load_state_dict(torch.load(_pretrain_model_dir), strict=False)  # 0107
    else:
        model.apply(init_weights)  # Yu 2024/1/4
    # else:
    #     torch.nn.init.normal_(model.weight,0.00001,0.1**7)
    # define a loss function and an optimizer
    criterion = nn.BCELoss()
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=_lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=_lr)
    # adaptive regularize
    adaptive_regular = Adaptive_Regulariz(velocity=_regul_v,
                                          target_deviation_ratio=_regul_tgt_dev_rat)  # velocity=[0.02,0.001], target_deviation_ratio=0.08
    weight_regulariz_neu = adaptive_regular.weight

    # train the network
    if _train_mode == 'sars+flu':
        _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = {'sars': 0, 'flu': 0}, {'sars': 0, 'flu': 0}, {
            'sars': 0, 'flu': 0}, {'sars': 0, 'flu': 0}
    elif _train_mode == 'sars+flu+hiv':### 240314
        _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = {'sars': 0, 'flu': 0, 'hiv': 0}, {'sars': 0, 'flu': 0, 'hiv': 0}, {
            'sars': 0, 'flu': 0, 'hiv': 0}, {'sars': 0, 'flu': 0, 'hiv': 0}
    else:
        _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = 0, 0, 0, 0
    global_iter = 0

    # fast model
    # fast_model = model_deep_duplicate(model)
    fast_model = model.clone_meta()

    fast_optimizer = get_optimizer(fast_model,None)
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
    for epoch in range(_num_epochs):
        running_loss = 0.0  # for loss printing
        predictions_tr, labels_tr = [], []
        weight_regulariz = [weight_regulariz_neu, 0.]

        # meta_lr = _lr * 1e5 * (1.01 - epoch / float(_num_epochs))
        meta_lr = _lr * 1e4 * (1.01 - epoch / float(_num_epochs))
        set_learning_rate(optimizer, meta_lr)

        for i, data in enumerate(train_loader, 0):
            global_iter += 1
            # data pre-process
            input_ids_ab_v = data['input_ids_ab_v'].to(_device)
            attention_mask_ab_v = data['attention_mask_ab_v'].to(_device)
            input_ids_ab_l = data['input_ids_ab_l'].to(_device)
            attention_mask_ab_l = data['attention_mask_ab_l'].to(_device)
            input_ids_ag = data['input_ids_ag'].to(_device)
            labels = data['label'].to(_device)

            # fast learning
            fast_loss = fast_learn(fast_model, fast_optimizer,
                                   X=[input_ids_ab_v, attention_mask_ab_v,
                                      input_ids_ab_l, attention_mask_ab_l,
                                      input_ids_ag, ],
                                   Y=labels, )

            # update slow model
            if i % meta_update_iter == 0:
                # zero the meta-parameter gradients
                optimizer.zero_grad()

                # # forward
                # input_ids_ab_v = ab_bert(input_ids=input_ids_ab_v, attention_mask=attention_mask_ab_v).last_hidden_state
                # input_ids_ab_l = ab_bert(input_ids=input_ids_ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
                outputs = model(ag_x=input_ids_ag, ab_x=input_ids_ab_v, attention_mask_ab_v=attention_mask_ab_v,
                                ab_l=input_ids_ab_l, attention_mask_ab_l=attention_mask_ab_l,)

                # loss = criterion(outputs[0], labels.view(-1, 1).float())

                # model.module.point_grad_to(fast_model)
                # loss = 0.
                # regularize
                # loss_regulariz = model.module.get_variables()
                loss_regulariz = model.get_variables()
                loss = sum([w * x for w, x in zip(weight_regulariz, loss_regulariz)])
                # backward and optimize
                loss.backward()
                optimizer.step()
                # fast MAML
                optimizer.zero_grad()
                model.point_grad_to(fast_model)  ####
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
            if i % _print_step == (_print_step - 1):
                print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / _print_step:.3f} wregul: {weight_regulariz_neu:.3f}')
                running_loss = 0.0
            # meta
            if global_iter % _regul_step == (_regul_step - 1):
                # get train confusion matrix
                confusion_mat_tr = get_confusion_mat(predictions_tr, labels_tr)
                eval_confusion_tr = eval_confusion(confusion_mat_tr)
                # validation
                model.eval()
                predictions_val, labels_val, lossweight_val = implement(model, meta_loader, )
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
                if _benchmark == 'acc':
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
        predictions_val, labels_val, lossweight_val = implement(model, val_loader, )
        model.train()
        # get validate confusion matrix
        confusion_mat_val = get_confusion_mat(
            [predictions_val[i] for (i, v) in enumerate(lossweight_val) if v == 1],
            [labels_val[i] for (i, v) in enumerate(lossweight_val) if v == 1])
        eval_confusion_val = eval_confusion(confusion_mat_val)
        val_loss.append(
            BCE_loss(torch.Tensor(predictions_val).unsqueeze(1), torch.Tensor(labels_val)).item())
        val_acc.append(eval_confusion_val[0])
        val_precision.append(eval_confusion_val[3])
        val_recall.append(eval_confusion_val[1])

        print(f"[Val] epoch{epoch}, acc: {eval_confusion_val[0]:.2f}, "
              f"ppv: {eval_confusion_val[3]:.2f}, sns: {eval_confusion_val[1]:.2f}")
        if _benchmark == 'acc':
            if (epoch >= _saveaft or epoch == 0) and (eval_confusion_val[0] > best_val_acc):
                _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = epoch, eval_confusion_val[0], \
                                                                            eval_confusion_val[3], \
                                                                            eval_confusion_val[6]
                torch.save(model.state_dict(), f'{_RESULT_DIR}epoch.pth')

                model_name = f'{date}_fold{_fold}-maml'
                model.eval()
                predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader_bind_BNT,)
                data_test_bind_BNT['output'] = np.around(np.array(predictions_tst)).tolist()
                data_test_bind_BNT['predict'] = predictions_tst
                data_test_bind_BNT.to_excel(
                    f"0124_flu_bind_results/{test_name_BNT}_{model_name}_flu_binding_test.xlsx",
                    index=False, header=True)


                predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader_bind_clone, )
                data_test_bind_clone['output'] = np.around(np.array(predictions_tst)).tolist()
                data_test_bind_clone['predict'] = predictions_tst
                data_test_bind_clone.to_excel(
                    f"0124_flu_bind_results/{test_name_clone}_{model_name}_flu_binding_test.xlsx",
                    index=False, header=True)
                # predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader_flu, )
                # data_test_flu['output'] = np.around(np.array(predictions_tst)).tolist()
                # data_test_flu['predict'] = predictions_tst
                # data_test_flu.to_excel(f"0124_flu_bind_results/{test_name}_{model_name}_flu_binding_test.xlsx",
                #                        index=False, header=True)

                model.train()


        else:
            if (epoch >= _saveaft or epoch == 0) and (eval_confusion_val[6] > best_val_f1):
                _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = epoch, eval_confusion_val[0], \
                                                                            eval_confusion_val[3], \
                                                                            eval_confusion_val[6]
                # torch.save(model.state_dict(), f'{_RESULT_DIR}epoch_{epoch}.pth')
                torch.save(model.state_dict(), f'{_RESULT_DIR}epoch.pth')

        train_file = pd.DataFrame({'epoch': train_epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                               'train_precision': train_precision, 'train_recall': train_recall,
                               'val_loss': val_loss, 'val_acc': val_acc,
                               'val_precision': val_precision, 'val_recall': val_recall, })
        train_file.to_excel(f'{_RESULT_DIR}{model_name}_train_file.xlsx', index=False)
    os.rename(f'{_RESULT_DIR}epoch.pth', f'{_RESULT_DIR}epoch_{_best_val_epoch}.pth')

    print(f'Finished Training, best model from epoch {_best_val_epoch}, '
              f'f1 {best_val_f1:.2f}, acc {best_val_acc:.2f}, ppv {best_val_prec:.2f}.')
    print('... ...')

    #################### test ####################
    # print('Start Testing')
    # # define and load the best validation model
    # model = Model(extra_dense=True, block_num=8, freeze_bert=True, ab_freeze_layer_count=True)
    # model.to(_device)
    # ab_bert = get_frozen_bert("prot_bert")
    # ab_bert.to(_device)
    # print('Testing SARS')
    # best_val_epoch = _best_val_epoch['sars'] if _train_mode in ['sars+flu', 'sars+flu+hiv'] else _best_val_epoch  ### 240314
    # # model_dir = f'{_RESULT_DIR}epoch_{best_val_epoch}_sars.pth' if _train_mode == 'sars+flu' else f'{_RESULT_DIR}epoch_{best_val_epoch}.pth'### 240314
    # model_dir = f'{_RESULT_DIR}epoch_{best_val_epoch}_sars.pth' if _train_mode in ['sars+flu',
    #                                                                                'sars+flu+hiv'] else f'{_RESULT_DIR}epoch_{best_val_epoch}.pth'  ### 240314
    #
    # model.load_state_dict(torch.load(model_dir))
    # model.eval()
    #
    #
    # _fdir_tst_sars = _root_dir + f'data/20240521_rbd_flu_hiv_test_data/sars-neu_unique_test_randomseed-{hiv_split_seed}.xlsx'
    #
    # data_test_rbd = read_table(_fdir_tst_sars)
    # test_name = 'testset'
    # test_set_rbd = Ab_Dataset(datalist=[data_test_rbd], proportions=[None], sample_func=['sample'],
    #                           n_samples=data_test_rbd.shape[0], is_rand_sample=False,
    #                           onehot=_use_onehot, rand_shift=False)
    # test_loader_rbd = DataLoader_n(dataset=test_set_rbd, batch_size=_batch_sz, num_workers=0, shuffle=False)
    # predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader_rbd, ab_bert=ab_bert)
    #
    # data_test_rbd['output'] = np.around(np.array(predictions_tst)).tolist()
    # data_test_rbd['predict'] = predictions_tst
    # data_test_rbd.to_excel(f"{_RESULT_DIR}{test_name}_{model_name}_rbd_neutralization_test.xlsx", index=False, header=True)
