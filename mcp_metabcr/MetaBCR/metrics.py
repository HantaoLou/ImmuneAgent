from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np
import math
from math import sqrt


def get_confusion_mat(pred, gt, has_label_mask=None):
    # pred_logic = tf.cast(pred > 0.5, dtype='float32')
    # TP = tf.reduce_mean(pred_logic*gt)
    # TN = tf.reduce_mean((1-pred_logic)*(1-gt))
    # FP = tf.reduce_mean((pred_logic)*(1-gt))
    # FN = tf.reduce_mean((1 - pred_logic) * (gt))
    pred, gt = np.array(pred), np.array(gt)
    ones_target = np.ones(pred.shape)
    pred_logic = np.around(pred)

    if has_label_mask==None:
        TP = np.mean(pred_logic * gt)
        TN = np.mean((ones_target - pred_logic) * (ones_target - gt))
        FP = np.mean((pred_logic) * (ones_target - gt))
        FN = np.mean((ones_target - pred_logic) * (gt))
    else:
        has_label_mask = np.array(has_label_mask)
        num_of_label = np.sum(has_label_mask) + 1e-8
        TP = np.sum(has_label_mask * pred_logic * gt) / num_of_label
        TN = np.sum(has_label_mask * (ones_target - pred_logic) * (ones_target - gt)) / num_of_label
        FP = np.sum(has_label_mask * (pred_logic) * (ones_target - gt)) / num_of_label
        FN = np.sum(has_label_mask * (ones_target - pred_logic) * (gt)) / num_of_label

    return [TP, FP, FN, TN]


eval_name=['acc','sns','spc','ppv','npv','bac']
def eval_confusion(conf_mat,eps=1e-7):
    # consusion mat: [TP, FP, FN, TN]
    acc = np.sum(conf_mat[0] + conf_mat[3]) / (np.sum(conf_mat) + eps)
    sns = np.sum(conf_mat[0]) / np.sum(conf_mat[0] + conf_mat[2] + eps)
    spc = np.sum(conf_mat[3]) / np.sum(conf_mat[1] + conf_mat[3] + eps)
    ppv = np.sum(conf_mat[0]) / np.sum(conf_mat[0] + conf_mat[1] + eps)
    npv = np.sum(conf_mat[3]) / np.sum(conf_mat[2] + conf_mat[3] + eps)
    bac = (sns + spc) / 2
    f1 = 2*ppv*sns / (ppv+sns+eps)
    mcc = (conf_mat[3]*conf_mat[0]-conf_mat[2]*conf_mat[1])
    mcc /= (np.sqrt((conf_mat[0]+conf_mat[1])*(conf_mat[0]+conf_mat[2])*(conf_mat[3]+conf_mat[1])*(conf_mat[3]+conf_mat[2]))+eps)
    return [acc, sns, spc, ppv, npv, bac, f1, mcc]



def get_mse(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    loss = ((actual - predicted) ** 2).mean(axis=0)
    return loss


def get_accuracy(actual, predicted, threshold):
    correct = 0
    predicted_classes = []
    for prediction in predicted:
        if prediction >= threshold:
            predicted_classes.append(1)
        else:
            predicted_classes.append(0)
    for i in range(len(actual)):
        if actual[i] == predicted_classes[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def pred_to_classes(actual, predicted, threshold):
    predicted_classes = []
    for prediction in predicted:
        if prediction >= threshold:
            predicted_classes.append(1)
        else:
            predicted_classes.append(0)
    return predicted_classes


# precision
def get_tp(actual, predicted, threshold):
    predicted_classes = pred_to_classes(actual, predicted, threshold)
    tp = 0
    for i in range(len(predicted_classes)):
        if predicted_classes[i] == 1 and actual[i] == 1:
            tp += 1
    return tp


def get_fp(actual, predicted, threshold):
    predicted_classes = pred_to_classes(actual, predicted, threshold)
    fp = 0
    for i in range(len(predicted_classes)):
        if predicted_classes[i] == 1 and actual[i] == 0:
            fp += 1
    return fp


def get_tn(actual, predicted, threshold):
    predicted_classes = pred_to_classes(actual, predicted, threshold)
    tn = 0
    for i in range(len(predicted_classes)):
        if predicted_classes[i] == 0 and actual[i] == 0:
            tn += 1
    return tn


def get_fn(actual, predicted, threshold):
    predicted_classes = pred_to_classes(actual, predicted, threshold)
    fn = 0
    for i in range(len(predicted_classes)):
        if predicted_classes[i] == 0 and actual[i] == 1:
            fn += 1
    return fn


# precision = TP/ (TP + FP)
def precision(actual, predicted, threshold):
    prec = get_tp(actual, predicted, threshold) / (
                get_tp(actual, predicted, threshold) + get_fp(actual, predicted, threshold))
    return prec


# recall = TP / (TP + FN)
# sensitivity = recall 
def sensitivity(actual, predicted, threshold):
    sens = get_tp(actual, predicted, threshold) / (
                get_tp(actual, predicted, threshold) + get_fn(actual, predicted, threshold))
    return sens


# Specificity = TN/(TN+FP)
def specificity(actual, predicted, threshold):
    spec = get_tn(actual, predicted, threshold) / (
                get_tn(actual, predicted, threshold) + get_fp(actual, predicted, threshold))
    return spec


# f1 score  = 2 / ((1/ precision) + (1/recall))
def f_score(actual, predicted, threshold):
    f_sc = 2 / ((1 / precision(actual, predicted, threshold)) + (1 / sensitivity(actual, predicted, threshold)))
    return f_sc


# mcc = (TP * TN - FP * FN) / sqrt((TN+FN) * (FP+TP) *(TN+FP) * (FN+TP))
def mcc(act, pred, thre):
    tp = get_tp(act, pred, thre)
    tn = get_tn(act, pred, thre)
    fp = get_fp(act, pred, thre)
    fn = get_fn(act, pred, thre)
    mcc_met = (tp * tn - fp * fn) / (sqrt((tn + fn) * (fp + tp) * (tn + fp) * (fn + tp)))
    return mcc_met


def auroc(act, pred):
    return roc_auc_score(act, pred)


def auprc(act, pred):
    return average_precision_score(act, pred)
