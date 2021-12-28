from os import sep
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def evaluate(model, test_loader, threshold, id): 
    loss_sum = list()
    ac_list = list()
    pre_list = list()
    ac_spe_list = list()
    pre_spe_list = list()

    criterion = nn.BCELoss()
    for data in test_loader:
        inps, tgts = data
        inps = inps.cuda()
        tgts = tgts.cuda()
        n = inps[:, 0]
        d_i = inps[:, 1]
        
        spe_index = torch.where(d_i==id)[0]
    
        prediction_i, l2_loss = model(n, d_i)
        loss = criterion(prediction_i, tgts) + l2_loss
        loss_sum.append(loss.item())

        if len(spe_index):
            ac_spe_list.extend(tgts[spe_index].tolist())
            pre_spe_list.extend(prediction_i[spe_index].tolist())

        ac_list.extend(tgts.tolist())
        pre_list.extend(prediction_i.tolist())

    test_loss=round(np.mean(loss_sum),4)

    results, fpr, tpr = calculate(ac_list, pre_list, threshold)
    spe_results, spe_fpr, spe_tpr = calculate(ac_spe_list, pre_spe_list, threshold, True)

    return results, fpr, tpr, spe_results, spe_fpr, spe_tpr, test_loss


def calculate(ac_list, pre_list, threshold, is_print=False):
    results = dict()
    ac_list = np.array(ac_list)
    pre_list = np.array(pre_list)
    fpr, tpr, trds = roc_curve(ac_list, pre_list)
    results['AUC'] = auc(fpr, tpr)
    precision, recall, trds = precision_recall_curve(ac_list, pre_list)
    results['AUPR'] = auc(recall, precision)

    pre_list = np.where(pre_list>=threshold, 1, 0)
    if is_print:
        ac_list = np.array(ac_list)
        ac_list = ac_list.astype(np.int32)
        pre_list = np.array(pre_list)
        print(len(ac_list), np.sum(ac_list), sep='\n')
    results['ACC.'] = accuracy_score(ac_list, pre_list)
    results['P.'] = precision_score(ac_list, pre_list)
    results['R.'] = recall_score(ac_list, pre_list)
    results['F1'] = f1_score(ac_list, pre_list)

    # PR-curse-recommend-thresold
    index = np.arange(0,len(precision))
    e_index = index[np.abs(precision-recall)<=1e-2]
    if len(e_index):
        e_index = e_index[0]
        e_value = precision[e_index]
        print('PR-threshold:%.4f' % trds[e_index])
    return results, fpr, tpr
