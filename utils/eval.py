import os
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

'''
from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report, export_error_rates
from pyeer.plot import plot_eer_stats
'''

'''
pad_label=0 for attack samples, 
pad_label=1 for live samples
'''

def eval_argmax(pred, target):
    # pred: N*2, choose the max score as the predicted label
    pred_label = np.argmax(pred, axis=1)
    correct = np.sum(pred_label==target)
    acc = correct / len(target)
    return acc

def eval_argmax_pai(pred, target):
    # pred: N*m where m is the # pad types
    pred_label = np.argmax(pred, axis=1)
    pred_bi = np.where(pred_label==0, 1, 0)
    correct = np.sum(pred_bi==target)
    acc = correct / len(target)
    return acc    
    

def eval_th(pred, target, th=0.5):
    pred = pred[:, -1]
    binary_pred = np.where(pred >= th, 1, 0)
    accuracy = np.mean(binary_pred == target)
    APCER = np.sum((binary_pred == 1) & (target == 0)) / np.sum(target == 0)
    BPCER = np.sum((binary_pred == 0) & (target == 1)) / np.sum(target == 1)
    
    return {'APCER':APCER, 'BPCER':BPCER, 'acc':accuracy}


def get_err_threhold_cross_db(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0

    right_index = np.nanargmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]

    return err, best_th, right_index 


def eval_roc(pred, target):
    pred = pred[:, -1]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(target, pred)
    roc_auc = auc(fpr, tpr)

    val_err, val_threshold, right_index = get_err_threhold_cross_db(fpr, tpr, thresholds)
    FRR = 1 - tpr    # FRR = 1 - TPR
    HTER = (fpr+FRR)/2.0 
    
    # calculate accuracy
    binary_pred = np.where(pred >= val_threshold, 1, 0)
    accuracy = np.mean(binary_pred == target)
    
    return {'fpr':fpr, 'tpr':tpr, 'auc':roc_auc, 'eer':val_err, 'th':val_threshold, 
            'APCER':fpr[right_index], 'BPCER':FRR[right_index], 'acc':accuracy}