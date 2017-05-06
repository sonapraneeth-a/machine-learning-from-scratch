import numpy as np
import pandas as pd


def roc_auc_score(y_true, y_pred):
    return True


def f_beta_score(y_true, y_pred, beta=1, confusion=None):
    if confusion is None:
        cm = confusion_matrix(y_true, y_pred)
    else:
        cm = confusion
    recall_score = recall(y_true, y_pred, confusion=cm)
    precision_score = precision(y_true, y_pred, confusion=cm)
    f_beta_score_list = []
    for num in range(cm.shape[0]):
        rec = recall_score[num]
        pre = precision_score[num]
        score = (1+(beta*beta))*((rec*pre)/((beta*beta*pre)+rec))
        f_beta_score_list.append(score)
    return np.array(f_beta_score_list)


def precision(y_true, y_pred, confusion=None):
    if confusion is None:
        cm = confusion_matrix(y_true, y_pred)
    else:
        cm = confusion
    unique_values = list(set(y_true))
    precision_list = []
    for num in range(len(unique_values)):
        precision_list.append(cm[num,num]/np.sum(cm[:,num]))
    return np.array(precision_list)


def recall(y_true, y_pred, confusion=None):
    if confusion is None:
        cm = confusion_matrix(y_true, y_pred)
    else:
        cm = confusion
    unique_values = list(set(y_true))
    recall_list = []
    for num in range(len(unique_values)):
        recall_list.append(cm[num,num]/np.sum(cm[num,:]))
    return np.array(recall_list)


def accuracy_score(y_true, y_pred):
    combine_values = list(zip(y_true, y_pred))
    unique_values = list(set(y_true))
    accuracy = 0
    for value in unique_values:
        accuracy += (combine_values.count((value,value)))
    accuracy = accuracy / len(y_true)
    del combine_values
    del unique_values
    return accuracy


def confusion_matrix(y_true, y_pred, target_names=[]):
    combine_values = list(zip(y_true, y_pred))
    unique_values = list(set(y_true))
    num_classes = len(unique_values)
    confusion = np.zeros((num_classes,num_classes))
    row = 0
    for true_value in unique_values:
        col = 0
        for pred_value in unique_values:
            confusion[row, col] = combine_values.count((true_value,pred_value))
            col += 1
        row += 1
    return confusion


def classification_report(y_true, y_pred, target_names=[]):
    print('Classification report')
    pre = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = f_beta_score(y_true, y_pred)
    unique_values = list(set(y_true))
    support = []
    for value in unique_values:
        support.append(y_true.tolist().count(value))
    support = np.array(support)
    print('%s\t%s\t%s\t%s\t%s' %
          ('Class'.rjust(10), 'Precision'.rjust(10), 'Recall'.rjust(10), 'F1-score'.rjust(10), 'Support'.rjust(10)))
    print('-'*60)
    for cl in range(len(unique_values)):
        if len(target_names) != len(unique_values):
            print('%s %s %s %s %s' %
                  (str(unique_values[cl]).rjust(10), str(round(pre[cl], 4)).rjust(10),
                   str(round(rec[cl], 4)).rjust(12), str(round(f1[cl], 4)).rjust(11), str(support[cl]).rjust(11)))
        else:
            print('%s %s %s %s %s' %
                  (str(target_names[cl]).rjust(10), str(round(pre[cl],4)).rjust(11),
                   str(round(rec[cl],4)).rjust(11), str(round(f1[cl],4)).rjust(11), str(support[cl]).rjust(11)))
    pre_avg = np.average(pre, weights=support.tolist())
    re_avg = np.average(rec, weights=support.tolist())
    f1_avg = np.average(f1, weights=support.tolist())
    print()
    print('%s %s %s %s %s' %
          (str('avg / total').rjust(10), str(round(pre_avg,4)).rjust(10), str(round(re_avg,4)).rjust(11),
          str(round(f1_avg, 4)).rjust(11), str(np.sum(support,axis=0)).rjust(11)))
    return True
