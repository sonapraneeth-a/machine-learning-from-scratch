import numpy as np
np.set_printoptions(precision=3, linewidth=200, suppress=True)
from random import seed
from copy import deepcopy


def feature_normalize(features, n_type='z-score'):
    """

    :param features:
    :param n_type:
    :return:
    """
    answer = np.array([])
    if n_type == 'z-score':
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        if std != 0:
            answer = (features - mean) / std
        else:
            answer = features
    elif n_type == 'min-max':
        minimum = features.min(axis=0)
        maximum = features.max(axis=0)
        if maximum != minimum:
            answer = (features - minimum)/(maximum-minimum)
        else:
            answer = features
    return answer


def normalize_data(train_features, n_type='z-score'):
    """
    Feature scaling
    :param train_features: Training features
    :param n_type: Type of normalization
    :return:
    """
    row_no, col_no = train_features.shape
    normalize_train_features = deepcopy(train_features)
    for column_no in range(col_no):
        test = train_features[:, column_no]
        normalize_train_features[:, column_no] = feature_normalize(test, n_type=n_type)
    return normalize_train_features

