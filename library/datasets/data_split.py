import numpy as np
from random import random, seed, randrange
from copy import deepcopy


def cross_val_split(data, labels, num_folds=10, verbose=True):
    data_split = []
    labels_split = []
    data_copy = deepcopy(data)
    data_labels_copy = deepcopy(labels)
    fold_size = int(len(data) / num_folds)
    for fold in range(num_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(data_copy))
            fold.append(data_copy.pop(index))
        data_split.append(fold)
    return data_split, labels_split