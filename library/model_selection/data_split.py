import numpy as np
np.set_printoptions(precision=3, linewidth=200, suppress=True)
from random import seed


def train_validate_split(x_axis, features, labels, split=0.75, select_features=[],
                         verbose=False, randomize=False, random_state=7):
    """
    Split the features into train and validate set based on split ratio
    :param x_axis:
    :param features:
    :param labels:
    :param split:
    :param select_features:
    :param verbose:
    :param randomize:
    :param random_state:
    :return:
    """
    if randomize is True:
        seed(random_state)
        order = np.random.permutation(x_axis.shape[0])
        x_axis = x_axis[order]
        features = features[order]
        labels = labels[order]
    num_train_samples = int(split*labels.shape[0])
    num_validate_samples = labels.shape[0] - num_train_samples
    if verbose is True:
        print('Feature selected for training: ' + str(select_features))
        print('Split into train and validate sets: ' + str(split))
        print('Number of training samples: ' + str(num_train_samples))
        print('Number of validating samples: ' + str(num_validate_samples))
    if split != 1.0:
        train_x_axis = x_axis[:num_train_samples].reshape(num_train_samples, 1)
        train_labels = labels[:num_train_samples].reshape(num_train_samples, 1)
        validate_x_axis = x_axis[num_train_samples:].reshape(num_validate_samples, 1)
        validate_labels = labels[num_train_samples:].reshape(num_validate_samples, 1)
        if len(select_features) == 0:
            train_features = features[:num_train_samples, :]
            validate_features = features[num_train_samples:, :]
        else:
            train_features = features[:num_train_samples, select_features]
            validate_features = features[num_train_samples:, select_features]
        if verbose is True:
            print('Shape of train features: ' + str(train_features.shape))
            print('Shape of train labels: ' + str(train_labels.shape))
            print('Shape of validate features: ' + str(validate_features.shape))
            print('Shape of validate labels: ' + str(validate_labels.shape))
        return train_x_axis, train_features, train_labels, validate_x_axis, validate_features, validate_labels
    else:
        if verbose is True:
            print('Shape of features: ' + str(features.shape))
            print('Shape of labels: ' + str(labels.shape))
        return x_axis, features, labels


def cross_validate_split(x_axis, features, labels, select_features=[], num_folds=10,
                         randomize=False, random_state=7, verbose=False):
    validate_size = int(len(features) / num_folds)
    num_examples = features.shape[0]
    start_index = 0
    validate_features = list()
    train_features = list()
    validate_labels = list()
    train_labels = list()
    for i in range(num_folds):
        end_index = start_index + validate_size
        validate_features_split = list()
        train_features_split = list()
        validate_labels_split = list()
        train_labels_split = list()
        for j in range(num_examples):
            if j >= start_index and j < end_index:
                validate_features_split.append(features[j, :])
                validate_labels_split.append(labels[j])
            else:
                train_features_split.append(features[j, :])
                train_labels_split.append(labels[j])
        start_index += validate_size
        validate_features.append(validate_features_split)
        train_features.append(train_features_split)
        validate_labels.append(validate_labels_split)
        train_labels.append(train_labels_split)
    return np.array(train_features), np.array(train_labels), \
           np.array(validate_features), np.array(validate_labels)

