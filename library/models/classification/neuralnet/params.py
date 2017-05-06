import numpy as np
np.set_printoptions(precision=3, linewidth=100)


def weights(shape, initializer='standard_normal', random_state=1, verbose=False):
    np.random.seed(seed=random_state)
    if initializer == 'standard_normal':
        weight = np.random.randn(shape[0], shape[1])
    elif initializer == 'zeros':
        weight = np.zeros(shape)
    elif initializer == 'ones':
        weight = np.ones(shape)
    else:
        raise ValueError('Unknown weights initializer')
    return weight


def biases(shape, initializer='standard_normal', random_state=1, verbose=False):
    np.random.seed(seed=random_state)
    if initializer == 'standard_normal':
        bias = np.random.randn(shape[0])
    elif initializer == 'zeros':
        bias = np.zeros(shape[0])
    elif initializer == 'ones':
        bias = np.ones(shape[0])
    else:
        raise ValueError('Unknown bias initializer')
    return bias
