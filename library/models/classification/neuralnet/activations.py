import numpy as np
import math


class Sigmoid:

    def __init__(self):
        self.verbose = True

    @staticmethod
    def value(x):
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    def derivative(x):
        return np.multiply(Sigmoid.value(x), (1.0 - Sigmoid.value(x)))


class Tanh:

    def __init__(self):
        self.verbose = True

    @staticmethod
    def value(x):
        return np.tan(x)

    @staticmethod
    def derivative(x):
        return 1.0 - (np.multiply(Tanh.value(x), Tanh.value(x)))


class ReLU:

    def __init__(self):
        self.verbose = True

    @staticmethod
    def value(x):
        return np.maximum(0.0, x)

    @staticmethod
    def derivative(x):
        answer = x
        answer[answer > 0.0] = 1.0
        answer[answer <= 0.0] = 0.0
        return answer

