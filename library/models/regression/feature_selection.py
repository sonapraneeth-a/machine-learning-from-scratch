# http://machinelearningmastery.com/feature-selection-machine-learning-python/
# https://www.coursera.org/learn/machine-learning/lecture/GBFTt/principal-component-analysis-problem-formulation
# https://www.coursera.org/learn/machine-learning/lecture/ZYIPa/principal-component-analysis-algorithm

import numpy as np
import library.data as data
np.set_printoptions(precision=3, linewidth=200, suppress=True)


class PCA:

    def __init__(self, reduce_type='best_num', best_k=2, best_perc=2, verbose=False):
        self.output_dimensions = 3
        self.reduce_type = reduce_type
        self.pick_best_k = best_k
        self.pick_best_perc = best_perc
        self.verbose = verbose

    def pca(self, features):
        if self.verbose is True:
            print('Features shape: ' + str(features.shape))
        norm_features = features - np.mean(features, axis=0)
        # norm_features = data.normalize_data(features, n_type='z-score')
        if self.verbose is True:
            print('Normalized features shape: ' +str(norm_features.shape))
        # cov_matrix = (1/m) * np.dot(norm_features.T, norm_features)
        cov_matrix = np.cov(norm_features, rowvar=False)
        if self.verbose is True:
            print('Cov. Matrix shape: ' + str(cov_matrix.shape))
        u, s, v = np.linalg.svd(cov_matrix, full_matrices=True)
        if self.pick_best_k == -1:
            u_reduce = u
        else:
            u_reduce = u[:, 0:self.pick_best_k]
        z = np.dot(norm_features, u_reduce)
        if self.verbose is True:
            print('U shape: ' + str(u.shape))
            print('S shape: ' + str(s.shape))
            print('V shape: ' + str(v.shape))
            print('U_reduce shape: ' + str(u_reduce.shape))
            print('Z shape: ' + str(z.shape))
        return z, s

    def print_parameters(self):
        print('Parameters of PCA')
        print('\tNumber of Components: ' + str(self.pick_best_k))
        print('\tVerbose: ' + str(self.verbose))