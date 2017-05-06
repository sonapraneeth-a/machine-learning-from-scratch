# https://www.coursera.org/learn/machine-learning/supplement/pKAsc/regularized-linear-regression
# https://www.coursera.org/learn/machine-learning/lecture/DoRHJ/stochastic-gradient-descent
# https://www.coursera.org/learn/machine-learning/lecture/9zJUs/mini-batch-gradient-descent
# http://www.ritchieng.com/multi-variable-linear-regression/

import numpy as np
np.set_printoptions(precision=3, linewidth=200, suppress=True)
import library.data as data
import library.metrics as metrics
import time
import math


class LinearRegression:

    def __init__(self, bias=True, iterations=1000, alpha=0.001, reg_const=100, method='batch',
                 w_init='random', norm_data=False, batch_size=2, error='rmse', regularize=False,
                 verbose=False, norm=2, theta_init=[], step=100, tolerance=0.000001):
        self.theta = theta_init  # Parameters of linear regression
        self.fit_bias = bias  # If bias is to be added
        self.max_iterations = iterations  # Maximum number of iterations for running gradient descent
        self.alpha = alpha  # Learning rate for gradient descent
        self.reg_const = reg_const  # Regularization constant
        self.optimizer = method  # Gradient descent method
        self.theta_init = w_init  # Initialization of parameters
        self.normalize = norm_data  # If data is to be normalized
        self.batch_size = batch_size  # Number of examples to be trained at once in stochastic gradient descent
        self.error_type = error  # Error type
        self.regularize = regularize  # If regularization is required for learning parameters
        self.verbose = verbose  # If detailed debug information is required to be print while running
        self.power = norm
        self.learning_rate = 'constant'
        self.step_size = step
        self.tolerance = tolerance

    def parameters_init(self, number_of_features):
        # Initializing parameters with equal values
        if self.theta_init == 'uniform':
            self.theta = [1 / number_of_features] * np.ones((1, number_of_features))
        # Initializing parameters with random values
        elif self.theta_init == 'random':
            self.theta = [1 / number_of_features] * np.random.randn(1, number_of_features)
        # Initializing parameters with zero values
        elif self.theta_init == 'zero':
            self.theta = [1 / number_of_features] * np.zeros((1, number_of_features))
        # Problem specific weights
        elif self.theta_init == 'given' and self.fit_bias is True:
            fixed_param = np.array([22.972, -0.897, 1.197, 0.257, 0.675, -2.108, 2.697, \
                                    0.387, -3.141, 3.001, -2.408, -2.007, 0.934, -4.102])
            self.theta = fixed_param.reshape(1, fixed_param.shape[0])
        elif self.theta_init == 'given' and self.fit_bias is False:
            fixed_param = np.array([22.972, -0.897, 1.197, 0.257, 0.675, -2.108, 2.697, \
                                    0.387, -3.141, 3.001, -2.408, -2.007, 0.934, -4.102])
            self.theta = fixed_param.reshape(1, fixed_param.shape[0])
        # Reuse of old thetas for the problem
        elif self.theta_init == 're-use':
            self.theta = self.theta
        # Initializing parameters with random values
        else:
            self.theta = [1 / number_of_features] * np.random.randn(1, number_of_features)
            # self.theta = np.array([ 22.34241855, -0.88707706, 1.19870745, -0.02677393, 1.13696405, -2.13959902,
            #                        3.26395985, -0.02539667, -3.14429807, 2.9811639, -2.46937309, -2.05929054,
            #                        0.83202195, -3.29742582])
        if self.verbose is True:
            print('[ DEBUG] There are %d parameters needed to be estimated' % self.theta.shape[1])
            print('[ DEBUG] Initial set of parameters: ', end='')
            print(self.theta.flatten())

    def new_alpha(self, number_of_iterations):
        if self.learning_rate == 'constant':
            self.alpha = self.alpha
        elif self.learning_rate == 'optimal':
            self.alpha = 1 / (self.alpha * iteration_step)
        elif self.learning_rate == 'inv-scaling':
            self.alpha = self.alpha / np.power(number_of_iterations, 2)

    def fit_batch_gradient_descent(self, train_features, train_labels):
        print('Running Batch Gradient Descent')
        number_of_features = train_features.shape[1]
        number_of_samples = train_features.shape[0]
        self.parameters_init(number_of_features)
        converged = False
        epoch = 1
        while ((not converged) and (epoch != (self.max_iterations + 1))):
            old_error = np.dot(train_features, self.theta.T) - train_labels.reshape(number_of_samples, 1)
            old_rmse = np.sqrt((1.0 / number_of_features) * np.sum(np.power(old_error, 2)))
            gradient = ((1/number_of_samples) * np.dot(train_features.reshape(number_of_samples, number_of_features).T,
                                                     old_error.reshape(number_of_samples, 1)))
            self.theta -= ((self.alpha / number_of_samples) * gradient.T)
            if self.regularize is True:
                self.theta[0, 1:] -= ((self.alpha * self.reg_const)/(number_of_samples)) * self.theta[0, 1:]
            error = np.dot(train_features, self.theta.T) - train_labels.reshape(number_of_samples, 1)
            rmse = np.sqrt((1.0 / number_of_features) * np.sum(np.power(error, 2)))
            if math.fabs(old_rmse-rmse) < self.tolerance:
                converged = True
                print('> Converged at %d iterations with RMSE %.4f' % (epoch,rmse))
            if self.verbose is True and epoch % 5000 == 0:
                print('> Epoch: %6d, RMSE: %.6f' % (epoch, rmse))
            epoch += 1

    def fit_stochastic_gradient_descent(self, train_features, train_labels):
        print('Running Stochastic Gradient Descent')
        # Get the number of features and number of examples to be used for training
        number_of_features = train_features.shape[1]
        number_of_samples = train_features.shape[0]
        # Initialize parameters for the trainin
        self.parameters_init(number_of_features)
        # Set the number of examples to be trained at one go
        b = self.batch_size
        # Step 1: Randomize the train data and the corresponding labels
        order = np.random.permutation(train_features.shape[0])
        train_features = train_features[order]
        train_labels = train_labels[order]
        converged = False
        old_rmse = 0
        # Step 2: Run Step 3 for max_iterations number of times
        while ((not converged) and (epoch != (self.max_iterations + 1))):
            rmse = 0  # Root mean square error at the end of each iteration
            # Step 3: Do normal batch gradient descent for sample of b examples at once
            for example in range(0, number_of_samples, b):
                b_index = example
                num_examples = b
                # Loop through all the features in the train_features
                # Calculate each parameter of the linear model using b examples at once
                for feature in range(number_of_features):
                    if b_index + b > number_of_samples:
                        num_examples = number_of_samples - b_index
                        error = np.dot(train_features[b_index:, :], self.theta.T) \
                                - train_labels[b_index:b_index + b, 0].reshape(num_examples, 1)
                    else:
                        error = np.dot( train_features[b_index:b_index + b, :], self.theta.T) \
                                - train_labels[b_index:b_index + b, 0].reshape(num_examples, 1)
                    rmse += np.sum(np.power(error, 2))
                    if b_index + b > number_of_samples:
                        num_examples = number_of_samples - b_index
                        gradient = ((1 / num_examples) * np.dot(train_features[b_index:, feature].reshape( num_examples, 1 ),
                                                                error.reshape(num_examples, 1).T ))
                    else:
                        gradient = ((1 / num_examples) * np.dot(error.reshape(num_examples, 1).T,
                                                     train_features[b_index:b_index + b, feature].reshape(num_examples, 1)))
                    if feature != 0 and self.regularize is True:
                        const_mult = (self.alpha * self.reg_const) / num_examples
                        const_mult = const_mult * np.power(np.abs(self.theta[0, feature]), (self.power)-1)
                        self.theta[0, feature] = self.theta[0, feature] - const_mult - self.alpha * gradient
                    else:
                        self.theta[0, feature] -= self.alpha * gradient
            rmse /= number_of_features
            rmse = np.sqrt(rmse)
            if math.fabs(old_rmse-rmse) < self.tolerance:
                converged = True
                print('> Converged at %d iterations with RMSE %.4f' % (epoch,rmse))
            old_rmse = rmse
            epoch += 1
            if self.verbose is True and epoch % 5000 == 0:
                print('> Epoch = %6d, RMSE = %.6f' % (epoch, rmse))

    def fit_matrix_inv(self, train_features, train_labels):
        print('Running Matrix inversion')
        if self.verbose is True:
            self.print_parameters()
            print('[ DEBUG] Learning for parameters using linear regression')
        big_term = np.dot(train_features.T, train_features)  # X^{T}X
        small_term = np.dot(train_features.T, train_labels)  # X^{T}Y
        if self.regularize is True:
            big_term += self.reg_const * np.identity(train_features.shape[1])  # \lambda I
        self.theta = np.dot(np.linalg.pinv(big_term), small_term)  # (X^{T}X)^{-1}X^{T}Y
        self.theta = self.theta.T

    def fit(self, train_features, train_labels):
        start_time = time.time()
        if self.verbose is True:
            self.print_parameters()
            print('[ DEBUG] Learning for parameters using linear regression')
        number_of_features = train_features.shape[1]
        number_of_samples = train_features.shape[0]
        if self.verbose is True:
            print('[ DEBUG] Number of training features: ' + str(number_of_features))
            print('[ DEBUG] Number of training examples: ' + str(number_of_samples))
        if self.normalize is True:
            if self.verbose is True:
                print('[ DEBUG] Normalizing the data')
            normalize_train_features = data.normalize_data(train_features)
        else:
            normalize_train_features = train_features
        if self.fit_bias is True:
            if self.verbose is True:
                print('[ DEBUG] Adding bias component')
            normalize_train_features = np.column_stack((np.ones((train_features.shape[0],1)),normalize_train_features))
        if self.optimizer == 'batch':
            self.fit_batch_gradient_descent(normalize_train_features, train_labels)
        elif self.optimizer == 'stochastic':
            self.fit_stochastic_gradient_descent(normalize_train_features, train_labels)
        elif self.optimizer == 'matinv':
            self.fit_matrix_inv(normalize_train_features, train_labels)
        if self.verbose is True:
            print('Estimated parameters for the regression: ', end='')
            print(self.theta.flatten())
            print()
        end_time = time.time()
        print('Model estimation completed in %.4f seconds' % (end_time - start_time))

    def score(self, actual_labels, predicted_labels):
        if self.error_type == 'rmse':
            error = metrics.rmse(actual_labels, predicted_labels)
        elif self.error_type == 'mse':
            error = metrics.mse(actual_labels, predicted_labels)
        elif self.error_type == 'r2':
            error = metrics.r2_score(actual_labels, predicted_labels)
        elif self.error_type == 'exp_var':
            error = metrics.exp_var_score(actual_labels, predicted_labels)
        elif self.error_type == 'mean_abs':
            error = metrics.mean_abs_error(actual_labels, predicted_labels)
        elif self.error_type == 'med_abs':
            error = metrics.med_abs_error(actual_labels, predicted_labels)
        else:
            error = metrics.rmse(actual_labels, predicted_labels)
        return error

    def predict(self, features):
        if self.normalize is True:
            if self.verbose is True:
                print('[ DEBUG] Normalizing given data for prediction')
            new_features = data.normalize_data(features)
        else:
            new_features = features
        if self.fit_bias is True:
            new_test_features = np.column_stack((np.ones(new_features.shape[0]), new_features))
        else:
            new_test_features = new_features
        labels = np.dot(new_test_features, self.theta.T)
        return labels

    def print_parameters(self):
        print('Parameters for the linear regression:')
        if self.optimizer == 'batch':
            print('\tFit Bias: ', str(self.fit_bias))
            print('\tMaximum number of iterations: ', str(self.max_iterations))
            print('\tLearning Rate: ', str(self.alpha))
            if self.regularize is True:
                print('\tRegularization constant: ', str(self.reg_const))
                print('\tL_p norm: ', str(self.power))
            print('\tGradient Descent Style: ', str(self.optimizer))
            print('\tInitial Parameter Initializer Style: ', str(self.theta_init))
        if self.optimizer == 'stochastic':
            print('\tFit Bias: ', str(self.fit_bias))
            print('\tMaximum number of iterations: ', str(self.max_iterations))
            print('\tLearning Rate: ', str(self.alpha))
            if self.regularize is True:
                print('\tRegularization constant: ', str(self.reg_const))
                print('\tL_p norm: ', str(self.power))
            print('\tGradient Descent Style: ', str(self.optimizer))
            print('\tInitial Parameter Initializer Style: ', str(self.theta_init))
            print('\tNumber of examples trained at once: ', str(self.batch_size))
        if self.optimizer == 'matinv':
            print('\tMethod: Matrix inversion solution')
            if self.regularize is True:
                print('\tRegularization constant: ', str(self.reg_const))
        print('\tVerbose Information: ', str(self.verbose))
        print('\tNormalize data: ', str(self.normalize))
        print('\tLearning Rate Method: ', str(self.learning_rate))
