
# coding: utf-8

# # Multiple variable linear regression

# ## Make necessary imports

# In[1]:

import numpy as np
np.set_printoptions(precision=3, linewidth=200, suppress=True)
import library.data as data
import library.linear_regression as lr
import math
from copy import deepcopy

print_output = True


# ## Make Prediction on test data

# In[2]:

def make_predictions(clf, test_id, test_features, output_file_name, verbose=False):
    print('Predict the output for test set')
    test_labels = clf.predict(test_features)
    submit_test = np.column_stack((test_id, test_labels))
    # fmt = ','.join(['%d'] + ['%3.3f'])
    if verbose is True:
        print('Write predictions to ' + output_file_name)
        print(submit_test[:5,:])
    fmt = ','.join(['%d'] + ['%3.3f'] * (submit_test.shape[1]-1))
    return submit_test, fmt


# ## Validate your classifier against validation data

# In[3]:

def validate_classifier(clf, validate_features, validate_labels, verbose=False):
    print('Predict the output for validate set')
    predicted_labels = clf.predict(validate_features)
    if verbose is True:
        print('Coefficients: ' + str(clf.theta.flatten()))
        print('Calculating score of the regressor')
    clf.error_type = 'r2'; r2_value = clf.score(validate_labels, predicted_labels)
    clf.error_type = 'exp_var'; exp_var_value = clf.score(validate_labels, predicted_labels)
    clf.error_type = 'mean_abs'; med_abs_value = clf.score(validate_labels, predicted_labels )
    clf.error_type = 'rmse'; rmse_value = clf.score(validate_labels, predicted_labels)
    clf.error_type = 'med_abs'; med_abs_value = clf.score(validate_labels, predicted_labels)
    if verbose is True:
        print('R^2 score: %.3f' % (r2_value))
        print('Explained variance score: %.3f' % (exp_var_value))
        print('Mean absolute error: %.3f' % (med_abs_value))
        print('Root mean squared error: %.3f' % (rmse_value))
        print('Median absolute error: %.3f' % (med_abs_value))
        print()
    return rmse_value


# ## Import data from csv file

# In[4]:

train_file_name = './data/train.csv'
print('Importing data from \'%s\'' %train_file_name)
print('Reading train dataset from \'%s\' ' % train_file_name )
features, labels, attribute_names = data.read_data_from_csv(csv_file=train_file_name, label_name='MEDV')
attribute_list = list(attribute_names)
ids = features[:, 0]
features = features[:, 1:]
features_list = list(range(0,13))
if print_output is True:
    print('Size of features: ' + str(features.shape))
    print('Size of labels: ' + str(labels.shape))
    print('Features')
    print(attribute_list)
    print(features[:5,:])
    print()


# ## Remove outliers

# In[5]:

index_outliers = [369, 373, 372, 413]
print('Removing the following outliers: ', end='')
print(index_outliers)
features = np.delete(features, index_outliers, axis=0)
labels = np.delete(labels, index_outliers, axis=0)
ids = np.delete(ids, index_outliers, axis=0)


# ## Read test data

# In[6]:

test_file_name = './data/test.csv'
print('Reading test dataset from \'%s\' ' % test_file_name )
test_features, test_labels, attribute_names = data.read_data_from_csv(csv_file=test_file_name, label_name='')
if print_output is True:
    print('Size of test_features: ' + str(test_features.shape))
    print('Test features')
    print(test_features[:5,:])
test_id = test_features[:,0]
test_features = test_features[:,1:]
test_features = test_features[:, features_list]
if print_output is True:
    print('Test IDs')
    print(test_id[1:5])
    print('Test features')
    print(test_features[1:5,:])
    print()


# ## Run the linear regressor with cross validation data using L<sub>2</sub> norm gradient descent

# In[8]:

print('Performing linear regression with cross validation data using L2 norm gradient descent')
num_folds = 5
tf_s, tl_s, vf_s, vl_s = data.cross_validate_split(ids, features, labels, num_folds=num_folds)
num_splits = tf_s.shape[0]
clf = lr.LinearRegression(bias=True, iterations=100000, alpha=0.9, reg_const=0.009,
                              method='batch', norm_data= False, regularize=True,
                              norm=2, verbose=False, w_init='uniform', tolerance=0.0000000001)
clf.print_parameters()
# Normalizing the data
norm_test_features = data.normalize_data(test_features)
# Squaring the features
sq_test_features = deepcopy(norm_test_features)
sq_test_features = np.power(sq_test_features, 2)
# Appending the normalized original data to the square normalized data
new_test_features = np.column_stack((norm_test_features, sq_test_features))
split = 0
min_rmse = math.inf
best_split = 0
for i in range(num_splits):
    print('Split no.: ' + str(i+1))
    print('Estimate parameters for the train data')
    tf = tf_s[i, :, :]
    tl = tl_s[i, :].reshape(tl_s[i, :].shape[0], 1)
    vf = vf_s[i, :, :]
    vl = vl_s[i, :].reshape(vl_s[i, :].shape[0], 1)
    norm_tf = data.normalize_data(tf)
    norm_vf = data.normalize_data(vf)
    sq_tf = np.power(norm_tf, 2)
    sq_vf = np.power(norm_vf, 2)
    new_tf = np.column_stack((norm_tf, sq_tf))
    new_vf = np.column_stack((norm_vf, sq_vf))
    clf.fit(new_tf, tl)
    rmse_value = validate_classifier(clf, new_vf, vl)
    if rmse_value < min_rmse:
        best_split = i
        output_file_name = './output.csv'
        submit_test, fmt = make_predictions(clf, test_id, new_test_features, output_file_name)
        data.write_data_to_csv(submit_test, fmt, output_file_name, 'ID,MEDV\n')
        min_rmse = rmse_value
    print()
    clf.theta_init = 'uniform'
print('Best split occurs at %d split, Min. RMSE: %.4f' %(best_split,min_rmse))


# ## Output for L<sub>p</sub> norm for various p

# In[9]:

print('Printing linear regression for Lp norms')
num_folds = 5
tf_s, tl_s, vf_s, vl_s = data.cross_validate_split(ids, features, labels, num_folds=num_folds)
num_splits = tf_s.shape[0]
clf = lr.LinearRegression(bias=True, iterations=52000, alpha=0.9, reg_const=0.009,
                              method='batch', norm_data= False, regularize=True,
                              norm=2, verbose=False, w_init='uniform')
# Normalizing the data
norm_test_features = data.normalize_data(test_features)
# Squaring the features
sq_test_features = np.power(norm_test_features, 2)
# Appending the normalized original data to the square normalized data
new_test_features = np.column_stack((norm_test_features, sq_test_features))
clf_norm = lr.LinearRegression(bias=True, iterations=52000, alpha=0.9, reg_const=0.009,
                              method='batch', norm_data= False, regularize=True,
                              norm=2, verbose=False, w_init='uniform')
p_values = [1.2, 1.5, 1.8]
iter_p = 1

for p in p_values:
    print('Finding best weights for L_' + str(p))
    min_rmse = math.inf
    for i in range(num_splits):
        print('Split no.: ' + str(i+1))
        print('Estimate parameters for the train data')
        tf = tf_s[i, :, :]
        tl = tl_s[i, :].reshape(tl_s[i, :].shape[0], 1)
        vf = vf_s[i, :, :]
        vl = vl_s[i, :].reshape(vl_s[i, :].shape[0], 1)
        norm_tf = data.normalize_data(tf)
        norm_vf = data.normalize_data(vf)
        sq_tf = np.power(norm_tf, 2)
        sq_vf = np.power(norm_vf, 2)
        new_tf = np.column_stack((norm_tf, sq_tf))
        new_vf = np.column_stack((norm_vf, sq_vf))
        clf.theta_init = 'uniform'
        clf_norm.norm = p
        clf_norm.print_parameters()
        print('Estimate parameters for the train data')
        clf_norm.fit(new_tf, tl)
        print('Coefficients: ' + str(clf_norm.theta.flatten()))
        rmse_value = validate_classifier(clf_norm, new_vf, vl)
        if rmse_value < min_rmse:
            output_file_name = './output_p' + str(iter_p) + '.csv'
            submit_test, fmt = make_predictions(clf_norm, test_id, new_test_features, output_file_name)
            data.write_data_to_csv(submit_test, fmt, output_file_name, 'ID,MEDV\n')
            min_rmse = rmse_value
        print()
    print('RMSE for L_%.1f is %.4f' %(p,min_rmse))
    iter_p += 1


# ## Gradient descent for L<sub>2</sub> vs matrix inversion

# In[10]:

print('Comparing between L2 linear regression and matrix inversion')
print('L2 Linear Regression')
num_folds_array = [5]
min_rmse = math.inf
for num_folds in num_folds_array:
    tf_s, tl_s, vf_s, vl_s = data.cross_validate_split(ids, features, labels, num_folds=num_folds)
    num_splits = tf_s.shape[0]
    clf = lr.LinearRegression(bias=True, iterations=52000, alpha=0.9, reg_const=0.009,
                                  method='batch', norm_data= False, regularize=True,
                                  norm=2, verbose=False, w_init='uniform')
    for i in range(num_splits):
        print('Split no.: ' + str(i+1))
        print('Estimate parameters for the train data')
        tf = tf_s[i, :, :]
        tl = tl_s[i, :].reshape(tl_s[i, :].shape[0], 1)
        vf = vf_s[i, :, :]
        vl = vl_s[i, :].reshape(vl_s[i, :].shape[0], 1)
        norm_tf = data.normalize_data(tf)
        norm_vf = data.normalize_data(vf)
        sq_tf = np.power(norm_tf, 2)
        sq_vf = np.power(norm_vf, 2)
        new_tf = np.column_stack((norm_tf, sq_tf))
        new_vf = np.column_stack((norm_vf, sq_vf))
        clf.fit(new_tf, tl)
        rmse_value = validate_classifier(clf_norm, new_vf, vl)
        if rmse_value < min_rmse:
            output_file_name = ''
            submit_test, fmt = make_predictions(clf_norm, test_id, new_test_features, output_file_name)
            min_rmse = rmse_value
    print('Min RMSE: %.4f' %min_rmse)
    print()


# In[11]:

print('Matrix inversion')
num_folds_array = [5]
min_rmse = math.inf
for num_folds in num_folds_array:
    tf_s, tl_s, vf_s, vl_s = data.cross_validate_split(ids, features, labels, num_folds=num_folds)
    num_splits = tf_s.shape[0]
    for i in range(num_splits):
        clf_norm = lr.LinearRegression(bias=True, iterations=52000, reg_const=0.009,
                                          method='matinv', norm_data= True, regularize=True, verbose=False)
        print('Estimate parameters for the train data')
        tf = tf_s[i, :, :]
        tl = tl_s[i, :].reshape(tl_s[i, :].shape[0], 1)
        vf = vf_s[i, :, :]
        vl = vl_s[i, :].reshape(vl_s[i, :].shape[0], 1)
        norm_tf = data.normalize_data(tf)
        norm_vf = data.normalize_data(vf)
        sq_tf = np.power(norm_tf, 2)
        sq_vf = np.power(norm_vf, 2)
        new_tf = np.column_stack((norm_tf, sq_tf))
        new_vf = np.column_stack((norm_vf, sq_vf))
        clf_norm.fit(new_tf, tl)
        rmse_value = validate_classifier(clf_norm, new_vf, vl)
        if rmse_value < min_rmse:
            output_file_name = ''
            submit_test, fmt = make_predictions(clf_norm, test_id, new_test_features, output_file_name)
            min_rmse = rmse_value
    print('Min RMSE: %.4f' %min_rmse)
    print()


# # References
# 
# - [Coursera - Regularized Linear Regression](https://www.coursera.org/learn/machine-learning/supplement/pKAsc/regularized-linear-regression)
# - [Coursera - Stochastic gradient descent](https://www.coursera.org/learn/machine-learning/lecture/DoRHJ/stochastic-gradient-descent)
# - [Coursera - Mini batch gradient descent](https://www.coursera.org/learn/machine-learning/lecture/9zJUs/mini-batch-gradient-descent)
# - [Regression metrics](http://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
# - [Cross Validation](https://www.coursera.org/learn/ml-regression/lecture/FJcUw/k-fold-cross-validation)
