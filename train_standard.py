# Importing libraries
import numpy as np
np.set_printoptions(precision=3, linewidth=100, formatter={'all':lambda x: '{0: <6}'.format(str(x))})
import pandas as pd
import time

from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from library.datasets.census_income_kaggle import CensusIncome
from library.utils.csv_helpers import write_data_csv_pandas
from library.models.classification.neuralnet.nn import NeuralNetwork
from library.utils.file_utils import mkdir_p
from library.preprocessing.feature_transform import StandardScaler, MinMaxScaler
from library.plot_tools import plot

# Step 1.1: Initialize input filenames
train_dataset_dir = './'
output_dir = './'
train_dataset_filename = 'train.csv'
test_dataset_filename = 'kaggle_test_data.csv'
data_transform = 'StandardScaler'

# Step 2: Load Census Income Dataset
# Loading dataset with train_test_split of 0.8
# Encoding categorical columns with one hot labels and impute missing values
start = time.time()
income = CensusIncome(one_hot_encode=True, train_validate_split=0.8, num_input=1.0, 
                      impute=True, preprocess=False, add_columns=False, encode_category=True, 
                      train_filename=train_dataset_filename, test_filename=test_dataset_filename, 
                      encode_type='one-hot', endian='little')
income.load_data(train=True, test=True, data_directory=train_dataset_dir)
end = time.time()
print('[Step 2] Dataset loaded in %.4f seconds' % (end-start))
print()


print('Test dataset')
test_file = train_dataset_dir+test_dataset_filename
print('Reading test data from csv  : %s' % test_file)
test_df = pd.read_csv(test_file)
num_test_samples = test_df['sex'].size
test_ids = test_df['id'].as_matrix().reshape((num_test_samples,1))
print('Test dataset has %d samples' % num_test_samples)
print()

# Step 3: Transforming raw data by subtracting mean and dividing by standard deviation (StandardScaler Transformation)
print('Transforming Census Income Dataset using %s transform' % data_transform)
if data_transform == 'StandardScaler':
    ss = StandardScaler()
    train_data = ss.transform(income.train.data)
    validate_data = ss.transform(income.validate.data)
    test_data = ss.transform(income.test.data)
elif data_transform == 'MinMaxScaler':
    ss = MinMaxScaler()
    train_data = ss.transform(income.train.data)
    validate_data = ss.transform(income.validate.data)
    test_data = ss.transform(income.test.data)
else:
    train_data = income.train.data
    validate_data = income.validate.data
    test_data = income.test.data
print('[Step 3] Dataset transformed using %s transformation in %.4f seconds' % (data_transform, end-start))
print()

# Step 4.1: Adaboost Classifier
print('Adaboost Classifier')
adb = AdaBoostClassifier()
print(adb)
adb.fit(train_data, income.train.class_labels)
adb_predict = adb.predict(validate_data)
print('Accuracy score : %.4f' % adb.score(validate_data, income.validate.class_labels))
print('ROC score      : %.4f' % roc_auc_score(income.validate.class_labels, adb_predict))
print('Classification report')
print(classification_report(income.validate.class_labels, adb_predict, target_names=income.classes))
test_predicted_answer = adb.predict(test_data).reshape((num_test_samples, 1))
test_predicted_answer = np.hstack((test_ids, test_predicted_answer))
output_adb = output_dir+'predictions_1.csv'
print('Writing output of AdaBoost Classifier to %s' % output_adb)
write_data_csv_pandas(test_predicted_answer, output_file=output_adb, data_headers=['id','salary'])
print()

# Step 4.2: Decision Tree Classifier
print('Decision Tree Classifier')
dt = DecisionTreeClassifier()
print(dt)
dt.fit(train_data, income.train.class_labels)
dt_predict = dt.predict(validate_data)
print('Accuracy score : %.4f' % dt.score(validate_data, income.validate.class_labels))
print('ROC score      : %.4f' % roc_auc_score(income.validate.class_labels, dt_predict))
print('Classification report')
print(classification_report(income.validate.class_labels, dt_predict, target_names=income.classes))
test_predicted_answer = dt.predict(test_data).reshape((num_test_samples, 1))
test_predicted_answer = np.hstack((test_ids, test_predicted_answer))
output_dt = output_dir+'predictions_2.csv'
print('Writing output of Decision Tree Classifier to %s' % output_dt)
write_data_csv_pandas(test_predicted_answer, output_file=output_dt, data_headers=['id','salary'])
print()

# Step 4.3: Logistic Classifier
print('Logistic Classifier')
logr = LogisticRegression()
print(logr)
logr.fit(train_data, income.train.class_labels)
logr_predict = logr.predict(validate_data)
print('Accuracy score : %.4f' % logr.score(validate_data, income.validate.class_labels))
print('ROC score      : %.4f' % roc_auc_score(income.validate.class_labels, logr_predict))
print('Classification report')
print(classification_report(income.validate.class_labels, logr_predict, target_names=income.classes))
test_predicted_answer = logr.predict(test_data).reshape((num_test_samples, 1))
test_predicted_answer = np.hstack((test_ids, test_predicted_answer))
output_logr = output_dir+'predictions_3.csv'
print('Writing output of Logistic Classifier to %s' % output_logr)
write_data_csv_pandas(test_predicted_answer, output_file=output_logr, data_headers=['id','salary'])
print()

# print('Gaussian Naive Bayes Classifier')
# gnb = GaussianNB()
# print(gnb)
# gnb.fit(train_data, income.train.class_labels)
# gnb_predict = gnb.predict(validate_data)
# print('Accuracy score : %.4f' % gnb.score(validate_data, income.validate.class_labels))
# print('ROC score      : %.4f' % roc_auc_score(income.validate.class_labels, gnb_predict))
# print('Classification report')
# print(classification_report(income.validate.class_labels, gnb_predict, target_names=income.classes))
# test_predicted_answer = gnb.predict(test_data).reshape((num_test_samples, 1))
# test_predicted_answer = np.hstack((test_ids, test_predicted_answer))
# output_gnb = output_dir+'predictions_2.csv'
# print('Writing output to %s' % output_gnb)
# write_data_csv_pandas(test_predicted_answer, output_file=output_gnb, data_headers=['id','salary'])
# print()

