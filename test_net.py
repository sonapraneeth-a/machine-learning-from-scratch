# Importing libraries
import numpy as np
np.set_printoptions(precision=3, linewidth=100, formatter={'all':lambda x: '{0: <6}'.format(str(x))})
import pandas as pd
import time

from library.datasets.census_income_kaggle import CensusIncome
from library.utils.csv_helpers import write_data_csv_pandas
from library.models.classification.neuralnet.nn import NeuralNetwork
from library.metrics.classification import accuracy_score, classification_report, confusion_matrix
from library.utils.file_utils import mkdir_p
from library.preprocessing.feature_transform import StandardScaler, MinMaxScaler
from library.plot_tools import plot

# Step 1.1: Initialize input filenames
train_dataset_dir = './'
output_dir = './'
train_dataset_filename = 'train.csv'
test_dataset_filename = 'kaggle_test_data.csv'
data_transform = 'StandardScaler'
total_time = 0
output_file = output_dir+'predictions.csv'
model_filename = train_dataset_dir+'weights.txt'

# Step 2: Load Census Income Dataset
# Loading dataset with train_test_split of 0.8
# Encoding categorical columns with one hot labels and impute missing values
start = time.time()
# NOTE: Using hardcoded dictionary assuming train.csv is not allowed as input for this file
# This map has been used for encoding train.csv
encode_dict = \
    {'age': None,
     'workclass': {'Federal-gov': 1, 'Local-gov': 2, 'Never-worked': 3, 'Private': 4,
                   'Self-emp-inc': 5, 'Self-emp-not-inc': 6, 'State-gov': 7, 'Without-pay': 8},
     'fnlwgt': None,
     'education': {'10th': 1, '11th': 2, '12th': 3, '1st-4th': 4, '5th-6th': 5, '7th-8th': 6,
                   '9th': 7, 'Assoc-acdm': 8, 'Assoc-voc': 9, 'Bachelors': 10, 'Doctorate': 11,
                   'HS-grad': 12, 'Masters': 13, 'Preschool': 14, 'Prof-school': 15,
                   'Some-college': 16},
     'education-num': None,
     'marital-status': {'Divorced': 1, 'Married-AF-spouse': 2, 'Married-civ-spouse': 3,
                        'Married-spouse-absent': 4, 'Never-married': 5, 'Separated': 6, 'Widowed': 7},
     'occupation': {'Adm-clerical': 1, 'Armed-Forces': 2, 'Craft-repair': 3, 'Exec-managerial': 4,
                    'Farming-fishing': 5, 'Handlers-cleaners': 6, 'Machine-op-inspct': 7, 'Other-service': 8,
                    'Priv-house-serv': 9, 'Prof-specialty': 10, 'Protective-serv': 11, 'Sales': 12, 'Tech-support': 13,
                    'Transport-moving': 14},
     'relationship': {'Husband': 1, 'Not-in-family': 2, 'Other-relative': 3, 'Own-child': 4,
                      'Unmarried': 5, 'Wife': 6},
     'race': {'Amer-Indian-Eskimo': 1, 'Asian-Pac-Islander': 2, 'Black': 3, 'Other': 4, 'White': 5},
     'sex': {'Female': 1, 'Male': 2},
     'capital-gain': None,
     'capital-loss': None,
     'hours-per-week': None,
     'native-country': {'Cambodia': 1, 'Canada': 2, 'China': 3, 'Columbia': 4, 'Cuba': 5,
                        'Dominican-Republic': 6, 'Ecuador': 7, 'El-Salvador': 8, 'England': 9,
                        'France': 10, 'Germany': 11, 'Greece': 12, 'Guatemala': 13, 'Haiti': 14,
                        'Holand-Netherlands': 15, 'Honduras': 16, 'Hong': 17, 'Hungary': 18,
                        'India': 19, 'Iran': 20, 'Ireland': 21, 'Italy': 22, 'Jamaica': 23, 'Japan': 24,
                        'Laos': 25, 'Mexico': 26, 'Nicaragua': 27, 'Outlying-US(Guam-USVI-etc)': 28,
                        'Peru': 29, 'Philippines': 30, 'Poland': 31, 'Portugal': 32, 'Puerto-Rico': 33,
                        'Scotland': 34, 'South': 35, 'Taiwan': 36, 'Thailand': 37, 'Trinadad&Tobago': 38,
                        'United-States': 39, 'Vietnam': 40, 'Yugoslavia': 41}}
income = CensusIncome(one_hot_encode=True, train_validate_split=0.8, num_input=1.0,
                      impute=True, preprocess=False, add_columns=False, encode_category=True,
                      train_filename='', test_filename=test_dataset_filename, encode_dict=encode_dict,
                      encode_type='one-hot', endian='little')
income.load_data(train=False, test=True, data_directory=train_dataset_dir)
end = time.time()
total_time += (end-start)
print('[Step 2] Dataset loaded in %.4f seconds' % (end-start))
print()

# Step 3: Transforming raw data by subtracting mean and dividing by standard deviation (StandardScaler Transformation)
start = time.time()
print('Transforming Census Income Dataset using %s transform' % data_transform)
if data_transform == 'StandardScaler':
    ss = StandardScaler()
    test_data = ss.transform(income.test.data)
elif data_transform == 'MinMaxScaler':
    ss = MinMaxScaler()
    test_data = ss.transform(income.test.data)
else:
    test_data = income.test.data
end = time.time()
total_time += (end-start)
print('[Step 3] Dataset converted to standard form in %.4f seconds' %(end-start))
print()

# Step 3.1: Initiate Neural network class with dummy parameters
nn = NeuralNetwork(num_classes=0, config={}, batch='', verbose=False, one_hot=True, log=True)
# Step 3.2: Load trained parameters from model file
nn.load_model_from_file(filename=model_filename)
# Step 3.3: Print network setup
print(nn.print_network_setup())

# Step 4: Read test data from csv file
test_file = train_dataset_dir+test_dataset_filename
print('Reading test data from csv  : %s' % test_file)
test_df = pd.read_csv(test_file)
num_test_samples = test_df['sex'].size
test_ids = test_df['id'].as_matrix().reshape((num_test_samples,1))
print('Test dataset has %d samples' % num_test_samples)
test_probs, test_class_labels, test_one_hot_labels = nn.predict(test_data)
test_predicted_answer = test_class_labels.reshape((num_test_samples, 1))
test_predicted_answer = np.hstack((test_ids, test_predicted_answer))
print('Writing output to %s' % output_file)
write_data_csv_pandas(test_predicted_answer, output_file=output_file, data_headers=['id','salary'])
