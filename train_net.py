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
model_filename = train_dataset_dir+'weights.txt'

# Step 1.2: Initialize neural network parameters
nn_momentum = 0.0
nn_ita = 1
nn_max = 200
net_config = {
    'Layer1': {'hidden_units': 2000, 'activation': 'relu', },
    'OutputLayer': {'activation': 'sigmoid'}
}
print('[Step 1] Initialized neural network parameters')
print()

# Step 2: Load Census Income Dataset
# Loading dataset with train_test_split of 0.8
# Encoding categorical columns with one hot labels and impute missing values
start = time.time()
income = CensusIncome(one_hot_encode=True, train_validate_split=0.8, num_input=1.0,
                      impute=True, preprocess=False, add_columns=False, encode_category=True,
                      train_filename=train_dataset_filename, test_filename='',
                      encode_type='one-hot', endian='little')
income.load_data(train=True, test=False, data_directory=train_dataset_dir)
num_train_data = income.train.data.shape[0]
end = time.time()
total_time += (end-start)
print('[Step 2] Dataset loaded in %.4f seconds' % (end-start))
print()

# Step 3: Transforming raw data by subtracting mean and dividing by standard deviation (StandardScaler Transformation)
start = time.time()
print('Transforming Census Income Dataset using %s transform' % data_transform)
if data_transform == 'StandardScaler':
    ss = StandardScaler()
    train_data = ss.transform(income.train.data)
    validate_data = ss.transform(income.validate.data)
elif data_transform == 'MinMaxScaler':
    ss = MinMaxScaler()
    train_data = ss.transform(income.train.data)
    validate_data = ss.transform(income.validate.data)
else:
    train_data = income.train.data
    validate_data = income.validate.data
end = time.time()
total_time += (end-start)
print('[Step 3] Dataset transformed using %s transformation in %.4f seconds' % (data_transform, end-start))
print()

# Step 4: Initialize neural network class with a specific configuration
nn = NeuralNetwork(num_classes=income.num_classes, config=net_config, batch='',
                   learning_rate=nn_ita, max_iterations=nn_max, verbose=False, inspect=False,
                   debug=False, one_hot=True, log=True, momentum=nn_momentum)

# Step 4.1: Make the necessary network with the given parameters
start = time.time()
one_hot = True
if one_hot is True:
    nn.setup(train_data.shape[1], income.train.one_hot_labels.shape[1])
else:
    nn.setup(train_data.shape[1], income.train.class_labels.shape[1])
end = time.time()
print('[Step 4] Time taken to setup neural network is %.4f seconds'%(end-start))

# Step 4.2: Print the established network setup
nn.print_network_setup()

# Step 5: Train your neural network
start = time.time()
nn.train(train_data, income.train.one_hot_labels,
         train_class_labels=income.train.class_labels,
         validate_data=validate_data,
         validate_one_hot_labels=income.validate.one_hot_labels,
         validate_class_labels=income.validate.class_labels)
end = time.time()
total_time += (end-start)
print('[Step 5] Fit done in %.4f seconds' % (end-start))

# Step 5.1: Save the trained model to file
print(nn.save_model_to_file(filename=model_filename))

# Step 6.1: Make predictions using trained weights on train data
print('Train dataset')
train_probs, train_class_labels, train_one_hot_labels = nn.predict(train_data)
print('Accuracy score : %.4f' % accuracy_score(income.train.class_labels, train_class_labels))
print('Confusion matrix')
print(confusion_matrix(income.train.class_labels, train_class_labels))
classification_report(income.train.class_labels, train_class_labels, target_names=income.classes)
print()
# Step 6.2: Make predictions using trained weights on validation data
print('Validation dataset')
val_probs, val_class_labels, val_one_hot_labels = nn.predict(validate_data)
print('Accuracy score : %.4f' % accuracy_score(income.validate.class_labels, val_class_labels))
print('Confusion matrix')
print(confusion_matrix(income.validate.class_labels, val_class_labels))
classification_report(income.validate.class_labels, val_class_labels, target_names=income.classes)
print()

print('[End] Network training completed')
