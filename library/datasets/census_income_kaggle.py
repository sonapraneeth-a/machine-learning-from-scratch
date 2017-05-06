import time
import numpy as np
import pandas as pd
from library.datasets.dataset import Dataset
from library.preprocessing.impute import Impute
from library.preprocessing.encode_category import EncodeCategory, EncodeOneHotCategory


class CensusIncome:

    def __init__(self, num_input=1.0, one_hot_encode=False, train_validate_split=None,
                 num_test_input=1.0, endian='big', impute=True, encode_category=True,
                 train_filename='train.csv', test_filename='test.csv', encode_type='num', encode_dict={},
                 encode_keys={}, preprocess=True, add_columns=False, verbose=False):
        self.verbose = verbose
        self.preprocess = preprocess # Should the data use feature engineering?
        self.add_columns = add_columns # Do you want add more columns for feature engineering
        self.encode_type = encode_type
        self.classes = ['<=$50K', '>$50K'] # Classes for the dataset
        self.num_classes = 2 # Number of classes in the dataset
        self.one_hot_encode = one_hot_encode # Do you want one hot encoding of data class labels
        self.endian = endian # What should be the endianess of one hot labels? 1 on leftmost bit is high or low?
        self.train_validate_split = train_validate_split # Split the train data into train and validation split
        self.num_input = num_input # How many train samples to consider
        self.num_test_input = num_test_input # How many test samples to consider
        self.num_train_input = None
        self.num_validate_input = None
        self.num_features = None
        self.impute = impute
        self.encode_categorical = encode_category # Should I encode category values?
        self.train_file_name = train_filename # What is the file from which train dataset should be loaded
        self.test_file_name = test_filename # What is the file from which test dataset should be loaded
        self.attribute_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                'hours-per-week', 'native-country']
        self.continuous_attributes = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                                      'hours-per-week'] # Numeric attribute columns
        self.categorical_attributes = ['workclass', 'education', 'marital-status', 'occupation',
                                       'relationship', 'race', 'sex', 'native-country'] # Categorical attribute columns
        self.map_attr = {}
        self.encode_dict = encode_dict # Encoding dicitonary for data transformation
        self.encode_keys = encode_keys
        for attr in self.attribute_names:
            if attr in self.continuous_attributes:
                self.map_attr[attr] = 'continuous'
            elif attr in self.categorical_attributes:
                self.map_attr[attr] = 'category'
            else:
                raise ValueError('Unknown attribute: %s'%attr)
        self.converted_map_attr = self.map_attr.copy()
        self.train = Dataset() # Train dataset class
        self.validate = Dataset() # Validate dataset class
        self.test = Dataset() # Test dataset class

    def convert_one_hot_encoding(self, classes, data_type='train'):
        num_classes = np.max(classes) + 1
        if data_type == 'train':
            self.train.one_hot_labels = np.zeros((classes.shape[0], num_classes))
        if data_type == 'validate':
            self.validate.one_hot_labels = np.zeros((classes.shape[0], num_classes))
        if data_type == 'test':
            self.test.one_hot_labels = np.zeros((classes.shape[0], num_classes))
        for i in range(classes.shape[0]):
            if self.endian == 'big':
                if data_type == 'train':
                    self.train.one_hot_labels[i, num_classes - 1 - classes[i]] = 1
                if data_type == 'validate':
                    self.validate.one_hot_labels[i, num_classes - 1 - classes[i]] = 1
                if data_type == 'test':
                    self.test.one_hot_labels[i, num_classes - 1 - classes[i]] = 1
            if self.endian == 'little':
                if data_type == 'train':
                    self.train.one_hot_labels[i, classes[i]] = 1
                if data_type == 'validate':
                    self.validate.one_hot_labels[i, classes[i]] = 1
                if data_type == 'test':
                    self.test.one_hot_labels[i, classes[i]] = 1

    def process_data(self, data_frame):
        updated_data_df = data_frame
        # Country divided into two classes -> United States and Not-United-States
        # (Mapped not us countries to Not-United-States)
        replace_values = \
            updated_data_df['native-country'].loc[
                (updated_data_df['native-country'] != 'United-States')].unique().tolist()
        if self.add_columns is True:
            updated_data_df['new-native-country'] = \
                updated_data_df['native-country'].replace(replace_values,
                                                          'Not-United-States')
            self.converted_map_attr['new-native-country'] = 'category'
        else:
            updated_data_df['native-country'] = \
                updated_data_df['native-country'].replace(replace_values,
                                                          'Not-United-States')
        # Mapped captial loss and capital gain positive values to 1 and remaining values to 0
        replace_values = \
            updated_data_df['capital-loss'].loc[(updated_data_df['capital-loss'] > 0)].unique().tolist()
        if self.add_columns is True:
            updated_data_df['new-capital-loss'] = updated_data_df['capital-loss'].replace(replace_values, 1)
            self.converted_map_attr['new-capital-loss'] = 'continuous'
        else:
            updated_data_df['capital-loss'] = updated_data_df['capital-loss'].replace(replace_values, 1)
        replace_values = \
            updated_data_df['capital-gain'].loc[(updated_data_df['capital-gain'] > 0)].unique().tolist()
        if self.add_columns is True:
            updated_data_df['new-capital-gain'] = updated_data_df['capital-gain'].replace(replace_values, 1)
            self.converted_map_attr['new-capital-gain'] = 'continuous'
        else:
            updated_data_df['capital-gain'] = updated_data_df['capital-gain'].replace(replace_values, 1)
        # Race divided into two classes -> White and Not-White
        # (Mapped not white race to Not-White)
        replace_values = \
            updated_data_df['race'].loc[(updated_data_df['race'] != 'White')].unique().tolist()
        if self.add_columns is True:
            updated_data_df['new-race'] = updated_data_df['race'].replace(replace_values, 'Not-White')
            self.converted_map_attr['new-race'] = 'category'
        else:
            updated_data_df['race'] = updated_data_df['race'].replace(replace_values, 'Not-White')
        occupation_dict = {}
        for unique_feature in updated_data_df['occupation'].unique():
            occupation_dict[unique_feature] = unique_feature
        occupation_dict['Armed-Forces'] = 'Protective-serv'
        occupation_dict['Protective-serv'] = 'Protective-serv'
        updated_data_df['occupation'] = updated_data_df['occupation'].map(occupation_dict)
        married_dict = {}
        for unique_feature in updated_data_df['marital-status'].unique():
            married_dict[unique_feature] = unique_feature
        married_dict['Married-civ-spouse'] = 'Married'
        married_dict['Married-spouse-absent'] = 'Married'
        married_dict['Married-AF-spouse'] = 'Married'
        if self.add_columns is True:
            updated_data_df['new-marital-status'] = updated_data_df['marital-status'].copy()
            updated_data_df['new-marital-status'] = updated_data_df['new-marital-status'].map(married_dict)
            self.converted_map_attr['new-marital-status'] = 'category'
        else:
            updated_data_df['marital-status'] = updated_data_df['marital-status'].map(married_dict)
        education_dict = {}
        for unique_feature in updated_data_df['education'].unique():
            education_dict[unique_feature] = unique_feature
        education_dict['1st-4th'] = 'Grade-school'
        education_dict['5th-6th'] = 'Grade-school'
        education_dict['7th-8th'] = 'Junior-high'
        education_dict['9th'] = 'HS-nongrad'
        education_dict['10th'] = 'HS-nongrad'
        education_dict['11th'] = 'HS-nongrad'
        education_dict['12th'] = 'HS-nongrad'
        education_dict['Masters'] = 'Graduate'
        education_dict['Doctorate'] = 'Graduate'
        education_dict['Preschool'] = 'Grade-school'
        if self.add_columns is True:
            updated_data_df['new-education'] = updated_data_df['education'].copy()
            updated_data_df['new-education'] = updated_data_df['new-education'].map(education_dict)
            self.converted_map_attr['new-education'] = 'category'
        else:
            updated_data_df['education'] = updated_data_df['education'].map(education_dict)
        class_dict = {}
        for unique_feature in updated_data_df['workclass'].unique():
            class_dict[unique_feature] = unique_feature
        class_dict['Local-gov'] = 'Government'
        class_dict['State-gov'] = 'Government'
        class_dict['Federal-gov'] = 'Government'
        class_dict['Self-emp-not-inc'] = 'Self-employed'
        class_dict['Self-emp-inc'] = 'Self-employed'
        if self.add_columns is True:
            updated_data_df['new-workclass'] = updated_data_df['workclass'].copy()
            updated_data_df['new-workclass'] = updated_data_df['new-workclass'].map(class_dict)
            self.converted_map_attr['new-workclass'] = 'category'
        else:
            updated_data_df['workclass'] = updated_data_df['workclass'].map(class_dict)
        del class_dict, education_dict, occupation_dict, married_dict
        return updated_data_df

    def load_train_data(self, data_directory='/tmp/cifar10/'):
        print('\tLoading Census Income Train Dataset')
        train_df = pd.read_csv(data_directory+self.train_file_name)
        if self.num_input > 1.0 or self.num_input < 0.0:
            self.num_input = train_df['sex'].size
        else:
            self.num_input = int(self.num_input * train_df['sex'].size)
        if self.train_validate_split is not None:
            self.num_train_input = int(self.train_validate_split * self.num_input)
            self.num_validate_input = self.num_input - self.num_train_input
        else:
            self.num_train_input = int(self.num_input)
        data_labels = train_df['salary']
        train_df = train_df.drop('salary', axis=1)
        train_df = train_df.drop('id', axis=1)
        self.train.data_frame = train_df.head(n=self.num_train_input)
        self.validate.data_frame = train_df.drop(train_df.head(n=self.num_train_input).index)
        self.num_features = len(list(train_df))
        if self.impute is True:
            imputer = Impute()
            data_df = imputer.transform(data_attribute=train_df, data_type=self.map_attr)
        if self.preprocess is True:
            data_df = self.process_data(data_df)
        self.train.converted_frame = data_df
        if self.preprocess is True and self.add_columns is True:
            if self.encode_categorical is True and self.encode_type == 'num':
                encoder = EncodeCategory()
                data_df, transform_dict = encoder.transform(data_attribute=data_df, data_type=self.converted_map_attr)
            if self.encode_categorical is True and self.encode_type == 'one-hot':
                encoder = EncodeOneHotCategory()
                data_df, transform_dict = encoder.transform(data_attribute=data_df, data_type=self.converted_map_attr)
            self.encode_dict = transform_dict
        else:
            if self.encode_categorical is True and self.encode_type == 'num':
                encoder = EncodeCategory()
                data_df, transform_dict = encoder.transform(data_attribute=data_df, data_type=self.map_attr)
            if self.encode_categorical is True and self.encode_type == 'one-hot':
                encoder = EncodeOneHotCategory()
                data_df, transform_dict = encoder.transform(data_attribute=data_df, data_type=self.map_attr)
            self.encode_dict = transform_dict
        # print('Transform')
        # print(transform_dict)
        # print('Encode')
        # print(self.encode_dict)
        data = np.array(data_df.as_matrix())
        data_labels = np.array(data_labels)
        print('\t\tRequested to use only %d data samples' % self.num_input)
        print('\t\tLoading %d train data samples' % self.num_train_input)
        if self.train_validate_split is None:
            self.train.data = np.array(data[:self.num_input, :])
            if self.preprocess is True:
                age_index = list(self.encode_dict.keys()).index('age')
                fnlwgt_index = list(self.encode_dict.keys()).index('fnlwgt')
                fnlwgt_index = 6
                self.train.data[:, age_index] = np.log10(self.train.data[:, age_index].astype(np.float64))
                self.train.data[:, fnlwgt_index] = np.log10(self.train.data[:, fnlwgt_index].astype(np.float64))
            # self.train.converted_frame = pd.DataFrame(self.train.data, columns=self.attribute_names)
            self.train.class_labels = np.array(data_labels[:self.num_input])
            self.train.class_names = np.array(list(map(lambda x: self.classes[x], self.train.class_labels)))
            self.train.attribute_names = list(train_df)[:-1]
        else:
            self.train.data = np.array(data[:self.num_train_input, :])
            if self.preprocess is True:
                age_index = list(self.encode_dict.keys()).index('age')
                fnlwgt_index = list(self.encode_dict.keys()).index('fnlwgt')
                fnlwgt_index = 6
                self.train.data[:, age_index] = np.log10(self.train.data[:, age_index].astype(np.float64))
                self.train.data[:, fnlwgt_index] = np.log10(self.train.data[:, fnlwgt_index].astype(np.float64))
            # self.train.converted_frame = pd.DataFrame(self.train.data, columns=self.attribute_names)
            self.train.class_labels = np.array(data_labels[:self.num_train_input])
            self.train.class_names = np.array(list(map(lambda x: self.classes[x], self.train.class_labels)))
            self.train.attribute_names = list(train_df)[:-1]
            print('\t\tLoading %d validate data samples' % self.num_validate_input)
            self.validate.data = \
                np.array(data[self.num_train_input:self.num_train_input + self.num_validate_input, :])
            if self.preprocess is True:
                age_index = list(self.encode_dict.keys()).index('age')
                fnlwgt_index = list(self.encode_dict.keys()).index('fnlwgt')
                fnlwgt_index = 6
                self.validate.data[:, age_index] = np.log10(self.validate.data[:, age_index].astype(np.float64))
                self.validate.data[:, fnlwgt_index] = np.log10(self.validate.data[:, fnlwgt_index].astype(np.float64))
            # self.validate.converted_frame = pd.DataFrame(self.validate.data, columns=self.attribute_names)
            self.validate.class_labels = \
                np.array(data_labels[self.num_train_input:self.num_train_input + self.num_validate_input])
            self.validate.class_names = np.array\
                (list(map(lambda x: self.classes[x], self.validate.class_labels)))
            self.validate.attribute_names = list(train_df)[:-1]
        if self.one_hot_encode is True:
            self.convert_one_hot_encoding(self.train.class_labels, data_type='train')
            if self.train_validate_split is not None:
                self.convert_one_hot_encoding(self.validate.class_labels, data_type='validate')
        del data_labels
        del data
        del train_df
        return True

    def load_test_data(self, data_directory='/tmp/cifar10/'):
        print('\tLoading Census Income Test Dataset')
        test_df = pd.read_csv(data_directory + self.test_file_name)
        if self.num_test_input > 1.0 or self.num_test_input < 0.0:
            self.num_test_input = test_df['sex'].size
        else:
            self.num_test_input = int(self.num_test_input * test_df['sex'].size)
        print('\t\tLoading %d test data samples' % self.num_test_input)
        test_df = test_df.drop('id', axis=1)
        self.test.data_frame = test_df.head(n=self.num_test_input)
        self.test.attribute_names = list(test_df)[:-1]
        if self.impute is True:
            imputer = Impute()
            test_df = imputer.transform(data_attribute=test_df, data_type=self.map_attr)
        if self.preprocess is True:
            test_df = self.process_data(test_df)
        self.test.converted_frame = test_df
        if self.preprocess is True and self.add_columns is True:
            if self.encode_categorical is True and self.encode_type == 'num':
                encoder = EncodeCategory()
                test_df, _ = encoder.transform(data_attribute=test_df, data_type=self.converted_map_attr,
                                               data_key=self.encode_dict)
            if self.encode_categorical is True and self.encode_type == 'one-hot':
                encoder = EncodeOneHotCategory()
                test_df, _ = encoder.transform(data_attribute=test_df, data_type=self.converted_map_attr,
                                               data_key=self.encode_dict)
        else:
            if self.encode_categorical is True and self.encode_type == 'num':
                encoder = EncodeCategory()
                test_df, _ = encoder.transform(data_attribute=test_df, data_type=self.map_attr,
                                               data_key=self.encode_dict)
            if self.encode_categorical is True and self.encode_type == 'one-hot':
                encoder = EncodeOneHotCategory()
                test_df, _ = encoder.transform(data_attribute=test_df, data_type=self.map_attr,
                                               data_key=self.encode_dict)
        test_matrix = test_df.as_matrix()
        test_data = np.array(test_matrix)
        self.test.data = np.array(test_data[:self.num_test_input])
        if self.preprocess is True:
            # print(self.encode_dict.keys())
            # print(self.encode_dict.values())
            age_index = list(self.encode_dict.keys()).index('age')
            fnlwgt_index = list(self.encode_dict.keys()).index('fnlwgt')
            fnlwgt_index = 6
            self.test.data[:, age_index] = np.log10(self.test.data[:, age_index].astype(np.float64))
            self.test.data[:, fnlwgt_index] = np.log10(self.test.data[:, fnlwgt_index].astype(np.float64))
        del test_data
        del test_df
        return True

    def load_data(self, train=True, test=True, data_directory='/tmp/census_income/'):
        print('Loading Census Income Dataset')
        start = time.time()
        if train is True:
            self.load_train_data(data_directory=data_directory)
        if test is True:
            self.load_test_data(data_directory=data_directory)
        end = time.time()
        print('Loaded Census Income Dataset in %.4f seconds' % (end - start))
        return True