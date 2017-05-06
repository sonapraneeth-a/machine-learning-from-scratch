import pandas as pd
import numpy as np


class EncodeCategory:

    def __init__(self, verbose=False):
        self.verbose = verbose

    def hash_transform(self, data_attribute_column, mapped_key=None):
        if mapped_key is None:
            data_keys = np.sort(data_attribute_column.unique())
            mapped_key = {}
            index = 1
            for key in data_keys:
                mapped_key[key] = index
                index += 1
        output = []
        for attribute_value in data_attribute_column:
            output.append(mapped_key[attribute_value])
        output = np.array(output)
        return output, mapped_key

    def transform(self, data_attribute, data_type, data_key=None):
        attribute_list = list(data_attribute)
        encode_transform = np.array([])
        i = 0
        if self.verbose is True:
            print('Keys in the data : ' + str(list(data_type.keys())))
        i = 0
        column_names = []
        transform_dict = {}
        for index, attribute in enumerate(attribute_list):
            num_samples = data_attribute[attribute].size
            attribute_column = data_attribute[attribute]
            if data_type[attribute] == 'category':
                attribute_column = attribute_column.str.strip()
                if data_key is None:
                    output, encode_key = self.hash_transform(data_attribute_column=attribute_column)
                    transform_dict[attribute] = encode_key
                    column_names.append(attribute)
                else:
                    if attribute not in data_key.keys() or data_key[attribute] is None:
                        output, encode_key = self.hash_transform(data_attribute_column=attribute_column)
                    else:
                        output, encode_key = self.hash_transform(data_attribute_column=attribute_column,
                                                                 mapped_key=data_key[attribute])
                    transform_dict[attribute] = encode_key
                    column_names.append(attribute)
            else:
                output = attribute_column.as_matrix()
                column_names.append(attribute)
                transform_dict[attribute] = None
            if i == 0:
                encode_transform = output.reshape((num_samples, 1))
            else:
                encode_transform = np.hstack((encode_transform, output.reshape((num_samples, 1))))
            i += 1
        encode_transform = pd.DataFrame(data=encode_transform, columns=column_names)
        return encode_transform, transform_dict


class EncodeOneHotCategory:

    def __init__(self, verbose=False):
        self.verbose = verbose

    def hash_one_code_transform(self, data_attribute_column, mapped_key=None):
        if mapped_key is None:
            data_keys = np.sort(data_attribute_column.unique())
            mapped_key = {}
            index = 1
            for key in data_keys:
                mapped_key[key] = index
                index += 1
        output = np.zeros((data_attribute_column.shape[0], len(mapped_key.keys())))
        for index, attribute_value in enumerate(data_attribute_column):
            output[index, mapped_key[attribute_value]-1] = 1
        output = np.array(output)
        return output, mapped_key

    def transform(self, data_attribute, data_type, data_key=None):
        attribute_list = list(data_attribute)
        if self.verbose is True:
            print('Keys in the data : ' + str(list(data_type.keys())))
        i = 0
        column_names = []
        transform_dict = {}
        for index, attribute in enumerate(attribute_list):
            num_samples = data_attribute[attribute].size
            attribute_column = data_attribute[attribute]
            if data_type[attribute] == 'category':
                attribute_column = attribute_column.str.strip()
                if data_key is None or data_key[attribute] is None:
                    output, encode_key = self.hash_one_code_transform(data_attribute_column=attribute_column)
                else:
                    output, encode_key = self.hash_one_code_transform(data_attribute_column=attribute_column,
                                                                      mapped_key=data_key[attribute])
                transform_dict[attribute] = encode_key
                column_names.extend(list(encode_key.keys()))
            else:
                output = attribute_column.as_matrix()
                column_names.append(attribute)
                transform_dict[attribute] = None
            if output.ndim == 1:
                second_dim = 1
            else:
                second_dim = output.shape[1]
            if i == 0:
                encode_transform = output.reshape((num_samples, second_dim))
            else:
                encode_transform = np.hstack((encode_transform, output.reshape((num_samples, second_dim))))
            i += 1
        encode_transform = pd.DataFrame(data=encode_transform, columns=column_names)
        return encode_transform, transform_dict
