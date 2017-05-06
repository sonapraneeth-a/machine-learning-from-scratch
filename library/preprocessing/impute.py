import pandas as pd
import numpy as np


class Impute:

    def __init__(self, verbose=False):
        self.verbose = verbose

    def transform(self, data_attribute, data_type, missing_value='?', strategy='most_frequent'):
        attribute_list = list(data_attribute)
        imputed_transform = np.array([])
        i = 0
        if self.verbose is True:
            print('Keys in the data : ' + str(list(data_type.keys())))
        for index, attribute in enumerate(attribute_list):
            attribute_column = data_attribute[attribute]
            num_samples = data_attribute[attribute].size
            if data_type[attribute] == 'category':
                attribute_column = attribute_column.str.strip()
                output = attribute_column.as_matrix()
                if self.verbose is True:
                    print('Type                       : Categorical attribute')
                    print('Current Attribute : ' + attribute)
                    print('Index                      : ' + str(index))
                    print('Impute Strategy   : ' + strategy)
                if strategy == 'most_frequent':
                    value = attribute_column.mode()[0]
                    if self.verbose is True:
                        print('Most frequent value : ' + str(value))
                missing_indices = (attribute_column == missing_value).as_matrix().tolist()
                to_fill_indices = [index for index, x in enumerate(missing_indices) if x is True]
                if self.verbose is True:
                    print('Indices with unfilled values : ' + str(to_fill_indices))
                output[to_fill_indices] = [value]*len(to_fill_indices)
            else:
                output = attribute_column.as_matrix()
            if i == 0:
                imputed_transform = output.reshape((num_samples, 1))
            else:
                imputed_transform = np.hstack((imputed_transform, output.reshape((num_samples, 1))))
            i += 1
        imputed_transform = pd.DataFrame(data=imputed_transform,  columns=list(data_type.keys()))
        return imputed_transform
