import numpy as np
import pandas as pd


def read_data_from_csv(csv_file='./input/data.csv', label_name='', verbose=False):
    """

    :param csv_file:
    :param label_name:
    :param verbose:
    :return:
    """
    # Read data from csv
    if verbose is True:
        print('[ DEBUG] Reading data from ', csv_file)
    array = np.genfromtxt(csv_file, delimiter=',', names=True)
    column_names = array.dtype.names
    if label_name == '':
        number_of_columns = len(column_names)
        labels = []
    else:
        number_of_columns = len(column_names)-1
        labels = array[label_name]
    features = array[column_names[0]]
    for feature in range(1, number_of_columns):
        features = np.column_stack((features, array[column_names[feature]]))
    return features, labels, column_names


def write_data_to_csv_np(matrix, fmt, csv_file='./output/output.csv', heading_row='', verbose=False):
    if verbose is True:
        print('[ DEBUG] Writing data to ' + csv_file)
    with open(csv_file, 'wb') as f:
        f.write(bytes(heading_row, encoding='UTF-8'))
        np.savetxt(f, matrix, fmt=fmt, delimiter=',')


def write_data_csv_pandas(matrix, data_headers=None, output_file='./output/output.csv', verbose=False):
    if data_headers is not None:
        df = pd.DataFrame(matrix, columns=data_headers)
    else:
        df = pd.DataFrame(matrix)
    df.to_csv(output_file, index=False)
    return True
