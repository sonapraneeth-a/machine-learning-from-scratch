import numpy as np
from library.preprocessing.encode_category import EncodeCategory
from library.preprocessing.impute import Impute


class Dataset:

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.data = None
        self.data_frame = None
        self.converted_frame = None
        self.encoded_frame = None
        self.one_hot_labels = None
        self.class_labels = None
        self.filenames = None
        self.info = None
        self.class_names = None
        self.attribute_names = None

    def get_next_batch(self, batch_size=100):
        return True

    def prepare_dataset(self):
        if self.data_original is None:
            raise ValueError('Original Dataset not defined')
        encoder = EncodeCategory()
        return True