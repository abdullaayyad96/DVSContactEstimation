import numpy as np
import sys
import h5py
from random import shuffle
import tensorflow as tf
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, filename):
        self.data_ = h5py.File(filename)
        self.train_idx = []
        self.valid_idx = []
        self.test_idx = []

    def get_train_batches_fn_timeseries(self, batch_size):
        # shuffle data
        ind_list = [i for i in range(len(self.train_idx))]
        shuffle(ind_list)

        for batch_i in range(0, len(ind_list), batch_size):
            inputs = []
            outputs = []
            for i in range(batch_i, np.min([batch_i+batch_size, len(ind_list)])):
                inputs.append(self.data_['event_images'][self.data_['ex_input_image_idx_equalized'][ind_list[i]]])
                outputs.append(self.data_['ex_output_equalized'][ind_list[i]])

            yield np.array(inputs), np.array(outputs)

    def get_batches_fn_timeseries(self, batch_size):
        # shuffle data
        ind_list = [i for i in range(len(self.data_['ex_output_equalized']))]
        shuffle(ind_list)

        for batch_i in range(0, len(self.data_['ex_output_equalized']), batch_size):
            inputs = []
            outputs = []
            for i in range(batch_i, np.min([batch_i+batch_size, len(self.data_['ex_output_equalized'])])):
                inputs.append(self.data_['event_images'][self.data_['ex_input_image_idx_equalized'][ind_list[i]]])
                outputs.append(self.data_['ex_output_equalized'][ind_list[i]])

            yield np.array(inputs), np.array(outputs)

    def get_validation_data(self):
        # shuffle data
        inputs = []
        outputs = []
        for i in range(0, len(self.valid_idx)):
            inputs.append(self.data_['event_images'][self.data_['ex_input_image_idx_equalized'][self.valid_idx[i]]])
            outputs.append(self.data_['ex_output_equalized'][self.valid_idx[i]])

        return np.array(inputs), np.array(outputs)

    def get_test_data(self):
        # shuffle data
        inputs = []
        outputs = []
        for i in range(0, len(self.test_idx)):
            inputs.append(self.data_['event_images'][self.data_['ex_input_image_idx_equalized'][self.test_idx[i]]])
            outputs.append(self.data_['ex_output_equalized'][self.test_idx[i]])

        return np.array(inputs), np.array(outputs)

    def load_all(self, start_idx=0, end_idx=-1, step_size=1):
        if end_idx==-1:
            end_idx = len(self.data_['event_images'])

        for i in range(start_idx, end_idx, step_size):
            yield self.data_['event_images'][i], self.data_['contact_status']


    def divide_data(self, train_percentage=0.7, valid_percentage=0.15, test_percentage=0.15):
        ind_list = [i for i in range(len(self.data_['ex_output_equalized']))]
        shuffle(ind_list)

        self.train_idx, valid_test_idx = train_test_split(ind_list, test_size=(1-train_percentage))
        self.valid_idx, self.test_idx = train_test_split(valid_test_idx, test_size=(test_percentage/(test_percentage+valid_percentage)))
        

