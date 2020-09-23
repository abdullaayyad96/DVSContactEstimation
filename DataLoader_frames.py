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

    def get_train_batches_fn(self, batch_size):
        # shuffle data
        ind_list = [i for i in range(len(self.train_idx))]
        shuffle(ind_list)

        for batch_i in range(0, len(ind_list), batch_size):
            inputs = []
            outputs = []
            batch_i_size = 0
            for i in range(batch_i, np.min([batch_i+batch_size, len(ind_list)])):
                inputs.append((self.data_['frames_augmented_equalized'][ind_list[i]]-128) / 128)
                reference_idx =int(self.data_['contact_status_augmented_equalized'][ind_list[i]])
                one_hot_vector = [0] * 18
                one_hot_vector[reference_idx] = 1
                outputs.append(one_hot_vector)
                batch_i_size = batch_i_size + 1
                
            yield np.array(inputs), np.array(outputs), batch_i_size

    def get_batches_fn_timeseries(self, batch_size):
        # shuffle data
        ind_list = [i for i in range(len(self.data_['ex_output_equalized']))]
        shuffle(ind_list)

        for batch_i in range(0, len(ind_list), batch_size):
            inputs = []
            outputs = []
            batch_i_size = 0
            for i in range(batch_i, np.min([batch_i+batch_size, len(ind_list)])):
                inputs.append(self.data_['frames_augmented_equalized'][ind_list[i]])
                reference_idx =int(self.data_['contact_status_augmented_equalized'][ind_list[i]])
                one_hot_vector = [0] * 18
                one_hot_vector[reference_idx] = 1
                outputs.append(one_hot_vector)
                batch_i_size = batch_i_size + 1
                
            yield np.array(inputs), np.array(outputs), batch_i_size
            
    def get_full_data_sequence(self, start=0):
        
        for i in range(start, len(self.data_['frames'])):
            
            yield np.array([[self.data_['frames'][i]]]), np.array([self.data_['contact_status'][i]])

    def get_validation_data(self):
        # shuffle data
        inputs = []
        outputs = []
        valid_size = 0
        for i in range(0, len(self.valid_idx)):
            inputs.append(self.data_['frames_augmented_equalized'][self.valid_idx[i]])
            reference_idx = int(self.data_['contact_status_augmented_equalized'][self.valid_idx[i]])
            one_hot_vector = [0] * 18
            one_hot_vector[reference_idx] = 1
            outputs.append(one_hot_vector)
            valid_size = valid_size + 1

        return np.array(inputs), np.array(outputs), valid_size


    def get_test_data(self):
        # shuffle data
        inputs = []
        outputs = []
        test_size = 0
        for i in range(0, len(self.test_idx)):
            inputs.append(self.data_['frames_augmented_equalized'][self.test_idx[i]])
            reference_idx = np.array(self.data_['contact_status_augmented_equalized'][self.test_idx[i]])
            one_hot_vector = [0] * 18
            one_hot_vector[reference_idx] = 1
            outputs.append(one_hot_vector)
            test_size = test_size + 1
    
        return np.array(inputs), np.array(outputs), test_size

    def load_all(self, start_idx=0, end_idx=-1, step_size=1):
        if end_idx==-1:
            end_idx = len(self.data_['frames_augmented_equalized'])

        for i in range(start_idx, end_idx, step_size):
            yield self.data_['frames_augmented_equalized'][i], self.data_['contact_status_augmented_equalized']


    def divide_data(self, train_percentage=0.7, valid_percentage=0.15, test_percentage=0.15):
        ind_list = [i for i in range(len(self.data_['frames_augmented_equalized']))]
        shuffle(ind_list)
        if test_percentage < 1:
            self.train_idx, valid_test_idx = train_test_split(ind_list, test_size=(1-train_percentage))
            self.valid_idx, self.test_idx = train_test_split(valid_test_idx, test_size=(test_percentage/(test_percentage+valid_percentage)))
        else:
            self.train_idx = []
            self.valid_idx = []
            self.test_idx = [i for i in range(len(self.data_['ex_output_equalized']))]
        

