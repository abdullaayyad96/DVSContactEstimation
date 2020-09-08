import numpy as np
import sys
import h5py
from random import shuffle
import tensorflow as tf


class DataLoader:
    def __init__(self, filename):
        self.data_ = h5py.File(filename)

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

    def load_all(self, start_idx=0, end_idx=-1, step_size=1):
        if end_idx==-1:
            end_idx = len(self.data_['event_images'])

        for i in range(start_idx, end_idx, step_size):
            yield self.data_['event_images'][i], self.data_['contact_status']
