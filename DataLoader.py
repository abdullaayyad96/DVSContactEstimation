import numpy as np
import sys
import h5py
from random import shuffle


class DataLoader:
    def __init__(self, filename):
        self.data_ = h5py.File(filename)

    def get_batches_fn_timeseries(self, batch_size):
        # shuffle data
        ind_list = [i for i in range(len(self.data_['normalized_ref_class']))]
        shuffle(ind_list)

        for batch_i in range(0, len(self.data_['normalized_ref_class']), batch_size):
            inputs = []
            outputs = []
            for i in range(batch_i, np.min([batch_i+batch_size, len(self.data_['normalized_ref_class'])])):
                inputs.append(self.data_['event_images'][self.data_['normalized_event_image_sequense'][ind_list[i]]])
                outputs.append(self.data_['normalized_ref_class'][ind_list[i]])


            yield np.array(inputs), np.array(outputs)
