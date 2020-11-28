import torch
import torch.utils.data as data
import numpy as np


class cropDataSet(data.Dataset):
    def __init__(self, data_all):
        self.data_sample = data_all[:,:,:7]
        self.data_label = data_all[:,:,7:]

    def __len__(self):
        return len(self.data_sample)

    def __getitem__(self, index):
        return self.data_sample[index],self.data_label[index]
