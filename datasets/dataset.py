from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils import data
import pickle
import numpy as np
import os
import random
import torch

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)
        
class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=None):
        self.path_list = []
        for file in os.listdir(path):
            if file.endswith('.npy') == True:
                if file.startswith('std') == False and file.startswith('mean') == False:
                        self.path_list.append(os.path.join(path, file))
        self.resolution = resolution
        self.transform = transform
        print(len(self.path_list))

    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, index):
        img = np.load(self.path_list[index])
        img = self.transform(img)
        return img



