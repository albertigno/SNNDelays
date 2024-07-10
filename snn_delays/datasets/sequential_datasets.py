"""
SEQUENTIAL DATASETS:
    -) AddTaskDataset
    -) MultTaskDataset
    -) CopyMemoryDataset

These datasets have been created based in the description given by Shaojie
Bai et al. (2018):
https://arxiv.org/abs/1803.01271

Created on 2018-2022:
    github: https://github.com/albertigno/HWAware_SNNs

    @author: Alberto
    @contributors: Laura
"""

import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import sys
import h5py
from snn_delays.config import DATASET_PATH

one_hot_encoder = OneHotEncoder(sparse=False)

# TODO: Cuando se usa el Dataloader con downsample = True, da error:
#  DL = DatasetLoader(dataset='psmnist', caching='disk', num_workers=0,
#                     batch_size=256, total_time=30, downsample=True)
class SequentialPMNIST(Dataset):
    """
    Sequential P-MNIST Dataset class

    The input samples consist of a one dimensional sequence obtained by
    resizing the classic MNIST dataset (the original 28x28 matrix turned on a
    784x1 array).

    The target labels consists of the label associated to each sample, the
    number that represent the input sequence.
    """    
    # TODO: ¿Split por qué puede tomar tres valores si en el loader solo se
    #  emplean dos?
    def __init__(self, split, downsample=False):
        """
        Initialization of the class

        :param split: This argument can take the values 'train', 'test'
        or 'validation'.
        :param downsample: Boolean to activate the downsample option, that
        loads a down-sampled version of the dataset where the original matrix
        has size 14x14.
        """
        super(SequentialPMNIST, self).__init__()

        # Attributes from inputs
        self.split = split

        # Define data path
        if downsample:
            data_path = os.path.join(
                DATASET_PATH, 'raw_datasets', 'PSMNIST', 'data_downsampled.h5')
        else:
        # this must be created beforehand (see create_dataset in exp_psmnist)
            data_path = os.path.join(
                DATASET_PATH, 'raw_datasets', 'PSMNIST', 'data.h5')

        hf = h5py.File(data_path, 'r')

        # Set attributes
        # self.data = torch.tensor(
        #     np.array(hf['{}_images'.format(split)])).to('cuda:0')
        # self.labels = torch.tensor(
        #     np.array(hf['{}_labels'.format(split)])).to('cuda:0')
        self.data = torch.tensor(np.array(hf['{}_images'.format(split)]))
        self.labels = torch.tensor(np.array(hf['{}_labels'.format(split)]))
        self.num_samples = len(self.labels)

    def __len__(self):
        """
        The number of samples in the dataset.

        :return: Dataset size.
        """
        return self.num_samples        

    def __getitem__(self, idx):
        """
        Get a sample of the dataset.

        :param idx: Index of the sample to be returned
        :return: A tuple with the original (sample) and the target (label)
        sequence
        """

        return self.data[idx], self.labels[idx]

    def get_train_attributes(self):
        """
        Function to get these three attributes which are necessary for a
        correct initialization of the SNNs: num_training samples, num_input...
        All Dataset should have this, if possible.
        """
        train_attrs = {'num_input': 1,
                       'num_training_samples': len(self),
                       'num_output': 10}

        return train_attrs


class DummyPoissonDataloader(Dataset):

    '''
    Dummy Poisson Dataloader, with len(rates) inputs, 1 output
    rates is avg number of spikes per second. Max rate = 1000
    '''

    def __init__(self, rates=[10], total_timesteps=500, dataset_size=128, device='cpu'):
        """
        Initialization of Dataset
        """
        super(DummyPoissonDataloader, self).__init__()

        self.rates = rates
        self.dataset_size = dataset_size
        self.total_timesteps = total_timesteps
        self.device = device

    def __len__(self):
        """
        The number of samples in the dataset

        :return: A integer with the dataset size
        """
        return self.dataset_size

    def __getitem__(self, idx):
        """
        Get a sample of the dataset. If the sample index is higher than the
        number of samples in dataset, it returns an error and stop the
        execution

        :param idx: Index of the sample to be returned
        :return: A tuple with the input sample and the target
        """

        _x, _y = self.create_sample(idx)
        return _x, _y


    def create_sample(self, idx):
        """
        Create a new sample of the dataset

        :param seq_len: Length of the input sequence
        :param rnd: Boolean to control de random seed. Take the value True for
        training datasets and False for testing datasets
        :param idx: Index of the sample to be returned
        :return: A tuple with the input sample and the target
        """

        seq = torch.zeros([self.total_timesteps, len(self.rates)], dtype=torch.float).to(self.device)

        for neuron, rate in enumerate(self.rates):
            for t in range(self.total_timesteps):
                seq[t, neuron] = 1 if np.random.rand() < rate/1000.0 else 0  

        label = torch.tensor(0.0).to(self.device)

        return seq.clone().detach(), label.clone().detach()    


    def get_train_attributes(self):
        '''
        This is to get these three atrributes which are necessary for a correct initialization
        of the SNNs: num_training samples, num_input, num_output
        All Dataset should have this, if possible
        '''
        train_attrs = {}
        train_attrs['num_input'] = len(self.rates)
        train_attrs['num_training_samples'] = len(self)
        train_attrs['num_output'] = 1

        return train_attrs

