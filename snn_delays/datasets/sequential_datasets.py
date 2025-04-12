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
import time

#one_hot_encoder = OneHotEncoder(sparse=False)

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
                DATASET_PATH, 'PSMNIST', 'data_downsampled.h5')
        else:
        # this must be created beforehand (see create_dataset in exp_psmnist)
            data_path = os.path.join(
                DATASET_PATH, 'PSMNIST', 'data.h5')

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

class AddTaskDataset(Dataset):
    """
    The adding problem Dataloader class

    The input samples consist of tensors of dimension (time_window x 2)
    where the first column is formed by random values between 0 and 1,
    and the second column is a vector of zeros where only two components
    take the value 1.

    The target labels consists of the sum of the two random components
    (values of the first column) associated with the positions where the 1s
    appear in the second column. This target is a tensor with a size of
    0.2 * sequence_length, but all the elements are the same target number
    (this format is only for representation).
    """

    def __init__(self, seq_length=50, dataset_size=128, randomness=False):
        """
        Initialization of Dataset

        :param seq_length: Length of the input sequence (tensor of dimension
        seq_length x 2)
        :param dataset_size: Number of samples in the dataset (the same as
        batch size)
        :param randomness: Control the random seed. If a dataset is generated
        for test, it must be set as False to obtain always the same test
        samples. Otherwise, for training datasets, it must be set as True.
        """
        super(AddTaskDataset, self).__init__()

        self.seq_length = seq_length
        self.dataset_size = dataset_size
        self.randomness = randomness

    def __len__(self):
        """
        The number of samples in the dataset

        :return: An integer with the dataset size
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
        _x, _y = self.create_sample(self.seq_length, self.randomness, idx)
        return _x, _y

    @staticmethod
    def create_sample(seq_len, rnd, idx):
        """
        Create a new sample of the dataset

        :param seq_len: Length of the input sequence
        :param rnd: Boolean to control de random seed. Take the value True for
        training datasets and False for testing datasets
        :param idx: Index of the sample to be returned
        :return: A tuple with the input sample and the target
        """

        # Set seed for repeated batches ir rnd=False
        if not rnd:
            torch.manual_seed(idx)

        #torch.manual_seed(idx if not rnd else idx + int(time.time()))

        # Initialization and levels definition
        seq = torch.zeros([seq_len, 2], dtype=torch.float)
        #levels = 2**16
        #levels = 2**8
        levels = 2**3

        # First input channel, random values
        seq[:, 0] = torch.randint(1, levels, (1, seq_len)) / levels

        # Second input channel, 2 random locations are 1
        a, b = torch.rand(2)        # Random locations
        wn = int(seq_len * 0.8)     # Pick random up to a point of the sequence
        idx_a = int(a * wn / 2)     # First half
        idx_b = int(wn / 2 + b * wn / 2) # Second half
        seq[[idx_a, idx_b], 1] = 1       # Set 'ones' in the second channel

        # Set label
        lbl = seq[idx_a, 0] + seq[idx_b, 0]
        #label = lbl.item() * torch.ones([int(0.1 * seq_len), 1], dtype=torch.int32)
        label = lbl.item() * torch.ones([int(0.1 * seq_len), 1])

        return seq.clone().detach(), label.clone().detach()
    
    def get_train_attributes(self):
        """
        Function to get these three attributes which are necessary for a
        correct initialization of the SNNs: num_training samples, num_input,
        etc. All Dataset should have this, if possible.
        """

        train_attrs = {'num_input': 2,
                       'num_training_samples': self.dataset_size,
                       'num_output': 1}

        return train_attrs


class SyntheticDataset(Dataset):

    def __init__(self, seq_length=50, dataset_size=128, randomness=False):

        super(SyntheticDataset, self).__init__()

        self.seq_length = seq_length
        self.mem_length = int(0.1*seq_length)
        self.dataset_size = dataset_size
        self.randomness = randomness     

    def __getitem__(self, idx):
        """
        Get a sample of the dataset. If the sample index is higher than the
        number of samples in dataset, it returns an error and stop the
        execution

        :param idx: Index of the sample to be returned
        :return: A tuple with the original (sample) and the target (label)
        sequence
        """
        _x, _y = self.create_sample(
            self.seq_length, self.mem_length, idx, self.randomness)
        return _x, _y

    def __len__(self):
        """
        The number of samples in the dataset

        :return: Dataset size
        """
        return self.dataset_size


class CopyMemoryDataset(SyntheticDataset):

    ### V6- AI speed optimized
    @staticmethod
    def create_sample(seq_length, mem_length, idx, rnd):
        # Set seed if not random
        if not rnd:
            torch.manual_seed(idx)

        # Vectorized operations
        with torch.no_grad():
            # Create base sequence
            max_noise = 0.2
            seq = torch.empty((seq_length, 3), dtype=torch.float32)
            seq[:, :] = max_noise * torch.rand(1, device='cpu')  # Broadcast
            
            # First column: random numbers 0.1-0.9
            seq[:, 0] = torch.randint(1, 10, (seq_length,), device='cpu').float() / 10.0
            
            # Determine start time
            start_time = torch.randint(high=seq_length//2, size=(1,), device='cpu').item()
            
            # Create masks
            memory_mask = torch.zeros(seq_length, device='cpu')
            memory_mask[start_time:start_time + mem_length] = 1
            
            output_mask = torch.zeros(seq_length, device='cpu')
            output_mask[seq_length-mem_length:] = 1
            
            # Apply masks
            seq[:, 1] = memory_mask
            seq[:, 2] = output_mask
            
            # Create label
            label_values = seq[start_time:start_time + mem_length, 0]
            label = label_values.view(-1, 1).expand(-1, mem_length).T
            
        return seq, label



    # ### V4: COPY TASK WITH multiple output neurons
    # ### V5: all sequence is random numbers
    # @staticmethod
    # def create_sample(seq_length, mem_length, idx, rnd):

    #     # Set seed for repeated batches ir rnd=False
    #     if not rnd:
    #         torch.manual_seed(idx)

    #     max_noise = 0.2
    #     seq = max_noise*torch.rand([seq_length, 3], dtype=torch.float)
    #     #seq[:,0] = torch.randint(1, 10, (seq_length, 1)) / 10.0 # random numbers from 0.1 to 0.9

    #     seq[:,0] = torch.randint(1, 10, (seq_length,)) / 10.0 # random numbers from 0.1 to 0.9
        
    #     # the time at which the number to memorize appears
    #     start_time = torch.randint(high=seq_length//2, size=(1,)).item()
        
    #     # marker for the sequence to remember
    #     seq[start_time:start_time + mem_length, 1] = torch.ones([mem_length])

    #     label = torch.zeros(mem_length, 1)
    #     label[:,0] = seq[start_time:start_time + mem_length, 0].T.clone().detach()
        
    #     seq[seq_length-mem_length:, 2] = torch.ones([mem_length])
        
    #     label = label.expand(-1, mem_length).T

    #     return seq.clone().detach(), label.clone().detach()

    def get_train_attributes(self):
        """
        Function to get these three attributes which are necessary for a
        correct initialization of the SNNs: num_training samples, num_input,
        etc. All Dataset should have this, if possible.
        """
        train_attrs = {'num_input': 3,
                       'num_training_samples': self.dataset_size,
                       'num_output': self.mem_length}

        return train_attrs


class MultiAddtaskDataset(SyntheticDataset):

    ### TWO SETS
    @staticmethod
    def create_sample(seq_length, mem_length, idx, rnd):

        # Set seed for repeated batches ir rnd=False
        if not rnd:
            torch.manual_seed(idx)

        max_noise = 0.2
        seq = max_noise*torch.rand([seq_length, 3], dtype=torch.float)
        seq[:,0] = torch.randint(1, 10, (seq_length,)) / 10.0 # random numbers from 0.1 to 0.9

        # the time at which the number to memorize appears
        half_seq = int(0.8*seq_length/2) - mem_length
        end_seq = int(0.8*seq_length) - mem_length
        start_time_1 = torch.randint(high=half_seq, size=(1,)).item()
        start_time_2 = torch.randint(low=half_seq+mem_length, high=end_seq, size=(1,)).item()

        # marker for the two sequence to add
        seq[start_time_1:start_time_1 + mem_length, 1] = torch.ones([mem_length])
        seq[start_time_2:start_time_2 + mem_length, 1] = torch.ones([mem_length])

        # marker for the queue at the end of the task
        seq[seq_length-mem_length:, 2] = torch.ones([mem_length])
        
        # Sum all elements of 'labels' to create a new label, normalize so the max is 2 (as in add task)
        operand1 = seq[start_time_1:start_time_1 + mem_length, 0]
        operand2 = seq[start_time_2:start_time_2 + mem_length, 0]
        lbl = torch.sum(operand1 + operand2)/(0.9*mem_length)
        label = lbl.item() * torch.ones([mem_length, 1])

        return seq.clone().detach(), label.clone().detach()

    # ### TWO SETS
    # @staticmethod
    # def create_sample(seq_length, mem_length, idx, rnd):

    #     # Set seed for repeated batches ir rnd=False
    #     if not rnd:
    #         torch.manual_seed(idx)

    #     # Initialization of the input and the target (label) sequence
    #     seq = torch.zeros([seq_length, 2], dtype=torch.float)
    #     label_1 = torch.randint(1, 10, (mem_length, 1)) / 10.0 # random numbers from 0.1 to 0.9
    #     label_2 = torch.randint(1, 10, (mem_length, 1)) / 10.0 # random numbers from 0.1 to 0.9

    #     # the time at which the number to memorize appears
    #     half_seq = int(0.8*seq_length/2) - mem_length
    #     end_seq = int(0.8*seq_length) - mem_length
    #     start_time_1 = torch.randint(high=half_seq, size=(1,)).item()
    #     start_time_2 = torch.randint(low=half_seq+mem_length, high=end_seq, size=(1,)).item()

    #     seq[start_time_1:start_time_1 + mem_length, 0] = label_1.T.clone().detach()
    #     seq[start_time_2:start_time_2 + mem_length, 0] = label_2.T.clone().detach()

    #     seq[seq_length-mem_length:, 1] = torch.ones([mem_length])
        
    #     # Sum all elements of 'labels' to create a new label, normalize so the max is 2 (as in add task)
    #     lbl = torch.sum(label_1 + label_2)/(0.9*mem_length)
    #     label = lbl.item() * torch.ones([mem_length, 1])

    #     return seq.clone().detach(), label.clone().detach()

    def get_train_attributes(self):
        """
        Function to get these three attributes which are necessary for a
        correct initialization of the SNNs: num_training samples, num_input,
        etc. All Dataset should have this, if possible.
        """
        train_attrs = {'num_input': 3,
                       'num_training_samples': self.dataset_size,
                       'num_output': 1}

        return train_attrs

    ### ONE SET
    # @staticmethod
    # def create_sample(seq_length, mem_length, idx, rnd):

    #     # Set seed for repeated batches ir rnd=False
    #     if not rnd:
    #         torch.manual_seed(idx)

    #     # Initialization of the input and the target (label) sequence
    #     seq = torch.zeros([seq_length, 2], dtype=torch.float)
    #     label = torch.randint(1, 10, (mem_length, 1)) / 10.0 # random numbers from 0.1 to 0.9

    #     # the time at which the number to memorize appears
    #     start_time = torch.randint(high=seq_length//2, size=(1,)).item()

    #     seq[start_time:start_time + mem_length, 0] = label.T.clone().detach()
        
    #     seq[seq_length-mem_length:, 1] = torch.ones([mem_length])
        
    #     # Sum all elements of 'label' to create a new label, normalize so the max is 2 (as in add task)
    #     lbl = 2.0*torch.sum(label)/(0.9*mem_length)
    #     #label = lbl.item() * torch.ones([int(0.1 * seq_len), 1], dtype=torch.int32)
    #     label = lbl.item() * torch.ones([mem_length, 1])

    #     # Set label (make +1, so the network has time to get ready to
    #     # recover the first pattern)

    #     return seq.clone().detach(), label.clone().detach()


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

