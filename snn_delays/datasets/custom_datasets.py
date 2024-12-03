
from tonic.dataset import Dataset
from typing import Callable, Optional
from snn_delays.config import DATASET_PATH
import numpy as np
from tonic.prototype.datasets.stmnist import STMNIST as ST_MNIST
import os

from itertools import product
from torch.utils.data import Subset

class CustomDataset(Dataset):
    """
    Dataloader for custom numpy or pytorch dataset.
    """
    def __init__(self, data, labels):
        """
        Initialization of the class.

        :param data: Input data.
        :param labels: Labels of the input data.
        """

        assert len(data)==len(labels), \
            "[ERROR] Data length must be equal to labels length."

        # Set attributes from input
        if len(data.shape)==3:
            self.images = np.expand_dims(data, axis=2)
        else:
            self.images = data
        # shape (num_samples, num_timesteps, num_input_neurons)
        self.labels = labels
        # shape (num_samples, num_output_neurons)

    def __len__(self):
        """
        The number of samples in the dataset.

        :return: Dataset size.
        """
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        """
        Get a sample of the dataset.

        :param idx: Index of the sample to be returned.
        :return: A tuple with the original (sample) and the target (label)
        sequence.
        """

        img, target = self.images[idx], self.labels[idx]
        return img, target    
    
    def get_train_attributes(self):
        """
        Function to get these three attributes which are necessary for a
        correct initialization of the SNNs: num_training samples, num_input...
        All Dataset should have this, if possible.
        """
        train_attrs = {'num_input': self.images.shape[3],
                       'num_training_samples': len(self),
                       'num_output': self.labels.shape[1]}

        return train_attrs        


class STMNIST(Dataset):

    def __init__(self, split: str, seed: int, 
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None):

        super().__init__(
            '',
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

        self.sensor_size = (10, 10, 2)
        self.dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
        self.label_numbers = list(range(0, 10))
        self.split = split
        #root = os.path.join(r'C:\Users\Alberto\OneDrive - UNIVERSIDAD DE SEVILLA\PythonData\Datasets\raw_datasets')

        data = ST_MNIST(root=os.path.join(DATASET_PATH, 'raw_datasets'))
        self.train, self.test = data.random_split(total_length=6953, weights={"train": 0.8, "valid": 0.2}, seed=seed)

        self.train_list = list(self.train)
        self.test_list = list(self.test)


    def __getitem__(self, idx):
        '''
        as the train-test split is done with a fixed seed in tonic_datasets, no need to 
        '''
        # if self.split == 'train':
        #     event, label = next(iter(self.train))
        # elif self.split == 'test':
        #     event, label = next(iter(self.test))
        if self.split == 'train':
            event, label = self.train_list[idx]
        elif self.split == 'test':
            event, label = self.test_list[idx]

        return self.transform(event), self.target_transform(label)
    
    def __len__(self):
        return int(6953*0.8) if self.split == 'train' else int(6953*0.2)
    

# Custom dataset for digit concatenation from a filtered dataset
class ConcatenatedDataset(Dataset):
    def __init__(self, base_dataset, sequence_length, target_classes):
        self.base_dataset = base_dataset
        indices = [i for i, (img, label) in enumerate(base_dataset) if np.argmax(label) in target_classes]
        self.filtered_dataset = Subset(base_dataset, indices)
        self.indices = list(range(len(self.filtered_dataset)))  # Indices of the base dataset
        self.target_classes = target_classes
        #self.pairs = [(i, j) for i in self.indices for j in self.indices]  # All possible pairs of indices
        self.sequence_length = sequence_length
        #self.pairs = list(product(self.indices, repeat=sequence_length))
        self.num_classes = len(target_classes)
        self.total_combinations = self.num_classes ** sequence_length

    def __len__(self):
        # Number of pairs
        #return len(self.pairs)
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get the indices for the current pair
        #indices = self.pairs[idx]
        # Retrieve the images and labels from the base dataset
        images = []
        labels = []

        for i in range(self.sequence_length):
            img, label = self.filtered_dataset[np.random.choice(self.indices)]
            images.append(img)
            labels.append(self.target_classes.index(np.argmax(label)))
        # Concatenate the images along the width (you can adjust as needed)
        concatenated_img = np.concatenate(images, axis=0)
        
        # Concatenate the labels one-hot
        encoded_label = sum(l * (self.num_classes ** i) for i, l in enumerate(reversed(labels)))
        concatenated_label = np.zeros(self.total_combinations)
        concatenated_label[encoded_label] = 1.0

        return concatenated_img, concatenated_label