from snn_delays.datasets.transforms_tonic import *
from snn_delays.datasets.custom_datasets import STMNIST
from snn_delays.datasets.tonic_datasets import TonicDataset

'''
have to do this because torchdata, needed to load STMNIST, requires cuda>12
'''

class STMNISTDataset(TonicDataset):

    '''
    Neuromorphic Spiking Tactile MNIST (ST-MNIST) dataset, which comprises handwritten 
    digits obtained by human participants writing on a neuromorphic tactile sensor array. 
    Download of the compressed dataset has to be done by the user by accessing 
    https://scholarbank.nus.edu.sg/bitstream/10635/168106/2/STMNIST%20dataset%20NUS%20Tee%20Research%20Group.zip
    where a form has to be completed. The uncompressed folder has to be copied to DATASET_PATH
    '''

    def __init__(self, dataset_name='stmnist', total_time=50, **kwargs):
        super().__init__(dataset_name=dataset_name,
                         total_time=total_time,
                         **kwargs)
        
        if 'seed' in kwargs.keys():
            seed = kwargs['seed']
        else:
            seed = 0

        # Train and test dataset definition
        self.train_dataset = STMNIST(
            split='train',
            seed=seed,
            transform=self.sample_transform,
            target_transform=self.label_transform)

        self.test_dataset = STMNIST(
            split='test',
            seed=seed,
            transform=self.sample_transform,
            target_transform=self.label_transform)