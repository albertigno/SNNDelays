from torch.utils.data import DataLoader
from tonic import DiskCachedDataset, MemoryCachedDataset
import os
from snn_delays.config import DATASET_PATH


class DatasetLoader:
    """
    Dataset Loader class

    Generate train and test data loaders for a dataset using DataLoader class
    from torch libraries. The caching method can be specified.
    """

    def __init__(self, dataset='shd', caching='disk', num_workers=0,
                 batch_size=256, total_time=50, **kwargs):
        """
        Initialization of DatasetLoader

        :param dataset: Specifies the dataset. It can take the values 'shd',
        'shd_crop', 'ssc', 'nmnist', 'nmnist784', 'ibm_gestures',
        'ibm_gestures_32', 'smnist', 'psmnist', 'addtask_episodic', 'addtask',
        'multask_episodic', 'multask', 'copymemtask_episodic', 'copymemtask'
        or 'custom'. Default = 'shd'.

        :param caching: Specifies the caching method. It can take the values
        'disk' or 'memory'. Default = 'disk'.

        :param num_workers: How many subprocesses are used for data loading.
        The value 0 means that the data will be loaded in the main process.
        Default = 0.

        :param batch_size: Number of samples to load per batch. Default = 256.

        :param total_time: Number of frames to fix when the dataset is
        transformed to frame. Default = 50.

        :param num_neurons: Number of neurons for the 'smnist' dataset.
        Default = 99.
        """
        super(DatasetLoader, self).__init__()

        # Set attributes for inputs
        self.dataset = dataset
        self.batch_size = batch_size
        self.total_time = total_time

        # Define available tonic datasets
        tonic_datasets = ['shd', 'shd_multicrop', 'ssc', 'nmnist','stmnist',
                          'nmnist784', 'ibm_gestures', 'smnist', 'lipsfus']

        # Generate train and test datasets
        if dataset in tonic_datasets:

            if dataset == 'shd':
                from snn_delays.datasets.tonic_datasets import SHDDataset
                _dataset = SHDDataset(
                    dataset_name=dataset, total_time=total_time, **kwargs)

            elif dataset == 'ssc':
                from snn_delays.datasets.tonic_datasets import SSCDataset
                _dataset = SSCDataset(
                    dataset_name=dataset, total_time=total_time, **kwargs)

            elif dataset == 'nmnist':
                from snn_delays.datasets.tonic_datasets import NMNISTDataset
                _dataset = NMNISTDataset(
                    dataset_name=dataset, total_time=total_time, **kwargs)

            elif dataset == 'ibm_gestures':
                from snn_delays.datasets.tonic_datasets import IbmGesturesDataset
                _dataset = IbmGesturesDataset(
                    dataset_name=dataset, total_time=total_time, **kwargs)

            elif dataset == 'smnist':
                from snn_delays.datasets.tonic_datasets import SMNISTDataset
                _dataset = SMNISTDataset(
                    dataset_name=dataset, total_time=total_time, **kwargs)

            elif dataset == 'stmnist':
                from snn_delays.datasets.tonic_prototype_datasets import STMNISTDataset
                _dataset = STMNISTDataset(
                    dataset_name=dataset, total_time=total_time, **kwargs)

            # Get dataset dictionary
            self.dataset_dict = _dataset.get_train_attributes()
            train_dataset = _dataset.train_dataset
            test_dataset = _dataset.test_dataset

            #expose tonic dataset
            self._dataset = _dataset
            
        else:
            if dataset == 'psmnist':
                from snn_delays.datasets.sequential_datasets import SequentialPMNIST
                downsample = kwargs['downsample']
                train_dataset = SequentialPMNIST('train', downsample)
                test_dataset = SequentialPMNIST('validation', downsample)
                if downsample:
                    self.change_total_time(196)
                else:
                    self.change_total_time(784)
            elif dataset == 'custom':
                from snn_delays.datasets.custom_datasets import CustomDataset
                data_train = kwargs['data_train']
                labels_train = kwargs['labels_train']
                data_test = kwargs['data_test']
                labels_test = kwargs['data_test']

                assert data_train.shape[1] == data_test.shape[1], \
                    "[ERROR] Check dimensions!"
                assert data_train.shape[2] == data_test.shape[2], \
                    "[ERROR] Check dimensions!"

                # Test and train are the same for this dataset
                train_dataset = CustomDataset(data=data_train,
                                              labels=labels_train)
                test_dataset = CustomDataset(data=data_test,
                                             labels=labels_test)
                self.change_total_time(data_train.shape[1])
            else:
                raise NotImplementedError

            self.dataset_dict = train_dataset.get_train_attributes()
            
        # Add the dataset and classes (optionally) names to the dictionary
        self.dataset_dict['dataset_name'] = dataset

        if 'crop_to' in kwargs:
            self.dataset_dict['time_ms'] = kwargs['crop_to']/1000
        elif 'random_crop_to' in kwargs:
            self.dataset_dict['time_ms'] = kwargs['random_crop_to'][-1]/1000
        else:
            self.dataset_dict['time_ms'] = False

        if 'mnist' in dataset or 'shd' in dataset:
            self.dataset_dict['class_names'] = [
                'zero, one, two, three, four, five, six, seven, eight, nine']

        # Check that the dataset is correctly created
        assert train_dataset is not None, \
            "[ERROR]: Dataset not found, check available options at " \
            "utils/loaders.py or examples/03_Load_a_dataset.ipynb."

        # Set dataset attributes
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # Select caching option
        if caching == 'disk':
            train_cache_path = os.path.join(
                DATASET_PATH, 'tonic_cache',
                'fast_data_loading_{}_train{}'.format(dataset, total_time))
            test_cache_path = os.path.join(
                DATASET_PATH,
                'tonic_cache',
                'fast_data_loading_{}_test{}'.format(dataset, total_time))

            train_dataset = DiskCachedDataset(train_dataset,
                                              cache_path=train_cache_path)
            test_dataset = DiskCachedDataset(test_dataset,
                                             cache_path=test_cache_path)

        elif caching == 'memory':
            train_dataset = MemoryCachedDataset(train_dataset)
            test_dataset = MemoryCachedDataset(test_dataset)

        # Define train and test loader using DataLoader from torch
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=False,
                                       pin_memory=True,
                                       num_workers=num_workers)

        self.test_loader = DataLoader(test_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      pin_memory=True,
                                      num_workers=num_workers)

    def change_total_time(self, time):
        """
        Function to change the total_time attribute.
        """

        self.total_time = time
        print(f'[INFO] Total_time changed to {time}')

    def get_dataloaders(self):
        """
        Get a tuple with the train and test data loader, and a dictionary
        which contains the information about the dataset (number of inputs,
        number of training samples..)
        """

        return self.train_loader, self.test_loader, self.dataset_dict
