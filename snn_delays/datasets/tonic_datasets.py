"""
TONIC DATASETS

Classes:
    -) TonicDataset (generic class for all the tonic dataset)
    -) SHDDataset
    -) SSCDataset
    -) NMNISTDataset
    -) IbmGesturesDataset
    -) SMNISTDataset

Created on 2018-2022:
    github: https://github.com/albertigno/HWAware_SNNs

    @author: Alberto
    @contributors: Laura
"""

import os
import numpy as np
import math
import tonic.datasets as datasets
import tonic.transforms as transforms
from snn_delays.config import DATASET_PATH
from snn_delays.datasets.transforms_tonic import *
#from snn_delays.datasets.custom_datasets import LIPSFUS
np.random


class TonicDataset:
    """
    Tonic Dataset class

    Defines the common attributes and methods for all tonic datasets.
    """
    PATH = os.path.join(DATASET_PATH, 'raw_datasets')

    def __init__(self, dataset_name, total_time, sensor_size_to=None,
                 crop_to=None, one_polarity=False, merge_polarity=False, 
                 random_crop=None):

        # Initialization of attributes
        self.dataset_name = dataset_name
        self.total_time = total_time
        self.train_dataset = None
        self.test_dataset = None

        # Set parameters for transformations
        parameter_dataset= {
            'shd': {'n_classes': 20,
                    'sensor_size': datasets.SHD.sensor_size},
            'ssc': {'n_classes': 35,
                    'sensor_size': datasets.SSC.sensor_size},
            'nmnist': {'n_classes': 10,
                    'sensor_size': datasets.NMNIST.sensor_size},
            'ibm_gestures': {'n_classes': 11,
                    'sensor_size': datasets.DVSGesture.sensor_size},
            'smnist': {'n_classes': 10,
                    'sensor_size': (99, 1, 1)},
            'lipsfus': {'n_classes': 10,
                    'sensor_size': (256, 1, 1)}
        }
        self.n_classes = parameter_dataset[self.dataset_name]['n_classes']
        original_sensor_size = \
            parameter_dataset[self.dataset_name]['sensor_size']

        # Initialize transformations
        list_sample_transform = list()
        #list_label_transform = list()

        # Sensor size and down-sampling factor
        sensor_size_list = list(original_sensor_size)
        spatial_factor = None

        if sensor_size_to is not None:
            if self.dataset_name in ['nmnist', 'ibm_gestures']:
                sensor_size_list[0:-1] = \
                    [sensor_size_to] * (len(sensor_size_list) - 1)
                target_size = (sensor_size_to, sensor_size_to)
            else:
                sensor_size_list[0] = sensor_size_to
                target_size = (sensor_size_to, 1)

            if self.dataset_name != 'smnist':
                spatial_factor = \
                    np.asarray(target_size) / original_sensor_size[:-1]

        # Crop transformation
        if crop_to is not None:
            list_sample_transform.append(
                transforms.CropTime(0, crop_to))
            
        # Random crop transformation
        if random_crop is not None:
            #duration = 2e6 # shd
            duration = 500000 # shd
            list_sample_transform.append(CropTimeRandom(duration))

        # Merge polarity transformation
        if merge_polarity:
            sensor_size_list[-1] = 1
            list_sample_transform.append(transforms.MergePolarities())

        # One polarity transformation
        if one_polarity:
            sensor_size_list[-1] = 1
            list_sample_transform.append(OnePolariy())

        # Define final sensor size
        self.sensor_size = tuple(sensor_size_list)

        # Down-sampling transformation
        if spatial_factor is not None:
            list_sample_transform.append(
                transforms.Downsample(spatial_factor=spatial_factor[0]))
            
            # rrrr = tuple([self.sensor_size[0],self.sensor_size[1]])
            # list_sample_transform.append(
            #     transforms.EventDownsampling(sensor_size=original_sensor_size,
            #                                  target_size=rrrr,
            #                                  dt = 1.0,
            #                                  downsampling_method='integrator',
            #                                  noise_threshold=10, 
            #                                  differentiator_time_bins=2))
            
        # Define final transformations
        list_sample_transform.append(
            transforms.ToFrame(
                sensor_size=self.sensor_size, n_time_bins=self.total_time))
        
        print(list_sample_transform)

        self.sample_transform = transforms.Compose(list_sample_transform)
        self.label_transform = transforms.ToOneHotEncoding(n_classes=self.n_classes)

    def get_train_attributes(self):
        """
        Function to get the attributes of the train dataset.

        :return: A dictionary that contains the features of the train dataset.
        """

        # Calculate the number of inputs using the sensor size
        num_input = 1
        for x in self.sensor_size:
            num_input *= x

        # Create the dictionary
        train_attrs = {'num_input': num_input,
                       'num_training_samples': len(self.train_dataset),
                       'num_output': self.n_classes}

        return train_attrs


class SHDDataset(TonicDataset):
    """
    SHD Dataset class

    The Spiking Heidelberg Digits (SHD) dataset is an audio-based
    classification datasets for which input spikes and output labels are
    provided. The SHD datasets are provided in HDF5 format. Documentation:
    https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/

    The input samples consist of  approximately 10k high-quality recordings
    of spoken digits ranging from zero to nine in English and German language,
    converted to spikes. The target labels consists of the number name (in
    English or German) associated to the input sample. There are 20 classes.
    """

    def __init__(self, dataset_name='shd', total_time=50, **kwargs):
        super().__init__(dataset_name=dataset_name,
                         total_time=total_time,
                         **kwargs)
        
        # Train and test dataset definition
        self.train_dataset = datasets.SHD(
            save_to=self.PATH,
            train=True,
            transform=self.sample_transform,
            target_transform=self.label_transform)

        self.test_dataset = datasets.SHD(
            save_to=self.PATH,
            train=False,
            transform=self.sample_transform,
            target_transform=self.label_transform)

class SSCDataset(TonicDataset):
    """
    SSC Dataset class

    The Spiking Speech Command (SSC) dataset are both audio-based
    classification datasets for which input spikes and output labels are
    provided. The SHD datasets are provided in HDF5 format.
    Documentation:
    https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/

    The input samples consist of spoken commands ranging from zero to nine in
    English and German language containing a single English word each and
    converted to spikes. The target labels consists of the command name
    associated to the input sample. There are 35 classes.
    """

    def __init__(self, dataset_name='ssc', total_time=50, **kwargs):
        super().__init__(dataset_name=dataset_name,
                         total_time=total_time,
                         **kwargs)

        # Train and test dataset definition
        self.train_dataset = datasets.SSC(
            save_to=self.PATH,
            split='train',
            transform=self.sample_transform,
            target_transform=self.label_transform)

        self.test_dataset = datasets.SSC(
            save_to=self.PATH,
            split='test',
            transform=self.sample_transform,
            target_transform=self.label_transform)


class NMNISTDataset(TonicDataset):
    """
    NMNIST Dataset class

    The NMNIST dataset is a spiking version of the original frame-based MNIST
    dataset. It consists of the same 60 000 training and 10 000 testing samples
    as the original MNIST dataset, and is captured at the same visual scale as
    the original MNIST dataset (28x28 pixels). Documentation:
    https://www.garrickorchard.com/datasets/n-mnist

    The input samples consist of the spiking version of the numbers from 0
    to 9. The target labels consists of the number associated to each input
    sample.
    """

    def __init__(self, dataset_name='nmnist', total_time=50, **kwargs):
        super().__init__(dataset_name=dataset_name,
                         total_time=total_time,
                         **kwargs)

        # Train and test dataset definition
        self.train_dataset = datasets.NMNIST(
            save_to=self.PATH,
            train=True,
            transform=self.sample_transform,
            target_transform=self.label_transform,
            first_saccade_only=True)

        self.test_dataset = datasets.NMNIST(
            save_to=self.PATH,
            train=False,
            transform=self.sample_transform,
            target_transform=self.label_transform,
            first_saccade_only=True)


class IbmGesturesDataset(TonicDataset):
    """
    IbmGesture Dataset class

    The IbmGesture dataset is used to build a real-time, gesture recognition
    system. The data was recorded using a DVS128. Documentation :
    https://research.ibm.com/interactive/dvsgesture/

    The input samples consist of the spiking version of the recordings of 29
    subjects making 11 hand gestures under 3 illumination conditions. The
    target labels consists of the gesture name of each input sample (arm roll,
    hand clap, left hand clockwise, air drums...).
    """

    def __init__(self, dataset_name='ibm_gestures', total_time=50, **kwargs):
        super().__init__(dataset_name=dataset_name,
                         total_time=total_time,
                         **kwargs)

        # Train and test dataset definition
        self.train_dataset = datasets.DVSGesture(
            save_to=self.PATH,
            train=True,
            transform=self.sample_transform,
            target_transform=self.label_transform)

        self.test_dataset = datasets.DVSGesture(
            save_to=self.PATH,
            train=False,
            transform=self.sample_transform,
            target_transform=self.label_transform)


class SMNISTDataset(TonicDataset):
    """
    SMNIST Dataset class

    The SMNIST dataset is a standard benchmark task for time series
    classification where each input consists of sequences of 784 pixel
    values created by unrolling the MNIST digits, pixel by pixel. In this
    spiking version, each of the 99 input neurons is associated with a
    particular threshold for the grey value, and this input neuron fires
    whenever the grey value crosses its threshold in the transition from
    the previous to the current pixel. Documentation:
    https://tonic.readthedocs.io/en/latest/reference/generated/tonic.datasets.SMNIST.html#tonic.datasets.SMNIST

    The input samples consist of the spiking version of the numbers from 0
    to 9. The target labels consists of the number associated to each input
    sample.
    """

    def __init__(self, dataset_name='smnist', total_time=50, **kwargs):
        super().__init__(dataset_name=dataset_name,
                         total_time=total_time,
                         **kwargs)

        # Train and test dataset definition
        self.train_dataset = datasets.SMNIST(
            save_to=self.PATH,
            train=True,
            num_neurons=self.sensor_size[0],
            dt=1.0,
            transform=self.sample_transform,
            target_transform=self.label_transform)

        self.test_dataset = datasets.SMNIST(
            save_to=self.PATH,
            train=False,
            num_neurons=self.sensor_size[0],
            dt=1.0,
            transform=self.sample_transform,
            target_transform=self.label_transform)
