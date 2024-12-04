import torch
import time
from snn_delays.snn import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils import get_device, print_spike_info, propagate_batch, set_seed
from snn_delays.utils.visualization_utils import plot_raster
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset, DataLoader
from collections import OrderedDict
device = get_device()
from torch.utils.data import Dataset, DataLoader
from tonic import MemoryCachedDataset
from snn_delays.utils.train_utils import train
from snn_delays.utils.test_behavior import tb_save_max_last_acc

# Custom dataset for digit concatenation from a filtered dataset
class SequentialMemoryRetrievalDataset(Dataset):
    def __init__(self, base_dataset, sequence_length, target_classes):
        self.base_dataset = base_dataset
        indices = [i for i, (img, label) in enumerate(base_dataset) if np.argmax(label) in target_classes]
        zero_indices = [i for i, (img, label) in enumerate(base_dataset) if np.argmax(label) == 0]
        
        self.filtered_dataset = Subset(base_dataset, indices)
        self.zeros_dataset = Subset(base_dataset, zero_indices)

        self.indices = list(range(len(self.filtered_dataset)))  # Indices of the base dataset
        self.zeros_indices = list(range(len(self.zeros_dataset)))

        self.target_classes = target_classes
        #self.pairs = [(i, j) for i in self.indices for j in self.indices]  # All possible pairs of indices
        self.sequence_length = sequence_length
        #self.pairs = list(product(self.indices, repeat=sequence_length))
        self.num_classes = len(target_classes) # 0 is not a class 
        self.total_combinations = self.num_classes ** 2

    def __len__(self):
        # Number of pairs
        #return len(self.pairs)
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get the indices for the current pair
        #indices = self.pairs[idx]
        # Retrieve the images and labels from the base dataset

        images = []
        all_labels = []
        retrieval_labels = []

        for i in range(self.sequence_length-2):
            img, label = self.filtered_dataset[np.random.choice(self.indices)]
            images.append(img)
            all_labels.append(self.target_classes.index(np.argmax(label)))

        # inserts zeroes randomly ()
        idx_frst_zero = np.random.randint(self.sequence_length//2)
        img, _ = self.zeros_dataset[np.random.choice(self.zeros_indices)]
        images.insert(idx_frst_zero, img)
        retrieval_labels.append(all_labels[idx_frst_zero])

        idx_scnd_zero = np.random.randint(idx_frst_zero+2, self.sequence_length-1)
        img, _ = self.zeros_dataset[np.random.choice(self.zeros_indices)]
        images.insert(idx_scnd_zero, img)
        retrieval_labels.append(all_labels[idx_scnd_zero-1])

        # Concatenate the images along the width (you can adjust as needed)
        concatenated_img = np.concatenate(images, axis=0)
        
        # Concatenate the labels one-hot
        encoded_label = sum(l * (self.num_classes ** i) for i, l in enumerate(reversed(retrieval_labels)))
        concatenated_label = np.zeros(self.total_combinations)
        concatenated_label[encoded_label] = 1.0

        return concatenated_img, concatenated_label

device = get_device()

# for reproducibility
torch.manual_seed(10)

dataset = 'stmnist'
total_time = 5
batch_size = 128

# DATASET
DL = DatasetLoader(dataset=dataset,
                  caching='memory',
                  num_workers=0,
                  batch_size=batch_size,
                  total_time=total_time)

train_loader, test_loader, dataset_dict = DL.get_dataloaders()

target_classes = [1, 3, 8]
test_dataset = DL._dataset.test_dataset
train_dataset = DL._dataset.train_dataset

for sequence_length in [5, 10, 20]:

    conc_test_dataset = SequentialMemoryRetrievalDataset(test_dataset, sequence_length, target_classes)
    conc_train_dataset = SequentialMemoryRetrievalDataset(train_dataset, sequence_length, target_classes)

    train_dataset = MemoryCachedDataset(conc_train_dataset)
    test_dataset = MemoryCachedDataset(conc_test_dataset)

    train_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=0)

    test_loader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=0)

    dataset_dict["num_output"] = conc_train_dataset.total_combinations
    dataset_dict["num_training_samples"] = len(conc_train_dataset)
    dataset_dict["time_ms"] = sequence_length*2e3 

    print(dataset_dict)

    ### fixed params

    model_params = {'dataset_dict': dataset_dict, 'delay_type':'h',
                    'reset_to_zero':True, 'win':total_time*sequence_length,
                    'loss_fn':'mem_sum', 'batch_size':batch_size, 'device':device,
                    'debug':False}

    ckpt_dir = f'abl6_stmnist_memory_sl{sequence_length}'

    train_params = {'learning_rate':1e-3, 'num_epochs':100, 'spk_reg':0.0, 'l1_reg':0.0,
            'dropout':0.0, 'lr_tau': 0.1, 'scheduler':(10, 0.95), 'ckpt_dir':ckpt_dir,
            'test_behavior':tb_save_max_last_acc, 'test_every':1, 'delay_pruning':None, 'weight_pruning':None,
            'lsm':False, 'random_delay_pruning' : None, 'weight_quantization': None, 'k':None, 'depth': None, 'verbose':True}

    #### first run (f+d)
    sweep_params = {
        'connection_type': ['f'],
        'delay': [(total_time, total_time//3 + 1)],
        'structure':[(64,2)],
        'tau_m':[20.0, 'normal'],
        'T_freeze_taus':[True, None]
        }

    sweep_params_names = {
        'connection_type': ['f'],
        'delay': ['dc'],
        'structure':['2l'],
        'tau_m':['hm', 'ht'],
        'T_freeze_taus':['ft', 'tt']
        }

    import itertools
    def get_configs(sweep_params, sweep_params_names):
        configurations = list(itertools.product(*sweep_params.values()))
        configurations_names = list(itertools.product(*sweep_params_names.values()))
        all_configs = []
        for config, config_names in zip(configurations, configurations_names):
            config_dict = dict(zip(sweep_params.keys(), config))
            config_dict['name']='_'.join(list(config_names))
            all_configs.append(config_dict)
        return all_configs

    cfgs = get_configs(sweep_params, sweep_params_names)

    num_repetitions = 1
    for cfg in cfgs:
        for repetition in range(0, num_repetitions):
            for key, value in zip(cfg.keys(), cfg.values()):
                if key != 'name':
                    if key.split('_')[0]=='T':
                        train_params[key[2:]] = value
                    else:
                        model_params[key] = value

            print('-----NEW TRAINING-------')
            print(model_params)
            print(train_params)

            snn = SNN(**model_params)
            snn.set_network()

            snn.model_name = cfg['name'] + '_rpt' + str(repetition)
            snn.save_model(snn.model_name + "_initial", ckpt_dir)

            train(snn, train_loader, test_loader, **train_params)


    #### second run (r)
    sweep_params = {
        'connection_type': ['r'],
        'delay': [None],
        'structure':[(64,2)],
        'tau_m':[20.0, 'normal'],
        'T_freeze_taus':[True, None]
        }

    sweep_params_names = {
        'connection_type': ['r'],
        'delay': ['nd'],
        'structure':['2l'],
        'tau_m':['hm', 'ht'],
        'T_freeze_taus':['ft', 'tt']
        }

    import itertools
    def get_configs(sweep_params, sweep_params_names):
        configurations = list(itertools.product(*sweep_params.values()))
        configurations_names = list(itertools.product(*sweep_params_names.values()))
        all_configs = []
        for config, config_names in zip(configurations, configurations_names):
            config_dict = dict(zip(sweep_params.keys(), config))
            config_dict['name']='_'.join(list(config_names))
            all_configs.append(config_dict)
        return all_configs

    cfgs = get_configs(sweep_params, sweep_params_names)

    num_repetitions = 1
    for cfg in cfgs:
        for repetition in range(0, num_repetitions):
            for key, value in zip(cfg.keys(), cfg.values()):
                if key != 'name':
                    if key.split('_')[0]=='T':
                        train_params[key[2:]] = value
                    else:
                        model_params[key] = value

            print('-----NEW TRAINING-------')
            print(model_params)
            print(train_params)

            snn = SNN(**model_params)
            snn.set_network()
            snn.model_name = cfg['name'] + '_rpt' + str(repetition)
            snn.save_model(snn.model_name + "_initial", ckpt_dir)

            train(snn, train_loader, test_loader, **train_params)


    #### third run (f)
    sweep_params = {
        'connection_type': ['f'],
        'delay': [None],
        'structure':[(64,4)],
        'tau_m':[20.0, 'normal'],
        'T_freeze_taus':[True, None]
        }

    sweep_params_names = {
        'connection_type': ['f'],
        'delay': ['nd'],
        'structure':['4l'],
        'tau_m':['hm', 'ht'],
        'T_freeze_taus':['ft', 'tt']
        }

    import itertools
    def get_configs(sweep_params, sweep_params_names):
        configurations = list(itertools.product(*sweep_params.values()))
        configurations_names = list(itertools.product(*sweep_params_names.values()))
        all_configs = []
        for config, config_names in zip(configurations, configurations_names):
            config_dict = dict(zip(sweep_params.keys(), config))
            config_dict['name']='_'.join(list(config_names))
            all_configs.append(config_dict)
        return all_configs

    cfgs = get_configs(sweep_params, sweep_params_names)

    num_repetitions = 1
    for cfg in cfgs:
        for repetition in range(0, num_repetitions):
            for key, value in zip(cfg.keys(), cfg.values()):
                if key != 'name':
                    if key.split('_')[0]=='T':
                        train_params[key[2:]] = value
                    else:
                        model_params[key] = value

            print('-----NEW TRAINING-------')
            print(model_params)
            print(train_params)

            snn = SNN(**model_params)
            snn.set_network()
            snn.model_name = cfg['name'] + '_rpt' + str(repetition)
            snn.save_model(snn.model_name + "_initial", ckpt_dir)

            train(snn, train_loader, test_loader, **train_params)

