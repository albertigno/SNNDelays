from snn_delays.snn import SNN
from snn_delays.datasets.custom_datasets import CustomDataset
from snn_delays.utils.train_utils import train, get_device
from snn_delays.utils.test_behavior import tb_save_max_last_acc
import torch
from snn_delays.config import DATASET_PATH
from torch.utils.data import DataLoader
import os

'''
first run: 1 layer
second run: 2 layer feedforward no delays
this was run with three different loss_fn: 'mem_sum', 'mem_mot', 'mem_last'
'''

device = get_device()
torch.manual_seed(10)

from tonic import MemoryCachedDataset
import numpy as np

#data = np.load(os.path.join(DATASET_PATH, 'raw_datasets', 'Letters', 'letter_classification_dataset.npz'))
data = np.load(os.path.join(DATASET_PATH, 'raw_datasets', 'Letters', 'three_letter_classification_dataset.npz'))

train_data = data['train_data']
test_data = data['test_data']
train_labels= data['train_labels']
test_labels = data['test_labels']

num_samples = len(train_labels)

train_dataset = CustomDataset(train_data, train_labels)
test_dataset = CustomDataset(test_data, test_labels)

dataset_dict = train_dataset.get_train_attributes()

cached_train_dataset = MemoryCachedDataset(train_dataset)
cached_test_dataset = MemoryCachedDataset(test_dataset)

total_time = train_data.shape[1]
print(f'num timesteps per sample: {total_time}')
batch_size = 512

train_loader = DataLoader(cached_train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=0)

test_loader = DataLoader(cached_test_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=0)

dataset_dict["time_ms"] = 1e3
dataset_dict["dataset_name"] = "3letters"

print(dataset_dict)

### fixed params

model_params = {'dataset_dict': dataset_dict, 'delay_type':'h',
                 'reset_to_zero':True, 'win':total_time,
                 'loss_fn':'mem_sum', 'batch_size':batch_size, 'device':device,
                 'debug':False}

ckpt_dir = 'abl5_4letters'

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
    'delay': ['d'],
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
    'structure':[(64,2), (64,4)],
    'tau_m':[20.0, 'normal'],
    'T_freeze_taus':[True, None]
    }

sweep_params_names = {
    'connection_type': ['f'],
    'delay': ['nd'],
    'structure':['2l', '4l'],
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