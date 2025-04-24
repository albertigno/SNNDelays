from snn_delays.snn_refactored import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.datasets.custom_datasets import CustomDataset
from snn_delays.utils.memory_cached_dataset import MemoryCachedDataset
from snn_delays.utils.train_utils_refact_minimal import train, get_device
from snn_delays.utils.test_behavior import tb_save_max_acc_refac
from snn_delays.config import DATASET_PATH
from torch.utils.data import DataLoader
import numpy as np
import multiprocessing
import os

device = get_device()

time_window = 64
batch_size = 64
num_epochs = 100 

ckpt_dir = 'abl10_letters4_temporal_field'

#data = np.load(os.path.join(DATASET_PATH, 'raw_datasets', 'Letters', 'letter_classification_dataset.npz'))
data = np.load(os.path.join(DATASET_PATH, 'Letters', 'four_letter_classification_dataset.npz'))

train_data = data['train_data']
test_data = data['test_data']
train_labels= data['train_labels']
test_labels = data['test_labels']

num_samples = len(train_labels)

train_dataset = CustomDataset(train_data, train_labels)
test_dataset = CustomDataset(test_data, test_labels)

dataset_dict = train_dataset.get_train_attributes()
dataset_dict["dataset_name"] = "letters4"

cached_train_dataset = MemoryCachedDataset(train_dataset, device=device)
cached_test_dataset = MemoryCachedDataset(test_dataset, device=device)

total_time = train_data.shape[1]
print(f'num timesteps per sample: {total_time}')
batch_size = 128

train_loader = DataLoader(cached_train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False,
                            pin_memory=False,
                            num_workers=0)

test_loader = DataLoader(cached_test_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False,
                            pin_memory=False,
                            num_workers=0)

model_params = {'dataset_dict': dataset_dict,
                 'win':time_window,
                 'tau_m': 'log-uniform-st',
                 'loss_fn':'mem_prediction', 
                 'batch_size':batch_size,
                 'device':device,
                 'debug':False,
                 'pruned_delays': 3}

train_params = {'learning_rate':1e-3, 'num_epochs':num_epochs, 'scheduler':(10, 0.95), 'ckpt_dir':ckpt_dir,
          'test_behavior':tb_save_max_acc_refac, 'test_every':1, 'freeze_taus':True}

#### first run (f+d)

delay_range = [3, 16, 48]

sweep_params = {
    'delay_range': [(x, 1) for x in delay_range],
    'structure':[(8, 2, 'd'), (64, 2, 'd')]
    }

sweep_params_names = {
    'delay_range': ['drf'+str(x) for x in delay_range],
    'structure':['h16', 'h64']
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

def train_model(cfg_id, repetition):

    cfg = cfgs[cfg_id]

    for key, value in zip(cfg.keys(), cfg.values()):
        if key != 'name':
            if key.split('_')[0] == 'T':
                train_params[key[2:]] = value
            else:
                model_params[key] = value

    snn = SNN(**model_params)
    snn.set_layers()
    snn.to(device)
    snn.model_name = cfg['name'] + '_rpt' + str(repetition)
    
    snn.initial_model_name = snn.model_name+f'_first_0epoch'
    snn.save_model(snn.initial_model_name, ckpt_dir)

    train(snn, train_loader, test_loader, **train_params)

# # Main function to manage parallel processes
if __name__ == "__main__":

    multiprocessing.set_start_method("spawn")

    num_repetitions = 2
    repetitions = range(num_repetitions)
    cfg_ids = range(len(cfgs))
    #configs = list(itertools.product(cfg_ids, repetitions))

    for repetition in repetitions:

        # Create and start processes
        processes = []

        for cfg_id in cfg_ids:
            process = multiprocessing.Process(target=train_model, args=(cfg_id,repetition))
            processes.append(process)
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()

        print(f"All training runs completed for rpt {repetition+1}/{num_repetitions}! ")