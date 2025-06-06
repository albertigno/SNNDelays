from snn_delays.snn import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils import train, get_device
from snn_delays.utils.test_behavior import tb_save_max_last_acc
from snn_delays.config import CHECKPOINT_PATH
import os
import torch
import multiprocessing

'''
demonstration of parallel ablation.
serial: 200MB 6'30 min (caching: ram) aprox 260 mins in total
serial: 1.4GB 5'30 min (caching: gpu ram) aprox 220 min in total
parallel (1 rpt): 1.7GB: 160 minutes total (caching: ram)
parallel (1 rpt): 3.9GB: ?? minutes total (caching: gpu ram)
parallel (2 rps): 3.2GB: 165 minutes total (num_workers=0)
'''

device = get_device()

dataset = 'shd'
total_time = 50
batch_size = 1024

# DATASET
DL = DatasetLoader(dataset=dataset,
                   caching='gpu',
                   num_workers=0,
                   batch_size=batch_size,
                   total_time=total_time,
                   crop_to=1e6)
train_loader, test_loader, dataset_dict = DL.get_dataloaders()

#### natural tau is 20ms, for a total duration of 1000ms, too small. I "contract" the duration to match tau's timescale
dataset_dict["time_ms"] = 1

### fixed params

model_params = {'dataset_dict': dataset_dict, 'delay_type':'',
                 'reset_to_zero':True, 'win':total_time,
                 'loss_fn':'mem_sum', 'batch_size':batch_size, 'device':device,
                 'debug':False}

ckpt_dir = 'abl1_shd50_mf_lt'
# If the directory do not exist, it is created
model_path = os.path.join(CHECKPOINT_PATH, ckpt_dir)
if not os.path.isdir(model_path):
    os.makedirs(model_path)

train_params = {'learning_rate':1e-3, 'num_epochs':100, 'spk_reg':0.0, 'l1_reg':0.0,
          'dropout':0.0, 'lr_tau': 0.1, 'scheduler':(10, 0.95), 'ckpt_dir':ckpt_dir,
          'test_behavior':tb_save_max_last_acc, 'test_every':1, 'delay_pruning':None, 'weight_pruning':None,
          'lsm':False, 'random_delay_pruning' : None, 'weight_quantization': None, 'k':None, 'depth': None, 'verbose':False}

#### first run (f+d)
sweep_params = {
    'connection_type': ['mf'],
    'delay': [None],
    'structure':[(64,2)],
    'tau_m':['normal'],
    'T_freeze_taus':[True, None]
    }

sweep_params_names = {
    'connection_type': ['mf'],
    'delay': ['nd'],
    'structure':['2l'],
    'tau_m':['ht'],
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

def train_model(cfg_id, repetition):

    cfg = cfgs[cfg_id]

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
    snn.multi_proj = 3
    snn.set_network()
    snn.model_name = cfg['name'] + '_rpt' + str(repetition)
    snn.save_model(snn.model_name + "_initial", ckpt_dir)
    train(snn, train_loader, test_loader, **train_params)

# ## SERIAL TRAINING
# num_repetitions = 1
# for repetition in range(0, num_repetitions):
#     for cfg_id in range(len(cfgs)):
#         train_model(cfg_id, repetition)

# # Main function to manage parallel processes
if __name__ == "__main__":

    multiprocessing.set_start_method("spawn")

    num_repetitions = 3
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