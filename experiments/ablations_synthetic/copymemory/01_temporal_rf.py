from snn_delays.snn_refactored import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils_refact_minimal import train, get_device
from snn_delays.utils.test_behavior import tb_synthetic_refact
import numpy as np
import multiprocessing

device = get_device()

time_window = 50
batch_size = 64
num_epochs = 3000 # iterations steps
dataset_size = batch_size*num_epochs

ckpt_dir = 'abl9_copy_taskT50'
dataset = 'copymemory_episodic'

DL = DatasetLoader(dataset=dataset, caching='gpu', 
                   dataset_size = dataset_size,
                   num_workers=0, batch_size=batch_size, 
                   total_time=time_window)

train_loader, test_loader, dataset_dict = DL.get_dataloaders()

model_params = {'dataset_dict': dataset_dict,
                 'win':time_window,
                 'tau_m': 'log-uniform-st',
                 'loss_fn':'mem_prediction', 
                 'batch_size':batch_size,
                 'device':device,
                 'debug':False,
                 'pruned_delays': 3}

train_params = {'learning_rate':1e-3, 'num_epochs':1, 'scheduler':(100, 0.95), 'ckpt_dir':ckpt_dir,
          'test_behavior':tb_synthetic_refact, 'test_every':100, 'printed_steps':100, 'freeze_taus':True}

#### first run (f+d)
# sweep_params = {
#     'delay_range': [(x, 1) for x in [3, 5]],
#     'structure':[(8, 2, 'd'), (64, 2, 'd')]
#     }

# sweep_params_names = {
#     'delay_range': ['drf'+str(x) for x in [3, 5]],
#     'structure':['h8', 'h64']
#     }

# #### first run (f+d)
# sweep_params = {
#     'delay_range': [(x, 1) for x in [10, 15, 20]],
#     'structure':[(8, 2, 'd'), (64, 2, 'd')]
#     }

# sweep_params_names = {
#     'delay_range': ['drf'+str(x) for x in [10, 15, 20]],
#     'structure':['h8', 'h64']
#     }


#### first run (f+d)
sweep_params = {
    'delay_range': [(x, 1) for x in [25, 30, 35]],
    'structure':[(8, 2, 'd'), (64, 2, 'd')]
    }

sweep_params_names = {
    'delay_range': ['drf'+str(x) for x in [25, 30, 35]],
    'structure':['h8', 'h64']
    }


# sweep_params = {
#     'delay_range': [(x, 1) for x in [40, 45, 50]],
#     'structure':[(8, 2, 'd'), (64, 2, 'd')]
#     }

# sweep_params_names = {
#     'delay_range': ['drf'+str(x) for x in [40, 45, 50]],
#     'structure':['h8', 'h64']
#     }


### second run (f+d)
# sweep_params = {
#     'delay_range': [(x, 1) for x in range(60, 105, 5)],
#     'structure':[(8, 2, 'd'), (64, 2, 'd')]
#     }

# sweep_params_names = {
#     'delay_range': ['drf'+str(x) for x in range(60, 105, 5)],
#     'structure':['h8', 'h64']
#     }

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