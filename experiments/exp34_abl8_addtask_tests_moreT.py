from snn_delays.snn import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.experimental_models.snn_delay_prun import P_DelaySNN
from snn_delays.utils.train_utils import train, get_device, propagate_batch, to_plot
from snn_delays.utils.test_behavior import tb_addtask
from snn_delays.utils.visualization_utils import plot_taus
import numpy as np
import multiprocessing

device = get_device()

time_window = 250
batch_size = 128 # 128: anil kag

ckpt_dir = 'abl8_addtask_st_T250'
#ckpt_dir = 'abl8_addtask_mt'
#ckpt_dir = 'abl8_addtask_lt'

dataset = 'addtask_episodic'

DL = DatasetLoader(dataset=dataset, caching='', num_workers=0, batch_size=batch_size, total_time=time_window)
train_loader, test_loader, dataset_dict = DL.get_dataloaders()
dataset_dict["time_ms"] = 5*2e3
#dataset_dict["time_ms"] = 150
#dataset_dict["time_ms"] = 5

model_params = {'dataset_dict': dataset_dict, 'delay_type':'',
                 'reset_to_zero':True, 'win':time_window,
                 'loss_fn':'mem_prediction', 'batch_size':batch_size, 'device':device,
                 'debug':False}

#lr_tau = 0.01 # default
lr_tau = 0.1 # variation for small tau

train_params = {'learning_rate':1e-3, 'num_epochs':3000, 'spk_reg':0.0, 'l1_reg':0.0,
          'dropout':0.0, 'lr_tau': lr_tau, 'scheduler':(100, 0.95), 'ckpt_dir':ckpt_dir,
          'test_behavior':tb_addtask, 'test_every':100, 'delay_pruning':None, 'weight_pruning':None,
          'lsm':False, 'random_delay_pruning' : None, 'weight_quantization': None, 'k':None, 'depth': None, 'verbose':False}

#parameters to join: value lists must have the same length

union = {
    'connection_type': ['r', 'mf', 'f', 'f'], 
    'delay': [None, None, None, (200, 5)],
    'delay_type': ['', '', '', 'h']
}

union_keys = [*union.keys()]

sweep_params_names = {
    'U_1': ['rnn','mf', 'f', 'rd'],
    'structure':['2l'],
    'tau_m':['ht'],
    'T_freeze_taus':['tt']
    }

# union = {
#     'connection_type': [ 'f'], 
#     'delay': [(40, 1)],
#     'delay_type': ['h']
# }

# union_keys = [*union.keys()]

# sweep_params_names = {
#     'U_1': ['rd'],
#     'structure':['2l'],
#     'tau_m':['ht'],
#     'T_freeze_taus':['tt']
#     }

sweep_params = {
    'U_1': list(zip(*union.values())),
    'structure':[(64,2)],
    'tau_m':['normal'],
    'T_freeze_taus':[None]
    }

# sweep_params = {
#     'U_1': list(zip(*union.values())),
#     'structure':[(64,2)],
#     'tau_m':['normal'],
#     'T_freeze_taus':[True, None]
#     }

# sweep_params_names = {
#     'U_1': ['rnn','mf', 'rd'],
#     'structure':['2l'],
#     'tau_m':['ht'],
#     'T_freeze_taus':['ft', 'tt']
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
            elif key.split('_')[0] == 'U':
                for u_k, u_v in zip(union_keys, value):
                    model_params[u_k] = u_v
            else:
                model_params[key] = value

    if model_params['connection_type'] == 'mf':
        snn = SNN(**model_params)
        snn.multi_proj = 3
    elif model_params['connection_type'] == 'f':
        model_params['n_pruned_delays'] = 3
        model_params['delay_mask'] = 'random'
        snn = P_DelaySNN(**model_params)
    elif model_params['connection_type'] == 'r':
        snn = SNN(**model_params)
    snn.set_network()
    snn.use_amp = False
    snn.input2spike_th = None
    snn.num_train_samples = batch_size
    snn.model_name = cfg['name'] + '_rpt' + str(repetition)
    train(snn, train_loader, test_loader, **train_params)

# # # Main function to manage parallel processes (without parallel repetitions)
# if __name__ == "__main__":

#     multiprocessing.set_start_method("spawn")

#     num_repetitions = 3
#     repetitions = range(num_repetitions)
#     cfg_ids = range(len(cfgs))
#     #configs = list(itertools.product(cfg_ids, repetitions))

#     for repetition in repetitions:

#         # Create and start processes
#         processes = []

#         for cfg_id in cfg_ids:
#             process = multiprocessing.Process(target=train_model, args=(cfg_id,repetition))
#             processes.append(process)
#             process.start()

#         # Wait for all processes to finish
#         for process in processes:
#             process.join()

#         print(f"All training runs completed for rpt {repetition+1}/{num_repetitions}! ")


# Main function to manage parallel processes (with parallel repetitions)

if __name__ == "__main__":

    multiprocessing.set_start_method("spawn")

    num_repetitions = 2
    repetitions = range(num_repetitions)
    cfg_ids = range(len(cfgs))
    configs = list(itertools.product(cfg_ids, repetitions))

    # Create and start processes
    processes = []

    for cfg_id, repetition in configs:
        process = multiprocessing.Process(target=train_model, args=(cfg_id,repetition))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    print(f"All training runs completed for rpt {repetition+1}/{num_repetitions}! ")