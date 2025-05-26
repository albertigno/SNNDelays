from snn_delays.snn_refactored import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils_refact_minimal import train, get_device
from snn_delays.utils.test_behavior import tb_save_max_last_refact
import torch
import multiprocessing
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

device = get_device()

dataset = 'shd'
total_time = 50
batch_size = 1024

ckpt_dir = 'abl1_shd50_refact_new_weight_init'

# DATASET
DL = DatasetLoader(dataset=dataset,
                   caching='memory',
                   num_workers=0,
                   batch_size=batch_size,
                   total_time=total_time,
                   crop_to=1e6)
train_loader, test_loader, dataset_dict = DL.get_dataloaders()

### fixed params

model_params = {'dataset_dict': dataset_dict, 'win':total_time,
                 'loss_fn':'mem_sum', 'batch_size':batch_size, 'device':device}

train_params = {'learning_rate':1e-3, 'num_epochs':5, 'spk_reg':0.0, 'lr_tau': 0.1,
                 'scheduler':(10, 0.95), 'ckpt_dir':ckpt_dir, 'test_behavior':tb_save_max_last_refact,
                'freeze_taus':True, 'test_every':1, 'clear':False, 'verbose':False}

#### first run (f+d)
sweep_params_mf = {
    'structure':[(64, 2, 'mf')],
    'tau_m':[20.0, 'log-uniform-st', 'log-uniform'],
    }

sweep_params_names_mf = {
    'structure':['mf'],
    'tau_m':['fx', 'st', 'lt'],
    }

extra_kwargs_mf = {'multifeedforward':3}


#### first run (f+d)
sweep_params_r = {
    'structure':[(64, 2, 'r')],
    'tau_m':[20.0, 'log-uniform-st', 'log-uniform'],
    }

sweep_params_names_r = {
    'structure':['r'],
    'tau_m':['fx', 'st', 'lt'],
    }

#### first run (f+d)
sweep_params_rd = {
    'structure':[(64, 2, 'd')],
    'tau_m':[20.0, 'log-uniform-st', 'log-uniform'],
    }

sweep_params_names_rd = {
    'structure':['rd'],
    'tau_m':['fx', 'st', 'lt'],
    }

extra_kwargs_rd = {'delay_range':(40, 1),
                'pruned_delays': 3}


cfgs_mf = get_configs(sweep_params_mf, sweep_params_names_mf)
cfgs_r = get_configs(sweep_params_r, sweep_params_names_r)
cfgs_rd = get_configs(sweep_params_rd, sweep_params_names_rd)

def train_model(cfgs_arch, cfg_id, repetition):

    cfg = cfgs_arch[cfg_id]

    for key, value in zip(cfg.keys(), cfg.values()):
        if key != 'name':
            if key.split('_')[0]=='T':
                train_params[key[2:]] = value
            else:
                model_params[key] = value
    print('-----NEW TRAINING-------')
    print(model_params)
    print(train_params)

    if model_params['structure'][-1] == 'mf':
        model_params.update(extra_kwargs_mf)
    if model_params['structure'][-1] == 'd':
        model_params.update(extra_kwargs_rd)


    snn = SNN(**model_params)
    snn.set_layers()
    snn.to(device)
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

    num_repetitions = 2
    repetitions = range(num_repetitions)
    cfg_ids = range(len(cfgs_rd))
    #configs = list(itertools.product(cfg_ids, repetitions))

    for repetition in repetitions:

        for cfg_arch in [cfgs_mf, cfgs_r, cfgs_rd]:
            print(f"Starting training for repetition {repetition+1}/{num_repetitions} with architecture {cfg_arch[0]['structure'][-1]}...")

            # Create and start processes
            processes = []

            for cfg_id in cfg_ids:
                process = multiprocessing.Process(target=train_model, args=(cfg_arch, cfg_id, repetition))
                processes.append(process)
                process.start()

            # Wait for all processes to finish
            for process in processes:
                process.join()

        print(f"All training runs completed for rpt {repetition+1}/{num_repetitions}! ")