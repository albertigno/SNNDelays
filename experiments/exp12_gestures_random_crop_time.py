from snn_delays.snn import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils import train, get_device
from snn_delays.utils.test_behavior import tb_save_max_last_acc
import torch

'''
first run: 1 layer
second run: 2 layer feedforward no delays
this was run with three different loss_fn: 'mem_sum', 'mem_mot', 'mem_last'
'''

device = get_device()
torch.manual_seed(10)

dataset = 'ibm_gestures'
total_time = 100
batch_size = 128

rpts = 3

# tried 1e5 but sometimes there are no spikes...

for rcrop in [1e5, 7e5, 3e6]:

    for test_cfg in ['randomtest', 'fixedtest']:

        # DATASET
        DL = DatasetLoader(dataset=dataset,
                        caching='memory',
                        num_workers=0,
                        batch_size=batch_size,
                        total_time=total_time,
                        sensor_size_to=64,
                        random_crop_to=(3e6, rcrop))
        train_loader, test_loader, dataset_dict = DL.get_dataloaders()

        if test_cfg == 'fixedtest':

            ### make test set fixed for reproducible results...? (hopefully)
            DL = DatasetLoader(dataset=dataset,
                            caching='memory',
                            num_workers=0,
                            batch_size=batch_size,
                            total_time=total_time,
                            sensor_size_to=64,
                            crop_to=rcrop)
            _, test_loader, _ = DL.get_dataloaders()

        ### fixed params

        model_params = {'dataset_dict': dataset_dict, 'delay_type':'h',
                        'reset_to_zero':True, 'win':total_time,
                        'loss_fn':'mem_sum', 'batch_size':batch_size, 'device':device,
                        'debug':False}

        ckpt_dir = 'exp12_gestures_random_crop_time'

        train_params = {'learning_rate':1e-3, 'num_epochs':50, 'spk_reg':0.0, 'l1_reg':0.0,
                'dropout':0.0, 'lr_tau': 0.1, 'scheduler':(10, 0.95), 'ckpt_dir':ckpt_dir,
                'test_behavior':tb_save_max_last_acc, 'test_every':1, 'delay_pruning':None, 'weight_pruning':None,
                'lsm':False, 'random_delay_pruning' : None, 'weight_quantization': None, 'k':None, 'depth': None, 'verbose':True}

        #### first run (f+d)
        sweep_params = {
            'connection_type': ['f'],
            'delay': [(48*2, 16*2)],
            'structure':[(64,2)],
            'tau_m':['normal'],
            'T_freeze_taus':[None]
            }

        sweep_params_names = {
            'connection_type': ['f'],
            'delay': ['d'],
            'structure':['2l'],
            'tau_m':['ht'],
            'T_freeze_taus':['tt']
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

        num_repetitions = rpts
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
                snn.input2spike_th = None 

                snn.model_name = test_cfg+str(rcrop)+'_'+cfg['name'] + '_rpt' + str(repetition)
                snn.save_model(snn.model_name + "_initial", ckpt_dir)

                train(snn, train_loader, test_loader, **train_params)

        #### second run (r)
        sweep_params = {
            'connection_type': ['r'],
            'delay': [None],
            'structure':[(64,2)],
            'tau_m':['normal'],
            'T_freeze_taus':[None]
            }

        sweep_params_names = {
            'connection_type': ['r'],
            'delay': ['nd'],
            'structure':['2l'],
            'tau_m':['ht'],
            'T_freeze_taus':['tt']
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

        num_repetitions = rpts
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
                snn.model_name = test_cfg+str(rcrop)+'_'+cfg['name'] + '_rpt' + str(repetition)
                snn.save_model(snn.model_name + "_initial", ckpt_dir)

                train(snn, train_loader, test_loader, **train_params)


        #### third run (f)
        sweep_params = {
            'connection_type': ['f'],
            'delay': [None],
            'structure':[(64,4)],
            'tau_m':['normal'],
            'T_freeze_taus':[None]
            }

        sweep_params_names = {
            'connection_type': ['f'],
            'delay': ['nd'],
            'structure':['4l'],
            'tau_m':['ht'],
            'T_freeze_taus':['tt']
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

        num_repetitions = rpts
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
                snn.model_name = test_cfg+str(rcrop)+'_'+cfg['name'] + '_rpt' + str(repetition)
                snn.save_model(snn.model_name + "_initial", ckpt_dir)

                train(snn, train_loader, test_loader, **train_params)