''''
common operations when loading datasets for analysis
'''
RESULTS_PATH = r'C:\Users\Alberto\OneDrive - UNIVERSIDAD DE SEVILLA\PythonData\Checkpoints'
from snn_delays.utils.model_loader import ModelLoader
from snn_delays.utils.train_utils import get_device
from snn_delays.utils.hw_aware_utils import prune_weights
from IPython.display import clear_output
device = get_device()
batch_size = 256
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch 

def get_param_count(snn):
    
    delays = len(snn.delays)
    params = 700*snn.num_neurons_list[0]

    for i, n in enumerate(snn.num_neurons_list[1:]):
        params += delays*snn.num_neurons_list[i-1]*n
        if snn.connection_type == 'r':
            params += delays*snn.num_neurons_list[i-1]**2
        
    params += delays*20*snn.num_neurons_list[-1]
    if snn.connection_type == 'r':
        params += delays*snn.num_neurons_list[-1]**2

    effective_params = 0
    for layer in ['f0_f1']+snn.proj_names:
        weights = torch.abs(getattr(snn,layer).weight.data.view(-1))
        threshold = torch.max(weights)*0.01
        percentage = torch.sum(weights<=threshold)/torch.sum(weights>=0)
        effective_params += torch.sum(weights>threshold).item()
        print(f'{threshold.item()}, {percentage.item()}%')
        prune_weights(snn, percentage.item(), layer_names=[layer])

    return params, effective_params

def get_results(ckpt_dir, sweep_params_names, rpts=3, mode='max'):

    '''
    rpts: number of repetitions of the experiment
    mode: 'max' (best accuracy regardless of epoch) or 'last' (final accuracy)
    '''

    models_dir = os.path.join(RESULTS_PATH, ckpt_dir)
    ### MODELS
    models = []
    for _, __, files in os.walk(models_dir, topdown=False):
        for name in files:
            if '.py' not in name:
                models.append(name)

    ### LOAD RESULTS
    acc = dict()
    spk = dict()
    spk_t= dict()
    train_loss = dict()
    test_loss = dict()
    num_params = dict()
    num_eff_params = dict()

    configurations_names = list(itertools.product(*sweep_params_names.values()))

    if type(rpts)==int:
        num_rpts = rpts
    else:
        num_rpts = 1

    for name in configurations_names:
        model_config = '_'.join(list(name))

        for rpt in range(num_rpts): 
            
            # Load model with maximum acc
            reference = f'{model_config}_rpt{rpt}' if type(rpts)==int else f'{model_config}'
            model_loaded_flag = False
            for model_name in models:
                if reference in model_name and mode in model_name:
                    print(model_name)
                    snn = ModelLoader(
                        model_name, models_dir, batch_size, device, True)
                    clear_output(wait=True)
                    max_acc = snn.acc[-1][-1]
                    # spikes per timestep per neuron
                    spike_density = len(snn.num_neurons_list)*snn.test_spk_count[-1][-1] / snn.win
                    # spikes per timestep in total
                    spike_per_time = spike_density*sum(snn.num_neurons_list)
                    model_loaded_flag = True

            if not(model_loaded_flag):
                    raise FileNotFoundError(f'model with reference {reference} not found')
            
            num_params[model_config], num_eff_params[model_config] = get_param_count(snn)

            # Save results acc
            if f'{model_config}' not in acc.keys():            
                acc[model_config] = [max_acc]
                spk[model_config] = [spike_density]
                spk_t[model_config] = [spike_per_time]
                train_loss[model_config] = [snn.train_loss]
                test_loss[model_config] = [snn.test_loss]
            else:
                acc[model_config].append(max_acc)
                spk[model_config].append(spike_density)
                spk_t[model_config].append(spike_per_time)
                train_loss[model_config].append(snn.train_loss)
                test_loss[model_config].append(snn.test_loss)

    results = (acc, spk, spk_t, train_loss, test_loss, num_params, num_eff_params)

    return results

def get_gap_losses(results_test_loss, results_train_loss):
    '''
    obtain the average gap between test and train losses per model_config
    '''
    gaps = dict()
    for (key1, val_test), (key2, val_train) in zip(results_test_loss.items(), results_train_loss.items()):
        mean_test_loss = np.mean(np.array(val_test), axis=0)
        mean_train_loss = np.mean(np.array(val_train), axis=0)
        gap = np.zeros(mean_test_loss.shape)
        for i, epoch in enumerate(mean_test_loss[:,0]):
            gap[i, 0] = int(epoch)
            i_train = int(np.argmax(mean_train_loss[:,0]==epoch))
            gap[i, 1] = mean_test_loss[i, 1] - mean_train_loss[i_train, 1]
        gaps[key1] = gap
    return gaps


def split_results(results):
    '''
    split in four groups f_d, f_nd, r_d, r_nd
    '''
    results_r_d = {key: results[key] for key in results if ('r_'  in key and '_d' in key)}
    results_r_nd = {key: results[key] for key in results if ('r_'  in key and '_nd' in key)}
    results_f_d = {key: results[key] for key in results if ('f_'  in key and '_d' in key)}
    results_f_nd = {key: results[key] for key in results if ('f_'  in key and '_nd' in key)}

    return results_r_d, results_r_nd, results_f_d, results_f_nd

def get_avgs(results):
    return {key: sum(value)/len(value) for key, value in results.items()}

def plot_bars(data, features, method = 'normal'):

    def plot_multiple_bars(y_values, labels1, labels2):
        # Define the data for each bar
        x = range(1, len(y_values[0]) + 1)  # x positions for each group of bars
        xx = range(1, len(labels1) + 1) 
        # Set the width of each bar
        width = 0.1
        # Create the bar plot
        for i, y in enumerate(y_values):
            q = [j + i * width for j in x]
            plt.bar(q, y, width=width, label=labels2[i])
        plt.xticks([j + width * (len(y_values) - 1) / 2 for j in xx], labels1, rotation=45)  # Set rotation to 45 degrees
        # Add a legend
        plt.legend()

        return plt.gca()

    def get_values(y, *params):
        y_values = []
        x_values = []
        for key in y.keys():
            if all(param in key.split('_') for param in params):
                y_values.append(y[key])
                k = '_'.join(key.split('_'))
                for param in params:
                    k = k.replace(param+'_', '')
                x_values.append(k)
        label = '_'.join(params)
        return x_values, y_values, label
    
    def get_values_by_name(y, name):
        y_values = []
        x_values = []
        for key in y.keys():
            if name in key:
                y_values.append(y[key])
                k = '_'.join(key.split('_'))
                k = k.replace(name, '')
                x_values.append(k)
        label = '_'.join(name)
        #ax = plt.bar(x_values, y_values)
        return x_values, y_values, label

    y_list = list()
    label_list = list()

    for feature in features:
        if method == 'normal':
            x, y, l = get_values(get_avgs(data), feature)
        elif method == 'by_name':
            x, y, l = get_values_by_name(get_avgs(data), feature)
        y_list.append(y)
        label_list.append(l)
    
    plot_multiple_bars(y_list, x, label_list)
    return plt.gca()