''''
common operations when loading datasets for analysis
'''
RESULTS_PATH = r'C:\Users\Alberto\OneDrive - UNIVERSIDAD DE SEVILLA\PythonData\Checkpoints'
from snn_delays.utils.model_loader import ModelLoader
from snn_delays.utils.train_utils import get_device, propagate_batch
from snn_delays.utils.hw_aware_utils import prune_weights
from IPython.display import clear_output
device = get_device()
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch 
from typing import List, Dict, Tuple, Any, Optional

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
        try:
            weights = torch.abs(getattr(snn,layer).weight.data.view(-1))
        except:
            weights = torch.abs(getattr(snn,layer).linear.weight.data.view(-1))

        threshold = torch.max(weights)*0.01
        percentage = torch.sum(weights<=threshold)/torch.sum(weights>=0)
        effective_params += torch.sum(weights>threshold).item()
        print(f'{threshold.item()}, {percentage.item()}%')
        prune_weights(snn, percentage.item(), layer_names=[layer])

    return params, effective_params

def get_results(ckpt_dir, sweep_params_names, rpts=3, mode='max', ablation_name=''):

    '''
    rpts: number of repetitions of the experiment
    mode: 'max' (best accuracy regardless of epoch) or 'last' (final accuracy)
    '''

    batch_size=64

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
            reference = f'{ablation_name}{model_config}_rpt{rpt}' if type(rpts)==int else f'{model_config}'
            model_loaded_flag = False
            for model_name in models:
                if reference in model_name and mode in model_name:
                    print(model_name)
                    snn = ModelLoader(
                        model_name, models_dir, batch_size, device, True)
                    clear_output(wait=True)
                    max_acc = snn.acc[-1][-1]
                    # spikes per timestep per neuron
                    print(snn.test_spk_count[-1][-1])
                    spike_density = len(snn.num_neurons_list)*snn.test_spk_count[-1][-1] / snn.win
                    # spikes per timestep in total
                    spike_per_time = spike_density*sum(snn.num_neurons_list)
                    model_loaded_flag = True

            if not(model_loaded_flag):
                    raise FileNotFoundError(f'model with reference {reference} not found')

            # num_params[model_config], num_eff_params[model_config] = get_param_count(snn)

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


# def get_states(ckpt_dir, sweep_params_names, rpts=3, mode='max', ablation_name='', loader=None, batch_size=None):

#     models_dir = os.path.join(RESULTS_PATH, ckpt_dir)
#     ### MODELS
#     models = []
#     for _, __, files in os.walk(models_dir, topdown=False):
#         for name in files:
#             if '.py' not in name:
#                 models.append(name)

#     mem_states = dict()
#     spike_states = dict()
#     refs = dict()
#     preds = dict()

#     configurations_names = list(itertools.product(*sweep_params_names.values()))

#     if type(rpts)==int:
#         num_rpts = rpts
#     else:
#         num_rpts = 1

#     for name in configurations_names:
#         model_config = '_'.join(list(name))

#         for rpt in range(num_rpts): 
            
#             # Load model with maximum acc
#             reference = f'{ablation_name}{model_config}_rpt{rpt}' if type(rpts)==int else f'{model_config}'
#             model_loaded_flag = False
#             for model_name in models:
#                 if reference in model_name and mode in model_name:
#                     print(model_name)
#                     snn = ModelLoader(
#                         model_name, models_dir, batch_size, device, True)
#                     clear_output(wait=True)
#                     model_loaded_flag = True

#             if not(model_loaded_flag):
#                     raise FileNotFoundError(f'model with reference {reference} not found')
            
#             snn.debug = True
#             ref, pred = snn.test(loader, only_one_batch=True)
            
#             if f'{model_config}' not in spike_states.keys():
#                 spike_states[model_config] = snn.spike_state
#                 mem_states[model_config] = snn.mem_state
#                 refs[model_config] = ref
#                 preds[model_config] = pred
#             else:
#                 spike_states[model_config].append(snn.spike_state)
#                 mem_states[model_config].append(snn.mem_state)
#                 refs[model_config].append(ref)
#                 preds[model_config].append(pred)

#     results = (spike_states, mem_states, refs, preds)

#     return results


### AI-enhanced
def get_states(
    ckpt_dir: str,
    sweep_params_names: Dict[str, List[Any]],
    attributes: List[str],  # List of attributes to extract (e.g., 'spike_state', 'mem_state')
    rpts: int = 3,
    mode: str = 'max',
    ablation_name: str = '',
    loader: Optional[Any] = None,
    batch_size: Optional[int] = None,
    device: str = 'cuda',  # Make the device configurable
) -> Tuple[Dict[str, List[Any]], ...]:
    """
    Extracts specified attributes from SNN models stored in a checkpoint directory.

    Args:
        ckpt_dir (str): Directory containing model checkpoints.
        sweep_params_names (Dict[str, List[Any]]): Dictionary of sweep parameters and their values.
        attributes (List[str]): List of model attributes to extract (e.g., 'spike_state', 'mem_state').
        rpts (int): Number of repetitions for each configuration.
        mode (str): Mode for selecting the best model (e.g., 'max' for maximum accuracy).
        ablation_name (str): Prefix for ablation studies.
        loader: Data loader for testing the model.
        batch_size (int): Batch size for testing.
        results_path (str): Base path for results directory.
        device (str): Device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[Dict[str, List[Any]], ...]: A tuple of dictionaries containing the extracted attributes.
    """

    models_dir = os.path.join(RESULTS_PATH, ckpt_dir)

    # Find all model files in the directory
    models = [
        name for _, __, files in os.walk(models_dir)
        for name in files
        if not name.endswith('.py')  # Exclude Python files
    ]

    # Initialize dictionaries to store results
    results = {attr: dict() for attr in attributes}
    results['refs'] = dict()  # Add refs to results
    results['preds'] = dict()  # Add preds to results

    # Generate all configurations from sweep parameters
    configurations_names = list(itertools.product(*sweep_params_names.values()))

    # Handle single repetition case
    num_rpts = rpts if isinstance(rpts, int) else 1

    # If need to do snn.test
    get_states = 'spike_state' in attributes or 'mem_state' in attributes

    for config_name in configurations_names:
        model_config = '_'.join(config_name)

        for rpt in range(num_rpts):
            # Construct model reference
            reference = f'{ablation_name}{model_config}_rpt{rpt}' if isinstance(rpts, int) else f'{model_config}'
            model_loaded_flag = False

            # Find and load the model
            for model_name in models:
                if reference in model_name and mode in model_name:
                    print(f'Loading model: {model_name}')
                    snn = ModelLoader(model_name, models_dir, batch_size, device, True)
                    
                    ######## TEMPORARY FIX to being unable to properly load MF-nets!!!
                    if 'mf' in model_name:
                        snn.multi_proj = 3
                    
                    #snn.use_amp = False
                    clear_output(wait=True)
                    model_loaded_flag = True
                    break

            if not model_loaded_flag:
                raise FileNotFoundError(f'Model with reference {reference} not found')

            
            if get_states:
                # Test the model
                #ref, pred = snn.test(loader, only_one_batch=True)
                ref, pred = propagate_batch(snn, loader)

            # Extract and store the specified attributes
            for attr in attributes:
                if not hasattr(snn, attr):
                    raise AttributeError(f'Model does not have attribute: {attr}')
                if model_config not in results[attr]:
                    results[attr][model_config] = []
                results[attr][model_config].append(getattr(snn, attr))

            if get_states:
                # Store references and predictions
                if model_config not in results['refs']:
                    results['refs'][model_config] = []
                    results['preds'][model_config] = []
                results['refs'][model_config].append(ref)
                results['preds'][model_config].append(pred)

    # Return results as a tuple
    return (*results.values(),)

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


