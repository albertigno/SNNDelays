import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from snn_delays.utils.hw_aware_utils import pool_delays, quantize_weights, prune_weights, modify_weights
from snn_delays.utils.visualization_utils import training_plots
from torch.optim.lr_scheduler import StepLR
import torch.cuda.amp as amp
import numpy as np
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
import streamlit as st

def get_device():
    '''
    return current device
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on: {}'.format(device), flush=True)
    return device

def train(snn, train_loader, test_loader, learning_rate, num_epochs,
          lr_tau=0.1, scheduler=(1, 0.98), ckpt_dir='checkpoint',
          test_behavior=None, test_every=5, verbose=True, clear=False, **kwargs):
    """
    lr scale: originally I worked with same (1.0, 1.0 )lr for base (weights)
    tau_m, tau_adp
    then found tha for some nets its better to use different lr
    k, depth are to be set if you want truncated BPTT
    """

    tau_m_params = [param for name, param in snn.named_parameters() if 'tau' in name]
    weight_params = [param for name, param in snn.named_parameters() if 'linear' in name]

    weight_params = weight_params + [param for name, param in snn.named_parameters() if 'f' in name]

    if 'freeze_taus' in kwargs.keys():
        if kwargs['freeze_taus']:
            for param in tau_m_params:
                param.requires_grad = False

    optimizer = torch.optim.Adam([
        {'params': weight_params},
        {'params': tau_m_params, 'lr': lr_tau}],
        lr=learning_rate, eps=1e-5)
        
    step_size, gamma = scheduler[0], scheduler[1]
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    snn.ckpt_dir = ckpt_dir

    for epoch in range(num_epochs):
        
        start_time = time.time()

        current_lr = optimizer.param_groups[0]['lr']
        current_lr_tau = optimizer.param_groups[1]['lr']
        print('Epoch [%d/%d], learning_rates %f, %f' % (epoch + 1, num_epochs,
                                                         current_lr, current_lr_tau), flush=True)

        snn.train_step(train_loader, optimizer=optimizer, scheduler = scheduler, **kwargs)        

        if verbose:
            t = time.time() - start_time
            print('Time elasped:', t)

        if clear:
            clear_output(wait=True)

        # do the test every "test_every". if test_loader is a list, i.e [test_loader, train_loader],
        # test all the elements of the list
        dropout = 0.0
        if type(test_loader)==list:
            for loader in test_loader:
                test_behavior(snn, ckpt_dir, loader, test_every)
        else:        
            test_behavior(snn, ckpt_dir, test_loader, test_every)

    # by default, plot curves at the end of the training
    training_plots(snn)

    # empty the cuda cache after every training session
    torch.cuda.empty_cache()


def propagate_batch_simple(snn, data):
    
    '''
    data is either a train or a test loader
    '''

    for images, labels in data:
        snn.propagate(images, labels)
        break

    return images, labels


def check_dataloader(loader, batch_size, total_time):
    '''
    Use this to check the dimensions of the images and labels generates
    by the test or train loaders
    '''

    for images, labels in loader:

        # Resize and reformat of images and labels
        images = images > 0
        images = images.view(batch_size, total_time,
                                -1).float().squeeze()
        labels = labels.float()
        break

    print(f'shape of inputs is: {images.shape}')
    print(f'shape of labels is: {labels.shape}')


def calculate_metrics(all_refs, all_preds, print_per_class = False):
    """
    Function to calculate, print and save several metrics:
        - confusion matrix
        - precision
        - recall (or sensitivity)
        - f1 score

    :param test_loader: Test dataset loader (default = None)
    :param dropout: Parameter to calculate the dropout of the test images
    (default = 0.0)
    :param directory: Directory to save the model (relative to
    CHECKPOINT_PATH) (default = 'default')
    """
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(all_refs, all_preds)
    precision = precision_score(all_refs, all_preds, average='macro')
    recall = recall_score(all_refs, all_preds, average='macro')
    f1 = f1_score(all_refs, all_preds, average='macro')

    all_prec, all_rec, all_f1, support = precision_recall_fscore_support(all_refs, all_preds)
    print(conf_matrix)
    print(f'Precision: {precision}, Recall: {recall}, f1-score: {f1}')

    if print_per_class:
        print(f'Per class precisions: {all_prec}')
        print(f'Per class recalls: {all_rec}')
        print(f'Per class f1-scores: {all_f1}')
        print(f'Support: {support}')

    return f1


def calc_metrics(func):
    '''
    A wrapper to add the 'calculate metrics' functionality when it's needed
    '''
    def wrapper(*args, **kwargs):

        all_refs, all_preds = func(*args, **kwargs)

        calculate_metrics(all_refs, all_preds)

        return all_refs, all_preds

    return wrapper


def copy_snn(snn, new_batch_size=None):

    '''
    create a copy of a given snn, with a diferent batch size
    '''

    if new_batch_size is None:
        new_batch_size = snn.batch_size

    kwargs = snn.kwargs.copy()
    kwargs.pop('self', None)
    kwargs.pop('__class__', None)
    snn_type = type(snn)
    kwargs['batch_size'] = new_batch_size
    snn_copy = snn_type(**kwargs)
    snn_copy.set_network()
    snn_copy.load_state_dict(snn.state_dict())

    snn.to('cuda') ###fix this!

    stored_grads = get_gradients(snn)

    # Transfer parameters and their gradients
    for name, param in snn_copy.named_parameters():
        if name in stored_grads:
            param.grad = stored_grads[name].clone()

    return snn_copy

def transfer_weights_taus(source_snn, target_snn):

    weight_taus = [(name, w) for name, w  in target_snn.named_parameters() if 's' not in name]

    for (name_src, param_src), (name_dst, param_dst) in zip(source_snn.named_parameters(), weight_taus):
        assert name_src == name_dst, f"Parameter mismatch: {name_src} != {name_dst}"
        param_dst.data.copy_(param_src.data)

    return target_snn


def get_gradients(snn):
        # Store gradients before optimizer step
    stored_grads = {
        name: param.grad.clone() 
        for name, param in snn.named_parameters() 
        if param.grad is not None
    }

    return stored_grads


def print_spike_info(snn, layer):
    total_spikes = torch.sum(snn.spike_state[layer]).item()
    dim = snn.spike_state[layer].shape[-1]
    spk_per_sample = total_spikes/snn.batch_size
    spk_per_timestep = spk_per_sample/snn.win
    spk_per_neuron = spk_per_sample/dim
    spk_density = spk_per_timestep/(snn.win*dim)

    print(f'for {layer} layer')
    print(f'total spikes: {total_spikes}')
    print(f'spikes per sample: {spk_per_sample}')
    print(f'spikes per timestep: {np.round(spk_per_timestep, 2)} / {dim}')
    print(f'spikes per neuron: {np.round(spk_per_neuron, 2)} / {snn.win}')
    print(f'spike density: {spk_density}')


def to_plot(tensor):
    return tensor.detach().cpu().numpy()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


