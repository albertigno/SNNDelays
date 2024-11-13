import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from snn_delays.utils.hw_aware_utils import pool_delays, quantize_weights, prune_weights, modify_weights
from torch.optim.lr_scheduler import StepLR
import torch.cuda.amp as amp
import numpy as np
import time

def get_device():
    '''
    return current device
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on: {}'.format(device), flush=True)
    return device

def train(snn, train_loader, test_loader, learning_rate, num_epochs, spk_reg=0.0, l1_reg=0.0,
          dropout=0.0, lr_tau=0.1, scheduler=(1, 0.98), ckpt_dir='checkpoint',
          test_behavior=None, test_every=5, delay_pruning = None, weight_pruning=None, lsm=False,
          random_delay_pruning = None, weight_quantization = None, k=None, depth= None, freeze_taus = None, 
          verbose=True):
    """
    lr scale: originally I worked with same (1.0, 1.0 )lr for base (weights)
    tau_m, tau_adp
    then found tha for some nets its better to use different lr
    k, depth are to be set if you want truncated BPTT
    """

    tau_m_params = [getattr(
        snn, name.split('.')[0]) for name, _ in snn.state_dict().items()
        if 'tau_m' in name]

    # print(tau_m_params)
    
    # tau_adp_params = [getattr(
    #     snn, name.split('.')[0]) for name, _ in snn.state_dict().items()
    #     if 'tau_adp' in name]

    if lsm:
        # Freeze all parameters except the last layer
        for name, param in snn.named_parameters():
            if 'o.weight' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        optimizer = torch.optim.Adam([param for param in snn.parameters() if param.requires_grad], lr=learning_rate)
    else:
        optimizer = torch.optim.Adam([
            {'params': snn.base_params},
            {'params': tau_m_params, 'lr': lr_tau}],
            lr=learning_rate, eps=1e-5)
        
    step_size, gamma = scheduler[0], scheduler[1]
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    if freeze_taus:
        for param in tau_m_params:
            param.requires_grad = False

    # act_fun = ActFun.apply
    print(f'training {snn.model_name} for {num_epochs} epochs...', flush=True)

    # get fixed masks for the random delays
    if random_delay_pruning:
        assert 'ho' in snn.delay_type , "random_delays only implemented for delay_type: iho, ho"
        assert type(random_delay_pruning) == int, "random delays must be int: average number of delays kept"
        proj_names_delays = snn.proj_names
        if 'i' in snn.delay_type:
            proj_names_delays = 'f0_f1' + proj_names_delays
        random_proj_mask = [] # list of random projection mask
        for proj in proj_names_delays:
            random_proj_mask.append(torch.rand(getattr(snn, proj).weight.shape)>((50-random_delay_pruning)/50))

    for epoch in range(num_epochs):
        
        start_time = time.time()

        current_lr = optimizer.param_groups[0]['lr']
        current_lr_tau = optimizer.param_groups[1]['lr']
        print('Epoch [%d/%d], learning_rates %f, %f' % (epoch + 1, num_epochs,
                                                         current_lr, current_lr_tau), flush=True)


        if k==None:
            snn.train_step(train_loader,
                        optimizer=optimizer,
                        scheduler = scheduler,
                        spk_reg=spk_reg,
                        l1_reg=l1_reg,
                        dropout=dropout,
                        verbose=verbose)        
        else:
            snn.train_step_tr(train_loader=train_loader, optimizer=optimizer,
                            criterion=snn.criterion, spk_reg=spk_reg,
                            depth=depth, k=k, last=False)
         
        if weight_quantization is not None:
            assert (type(weight_quantization) == tuple and type(weight_quantization[-1]) == int), "weight_quantization must be a N-tuple that contains the N-params of quantize_weights() in hw_aware_utils.py plus the frequency in epochs of applying weight quantization"
            if snn.epoch % weight_quantization[-1] == 0:
                print(f'in-training weight quantization applied -> {weight_quantization[0]} bit', flush=True)
                quantize_weights(snn, *weight_quantization[:-1])

        if delay_pruning:
            assert type(delay_pruning) == tuple and len(delay_pruning)==5, "delay_pruning must be a 5-tuple with the 4 params of pool_delays() in utils.py plus the frequency in epochs of applying delay pruning"
            if (snn.epoch) % delay_pruning[-1] == 0:
                print(f'pruning {delay_pruning[2]} for layers {delay_pruning[1]}', flush=True)
                pool_delays(snn, delay_pruning[0], delay_pruning[1], delay_pruning[2], delay_pruning[3])

        if weight_pruning:
            if snn.epoch % weight_pruning[-1] == 0:
                print(f'pruning {weight_pruning[0]*100}% for layers {weight_pruning[1]}', flush=True)
                prune_weights(snn, weight_pruning[0], weight_pruning[1])

        if random_delay_pruning:
            for proj, mask in zip(proj_names_delays, random_proj_mask):
                modify_weights(getattr(snn, proj), mask, 'mask')

        if verbose:
            t = time.time() - start_time
            print('Time elasped:', t)

        # # update scheduler (adjust learning rate)
        # if scheduler:
        #     # optimizer = snn.lr_scheduler(optimizer, lr_decay_epoch=10) bojian
        #     optimizer = snn.lr_scheduler(
        #         optimizer=optimizer, lr_decay_epoch=scheduler[0], lr_decay=scheduler[1])

        # do the test every "test_every". if test_loader is a list, i.e [test_loader, train_loader],
        # test all the elements of the list
        if type(test_loader)==list:
            for loader in test_loader:
                test_behavior(snn, ckpt_dir, loader, dropout, test_every)
        else:        
            test_behavior(snn, ckpt_dir, test_loader, dropout, test_every)

    # empty the cuda cache after every training session
    torch.cuda.empty_cache()

def propagate_batch(snn, data, dropout = 0.0):
    
    '''
    data is either a train or a test loader
    '''



    dropout = torch.nn.Dropout(p=dropout, inplace=False)

    with amp.autocast(enabled=snn.use_amp):

        for images, labels in data:

            images = dropout(images.float())
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
    snn_copy.load_state_dict(snn.state_dict())

    stored_grads = get_gradients(snn)

    # Transfer parameters and their gradients
    for name, param in snn_copy.named_parameters():
        if name in stored_grads:
            param.grad = stored_grads[name].clone()

    return snn_copy

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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)