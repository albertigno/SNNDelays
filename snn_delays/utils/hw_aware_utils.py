import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
from snn_delays.config import CHECKPOINT_PATH


# TODO: ¿Qué hace esta funcion?
#  Terminar documentacion y testar
def modify_weights(layer, value, mode='mask', trainable=True):
    """
    Function to mask weights

    :param layer: a snn layer (e.g: )
    :param value: a Tensor or numpy array
    :param mode: ily current weights by value, 
                 if 'replf 'mask' multipace' assign new value to weights 
    :param trainable: (default = True)
    """

    #print(layer.device)
    #device = layer.device

    value = torch.Tensor(value).to(layer.weight.device)

    if layer.weight.data.shape == value.shape:
        new_weight = value if mode=='replace' else layer.weight.data * value
        layer.weight = torch.nn.Parameter(
            new_weight, requires_grad=trainable)
    else:
        print(f'Mask weights failed: dimension mismatch. make sure the '
              f'weights are shape {layer.weight.data.shape}')

def scale_weights(layer, scale_factor):
    value = scale_factor*layer.weight.data
    layer.weight = torch.nn.Parameter(
        value, requires_grad=True)    
    #modify_weights(layer, value, 'replace')


"""
HW_Aware_utils

This script includes methods such as weight quantization, pruning, delay 
pooling, etc.
"""
def quantize_weights(snn, bits, last_layer=False, symmetry=True, print_info=False):
    """
    This function quantize the weights of all the layers of the neural
    network.

    The bits param determines the method of quantization. if it is an int
    it implie linear quantization which is based on the histogram method. As
    a string in may define all sorts of other methods: 'bf16' (brainfloat16),
    'lsq8' (learned step 8-bit quantization), 'fixed4.8' (fixed point 4bit
    int, 8bit fractional), etc.

    :param snn: The network which weight will be pruned.
    :param bits: (int) Number of bits to scale the weights, (str) 'bf16'
    for brainfloat16
    :param last_layer: Boolean to apply (True) or not (False) the
    quantitation to the last layer of the network (default = False)
    :param symmetry: (False) No weight symmetry is enforced, (True) weights are
    quantized such that the number of levels/bins are equal on the positive and
    negative axis
    :param test: Boolean to print information about initial and final weights
    for each layer (default = False)
    """

    def hist_quantize(_weights, _num_bins, _symmetry, _max_w=None):
        """
        Auxiliary function to reduce the precision when weights are
        quantized using the histogram quantization method.

        :param _weights: The tensor of weights to be quantized
        :param _num_bins: Number of bits to scale the weights
        :param _symmetry: Quantization bins are made symetric (same number of
        positives and negatives)
        return: The tensor of weights quantized
        """

        # Flatten the tensor to a 1D array
        values = _weights.flatten().detach().cpu().numpy()

        # Calculate the histogram of the values
        if _symmetry:
            if _max_w is None:
                vmax = np.max(np.abs(values))
            else:
                vmax = _max_w*(1 - 1/(_num_bins-1))
            b = np.linspace(-vmax, vmax, _num_bins-1) 
            _, bin_edges = np.histogram(values, bins=b, range=(-vmax, vmax))
        else:
            _, bin_edges = np.histogram(values, bins=_num_bins-1)

        # Calculate the center of each bin
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        if not _symmetry:
            offset = bin_centers[np.argmin(np.abs(bin_centers))]
            bin_edges = bin_edges - offset
            bin_centers = bin_centers - offset

        #ind = np.digitize(values, bin_edges[:-1], right=True)
        ind = np.digitize(values, bin_edges[:-1])

        quantized_values = np.array([bin_centers[x - 1] for x in ind])

        # Reshape the quantized values back to the original shape of the
        # tensor
        quantized_tensor = torch.tensor(
            quantized_values.reshape(_weights.shape))

        return quantized_tensor
    
   
    # Define the number of bins for the histogram in int quantization
    if type(bits) == int:
        num_bins = 2 ** bits + 1

    # Get the name of all the layers
    layer_names = copy.deepcopy( snn.__dict__['base_params_names'] )   # otherwise "if not last_layer:" below will modify the orig list

    # Don't apply quantization in the last layer if last_layer=False
    if not last_layer:
        layer_names.pop()

    # Deactivate grad to quantize
    with torch.no_grad():

        # Loop over all the layers
        for _layer in layer_names:

            # Get the layer and their weights
            layer = getattr(snn, _layer)

            # max_weight = 0.16384
            max_weight = torch.max(torch.abs(layer.weight.data)).cpu().numpy()

            # Quantize weights of the layers
            if type(bits) == int:                # int histogram-based
                new_weights = hist_quantize(layer.weight.data, num_bins, symmetry, max_weight).to(torch.float32)
            elif bits == 'bf16':                 # brain float 16bit
                new_weights = layer.weight.data.to(torch.bfloat16).to(torch.float32)
            else:
                raise (f'quantization {bits} not supported (yet)')


            # Plot and print data
            if print_info:
                print(f'----{_layer}----')
                u_nl = torch.unique(new_weights)
                u_ol = torch.unique(layer.weight)

                print(f'n_unique before quantization: {u_ol.size()[0]}, {(u_ol>0).sum().item()} pos {(u_ol<0).sum().item()} neg')
                print(f'max_value before quantization: {max(abs(u_ol)).item()}' )
                print(f'n_unique after quantization: {u_nl.size()[0]}, {(u_nl>0).sum().item()} pos {(u_nl<0).sum().item()} neg')
                print(f'max_value after quantization: {max(abs(u_nl)).item()}, delta_w: {min(u_nl[u_nl>0]).item()}' )

            # Update the weights after quantization
            modify_weights(layer, new_weights, 'replace')
            # setattr(layer.weight, 'data',
            #         torch.nn.Parameter(new_weights, requires_grad=True))


def prune_weights(snn, percentage, layer_names = [], test=False):
    """
    This function select the percentage of weight per layer with lower
    value and turn them to zero. With this, the number of parameters in the
    model and the computation time are reduced.

    :param snn: The network which weight will be pruned.
    :param percentage: Percentage of weights per layer to reduce to zero.
    It must be a decimal (e.g. 0.1 for 10%).
    :param test: Boolean for plot initial and final weights of each layer
    and print information about them
    """
    # Get the name of all the layers
    #layer_names = snn.__dict__['base_params_names']

    # Loop over all the layers
    for _layer in layer_names:

        # Get the layer and their weights
        layer = getattr(snn, _layer)
        weights = getattr(layer, 'weight')

        # Indexes of the lowest weights
        k = int(weights.numel() * percentage)
        sort_values, indices = torch.sort(
            torch.abs(weights.view(-1)), dim=-1, descending=False)

        # Create a mask of the same shape as the input tensor
        mask = torch.ones_like(weights)

        # Set the elements corresponding to the lowest percentage to 0
        mask.view(-1)[indices[:k]] = 0

        # Create a new tensor with the lowest elements set to 0
        new_tensor = weights * mask

        # Plot and print data about test
        if test:
            # # Plot the initial and new weights to compare
            # plt.plot(
            #     weights.view(-1).detach().cpu().numpy(), label='initial')
            # plt.plot(
            #     new_tensor.view(-1).detach().cpu().numpy(), label='final')
            # plt.legend()
            # plt.show()

            # Count non-zero values to compare
            non_zero_count = torch.sum(weights != 0).item()
            print('Number of non-zero elements in old_tensor: ',
                    non_zero_count)
            non_zero_count = torch.sum(new_tensor != 0).item()
            print('Number of non-zero elements in new_tensor: ',
                    non_zero_count)

        # Change the weights of the layer
        setattr(getattr(snn, _layer), 'weight',
                torch.nn.parameter.Parameter(new_tensor,
                                                requires_grad=True))



def pool_delays(snn, mode='synaptic', lyr='iho', k=1, freeze=True):
    """
    Function to create one delay per synapse in multi-delay model, by
    choosing the one with the highest absolute value.

    :param mode: 'synaptic' or 'axonal'
    :param snn: The network which delays will be pooled.
    :param lyr: Lyrics to select which layers are going to be pooled. It
        can take the value 'i', 'h', 'o', or a combination of these three
        lyrics; e.g. 'ho' or 'iho' (default = 'i').
    :param k: Number of delays to be selected (default = 1)
    :param freeze: Boolean to control the training (default = True)

    """

    def get_pooling_mask_syn(_w):
        """
        Auxiliary function to get a pooling mask.

        :param _w: Weights to be pooled
        :return:
        """

        # Initialize the mask
        _mask = torch.zeros(_w.shape, device=snn.device)

        # Absolute value of weight tensor
        ww = torch.abs(_w)

        # Loop over all the weights
        for i in range(_w.shape[0]):
            for j in range(_w.shape[1]):

                # Find the indices of the k-highest values
                idx_k = np.argpartition(ww[i,j,:].cpu().numpy(), -k)[-k:]

                # Set the value of the mask
                for d in idx_k:
                    _mask[i, j, d] = 1.0
        return _mask
    
    def get_pooling_mask_axn(_w):
        """
        Auxiliary function to get a pooling mask (axonal delays).
        Axonal delay pooling selects the top k delays from each pre-synaptic
        neuron, based on the l1-norm.

        :param _w: Weights to be pooled
        :return:
        """

        # Initialize the mask
        _mask = torch.zeros(_w.shape, device=snn.device)

        # Absolute value of weight tensor
        ww = torch.abs(_w)

        # get most important delays per presynaptic neuron
        # the importance is given by the sum of the absolute values
        www = np.sum(ww.cpu().numpy(), axis=0)

        # Loop over all the input neurons
        for j in range(_w.shape[1]):

            # Find the indices of the k-highest values
            idx_k = np.argpartition(www[j,:], -k)[-k:]

            # print(idx_k)

            # Set the value of the mask
            for d in idx_k:
                _mask[:, j, d] = 1.0
        return _mask

    def get_pooling_mask_axn_layerwise(_w):
        """
        Auxiliary function to get a pooling mask (axonal delays).
        Layerwise axonal delays selects the top k*number_presyn_neurons
        delays considering all pre-synaptic neurons together.
        :param _w: Weights to be pooled
        :return:
        """

        # Absolute value of weight tensor
        ww = torch.abs(_w)

        # Initialize the mask
        _mask = torch.zeros(_w.shape, device=snn.device)

        num_pre = _w.shape[1]
        num_delays = _w.shape[2]
        # get most important delays per presynaptic neuron
        # the importance is given by the sum of the absolute values

        www = np.sum(ww.cpu().numpy(), axis=0).reshape(num_pre*num_delays)

        temp_mask = torch.zeros(www.shape, device=snn.device)

        idx_k = np.argsort(www)[::-1][:k*num_pre]

        idx_k = idx_k.copy()

        temp_mask[idx_k] = 1.0

        temp_mask = temp_mask.reshape(num_pre,num_delays)

        for j in range(temp_mask.shape[0]):
            for d in range(temp_mask.shape[1]):
                    _mask[:, j, d] = temp_mask[j, d]

        return _mask

    if mode == 'synaptic':
        get_pooling_mask = get_pooling_mask_syn
    elif mode == 'axonal':
        get_pooling_mask = get_pooling_mask_axn
    elif mode == 'axonal_variable':
        get_pooling_mask = get_pooling_mask_axn_layerwise
    # elif mode == 'dropdelays':
    #     get_pooling_mask = get_pooling_mask_dropdelays
        

    # Set trainable option
    trainable = not freeze

    # Get the number of delay values
    num_d = len(snn.delays)
    
    if 'i' in lyr and 'i' in snn.delay_type:

        w = snn.f0_f1.weight.data.reshape(
            snn.num_neurons_list[0], snn.num_input, num_d)
        mask = get_pooling_mask(w)
        modify_weights(snn.f0_f1, mask.reshape(
            snn.num_neurons_list[0], snn.num_input*num_d), mode='mask',
                            trainable=trainable)

    if 'h' in lyr and 'h' in snn.delay_type:
        for i, layer in enumerate(snn.proj_names[:-1]):
            w = getattr(snn, layer).weight.data.reshape(snn.num_neurons_list[i+1],
                                                        snn.num_neurons_list[i],
                                                        num_d)
            mask = get_pooling_mask(w)
            modify_weights(getattr(snn, layer), mask.reshape(
                snn.num_neurons_list[i+1], snn.num_neurons_list[i]*num_d), mode='mask',
                                trainable=trainable)
    
    if 'o' in lyr and 'o' in snn.delay_type:
        # w = torch.abs(getattr(snn, snn.h_names[-1]).weight.data).reshape(snn.num_hidden,snn.max_d, snn.num_output)
        w = getattr(snn, snn.proj_names[-1]).weight.data.reshape(
            snn.num_output, snn.num_neurons_list[-1], num_d)
        mask = get_pooling_mask(w)
        modify_weights(getattr(snn, snn.proj_names[-1]), mask.reshape(
            snn.num_output, snn.num_neurons_list[-1]*num_d), mode='mask',
                            trainable=trainable)
    

def get_w_from_proj_name(snn, proj_name):

    w = None

    num_d = len(snn.delays)

    if proj_name[:2] == 'f0':
        #if snn.delay_type=='only_hidden':  # old convention
        if 'i' not in snn.delay_type:
            num_d = 1
        w = snn.f0_f1.weight.data.reshape(
                    snn.num_neurons_list[0], snn.num_input, num_d)

    elif proj_name[-1] == 'o':
        if 'o' not in snn.delay_type:
            num_d = 1
        w = getattr(snn, snn.proj_names[-1]).weight.data.reshape(
            snn.num_output, snn.num_neurons_list[-1], num_d)

    else:
        for i, layer in enumerate(snn.proj_names[:-1]):
            if layer == proj_name:
                if 'h' not in snn.delay_type:
                    num_d = 1
                w = getattr(snn, layer).weight.data.reshape(
                    snn.num_neurons_list[i+1], snn.num_neurons_list[i], num_d)

    assert w is not None, f"[Error]: provide a valid projection name: f0_i, {snn.proj_names}"

    return w


def get_weights_and_delays(snn, layer, prun_type = 'synaptic'):

    '''
    TODO: test in nets pruned with axonal delays
    from a layer the SNN, get weights and delays as separate matrices of shape (num_pos, num_pre, k)
    being k the number of delays per synapse
    '''

    w = get_w_from_proj_name(snn, layer).cpu()
    num_pos = w.shape[0]
    num_pre = w.shape[1]

    ### ATTENTION: check if this works well for quantized and pruned!
    if prun_type == 'synaptic':
        k = int(np.ceil(len(w.nonzero()) / (num_pos*num_pre)))
    else:
        k = int(len(w.view(-1)) / (num_pos*num_pre))

    weights = torch.zeros(num_pos, num_pre, k)
    delays = torch.zeros(num_pos, num_pre, k)

    if k>1:

        for v, nz in enumerate(w.nonzero()):
            i = nz[0] # postsynaptic index
            j = nz[1] # presynaptic index
            d = nz[2] # delay value
            m = v%k # multi-delay index

            weights[i, j, m] = w[i, j, d]
            delays[i, j, m] = d

        #  the obtained delays need to be reconverted according to the max_delay and stride
        delays = snn.max_d - snn.stride*delays

    else:
        print('no delays in this network. setting all delays to zero.')
        weights = w

    return weights, delays


def save_weights_delays(snn, path = 'default', format='split', prun_type = 'synaptic'):
    
    ''''
    :param format: split or joined
    
    '''

    if path == 'default':
        # Define the path to save the layer weights and biases
        layers_path = os.path.join(CHECKPOINT_PATH, 'default_weights_delays')
    else:
        layers_path = path

    # If the directory do not exist, it is created
    if not os.path.isdir(layers_path):
        os.mkdir(layers_path)

    # Initialize weight-and-biases and the state dictionary of the network
    weights_biases = []
    snn_state_dict = snn.state_dict()

    if format == 'joined':
        # Save several arrays into a single file in uncompressed .npz format
        for k in snn_state_dict:
            np.savez(layers_path + '/' + k,
                        snn_state_dict[k].data.cpu().numpy())
            weights_biases.append(snn_state_dict[k].data.cpu().numpy())

    elif format == 'split':

        #### need to consider different snn.delay_type

        layers = snn.proj_names

        if 'i' in snn.delay_type:
            layers = ['f0_f1'] + layers    
        else:
             np.save(os.path.join(layers_path, f'f0_f1_weights'), snn.f0_f1.weight.data.cpu().numpy())

        #layers = ['f0_f1'] + layers
        for layer in layers:
            weights, delays = get_weights_and_delays(snn, layer, prun_type=prun_type)
            np.save(os.path.join(layers_path, f'{layer}_weights'), weights)
            np.save(os.path.join(layers_path, f'{layer}_delays'), delays)          

    print('Weights and delays saved in ', layers_path)


def save_state(snn, save_path, relax_time=None, skip_mems=False, skip_input=False):

    '''
    save spikes and potentials in a NH-friendly format, that is, as a single
    stream of data with dimensions time*channel_size
    TODO: save input as event-driven data    
    
    '''

    assert snn.debug,"[ERROR] Debug mode must be active to save internal activity"


    def relax(im, spikes=True):

        if spikes:
            dtype = np.uint8
        else:
            dtype = float

        relaxed_im = np.zeros((im.shape[0], im.shape[1]+relax_time*snn.batch_size), dtype=dtype)

        for i in range(snn.batch_size):
            start = i*(snn.win+relax_time)
            relaxed_im[:, start:start+snn.win] = im[:, i*snn.win:(i+1)*snn.win]

        return relaxed_im

    for layer in snn.spike_state.keys():
        # assert snn.spike_state[layer].size()[1] == 1,\
        #         "Batch size must be equal to one"

        num_neurons = snn.spike_state[layer].shape[-1]
        #spikes = snn.spike_state[layer][:, 0, :].T.detach().cpu().numpy()
        spikes = snn.spike_state[layer].type(torch.uint8).cpu().detach().numpy().T.\
            reshape(num_neurons, snn.win * snn.batch_size)
        
        if relax_time is not None:
            spikes = relax(spikes, spikes=True)

        if not(layer == 'input' and skip_input):
            np.save(os.path.join(save_path, f'{layer}_spikes'), spikes)

        if layer != 'input' and not(skip_mems):
            #mems = snn.mem_state[layer][:, 0, :].T.detach().cpu().numpy()    
            mems = snn.mem_state[layer].cpu().detach().numpy().T.\
                reshape(num_neurons, snn.win * snn.batch_size)        

            if relax_time is not None:
                mems = relax(mems, spikes=False)                
            
            np.save(os.path.join(save_path, f'{layer}_potentials'), mems)
    
    print('activity of the batch saved in ', save_path)