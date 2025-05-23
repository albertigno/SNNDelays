"""
VISUALIZATION

    Visualization class include functions to represent results after
    training or testing of the network.

    All the abstract-network classes inherit from Visualization class.

Created on 2018-2023:
    github: https://github.com/albertigno/HWAware_SNNs

    @author: Alberto
    @contributors: Laura
"""

import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from snn_delays.config import CHECKPOINT_PATH
from snn_delays.utils.hw_aware_utils import get_w_from_proj_name
from scipy.stats import gaussian_kde, norm, gamma, lognorm
import torch

def save_fig(model_dir, fig_name):
    """
    Function to save a figure.

    :param fig_name: Name of the figure to be saved.
    :param model_dir: Directory to save the file.
    """

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    plt.savefig(os.path.join(model_dir, fig_name))
    print('Figure saved in ', model_dir)


def plot_param(w, mode='histogram', title='', label = '', xlabel='', ylabel='', ax=None, colormap='RdBu', vminmax=None, transform=None, **kwargs):
    """
    Function to plot histogram or matrix representation of a parameter.

    :param w: Parameter to represent. Usually, it will be the weights or
    constant times (taus).
    :param mode: Mode for the representation. This argument can take the
    value 'histogram'|'loghist'|'histogram_nonzero'|'loghist_nonzero'  for
    histogram plot; or '2D' or '2D_square', for 2D grid plot. Default = 'histogram'.
    :param title: Title of the figure. Default = ''.
    :param xlabel: Label for the x-axis of the figure. Default = ''.
    :param ylabel: Label for the y-axis of the figure. Default = ''.
    :param ax: Axis to plot the data. Default = 'None'.
    :param colormap: cmap label to use in 2D plots.
    :param vminmax: [-vminmax,vminmax] will be the range of the cmap for
    2D plots, else determine from the data in w

    :return ax: Axis with the plot.
    """

    # Generate axis
    if ax is None:
        ax = plt.gca()

    # Get parameters to plot
    try:
        w = w.weight.data.cpu().numpy()
    except:
        w = w.data.cpu().numpy()

    if transform is not None:
        w = transform(w)

    # Get minimum and maximum parameter value
    if vminmax is None:
        v = np.max([np.abs(np.min(w)), np.abs(np.max(w))])
    else:
        v = vminmax

    # Check that mode introduced is valid
    assert (mode == 'histogram' or 
            mode == 'loghist' or 
            mode == 'histogram_nonzero' or 
            mode == 'loghist_nonzero' or 
            mode == '2D' or 
            mode == '2D_square'), \
        '[ERROR] Mode either "histogram", "loghist", "2D" or "2D_square".'

    # Plot histogram or matrix
    if mode == 'histogram' or mode == 'loghist' or mode == 'histogram_nonzero' or mode == 'loghist_nonzero':
        log = True if 'log' in mode else False
        w = w.reshape(1, -1)[0]
        if 'nonzero' in mode:
            w = w[w.nonzero()]

        #plt.hist(w, bins=200, log=log)
        n, bins, patches = ax.hist(w, bins='auto', density=True, log=log, alpha=0.5, label=label)
        hist_color = patches[0].get_facecolor()

        if 'distribution' in kwargs.keys() and len(w) >= 2:
            if kwargs['distribution'] == 'lognorm':
                # Fit a log-normal distribution
                params = lognorm.fit(w, floc=0)
                x = np.linspace(min(w), max(w), 1000)
                pdf = lognorm.pdf(x, *params)
                plt.plot(x, pdf, color=hist_color)

            elif kwargs['distribution'] == 'normal':
                # Fit a normal distribution
                mu, std = norm.fit(w)
                x = np.linspace(min(w), max(w), 1000)
                pdf = norm.pdf(x, mu, std)
                plt.plot(x, pdf, color=hist_color)

            elif kwargs['distribution'] == 'gamma':
                # Fit a gamma distribution
                params = gamma.fit(w)
                x = np.linspace(min(w), max(w), 1000)
                pdf = gamma.pdf(x, *params)
                plt.plot(x, pdf, color=hist_color)

            elif kwargs['distribution'] == 'kde':
                # Fit a Kernel Density Estimate (KDE)
                kde = gaussian_kde(w)
                x = np.linspace(min(w), max(w), 1000)
                ax.plot(x, kde(x), color=hist_color)
                #plt.plot(x, kde(x), color=hist_color)

        ax = plt.gca()
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16)
        return ax
    
    elif mode.split('_')[0] == '2D':

        # Make it a square image for nicer visualization
        if 'square' in mode:
            w = w[:min(w.shape), :min(w.shape)]

        ax.imshow(w, cmap=colormap, vmin=-v, vmax=v)
        ax.set_xlabel('input', fontsize=14)
        ax.set_ylabel('output', fontsize=14)
        ax.set_title(title, fontsize=16)
        return ax


def plot_per_epoch(data, label='', ax=None):
    """
    Function to plot the evolution of some data through the training stages,
    for example: train loss (snn.train_loss), test loss (snn.test_loss),
    accuracy (snn.acc) or number of spikes (snn.test_spk_count).

    :param data: Data to be represented.
    :param label: Label of the data for the legend. Default = ''.
    :param ax: Axis to plot the data. Default = ''.

    :return ax: Axis with the plot.
    """
    # Transform data into numpy arrays
    data = np.array(data)

    # Select axis for representation and plot data
    if ax is None:
        ax = plt.gca()  # get current axis
    ax.plot(data[:, 0], data[:, 1], label=label)
    ax.set_xlabel('Epoch')
    ax.legend()

    return ax

def training_plots(snn, savefig_dir=None):
    """
    Function to represent train loss, test loss, accuracy and spike
    counting during test using the method plot_per_neuron. It also prints
    information about maximum accuracy.

    :param snn: Neural network which metrics will be plotted.
    :param savefig_dir: Directory to save the figure. Default = None,
    nothing is saved.

    :return ax: Axis with the plot.
    """

    # Generate figures
    fig_size = (10,10)
    _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=fig_size)
    plot_per_epoch(snn.train_loss, ax=ax1, label='train_loss')
    plot_per_epoch(snn.test_loss, ax=ax1, label='test_loss')
    plot_per_epoch(snn.acc, ax=ax2, label='test acc')
    plot_per_epoch(snn.test_spk_count, ax=ax3, label='test_spk_count')

    # Save figure
    if savefig_dir is not None:
        model_dir = os.path.join(CHECKPOINT_PATH, savefig_dir, snn.model_name)   
        save_fig(model_dir, 'acc_loss_spk_per_epoch.png')

    ax = plt.gca()

    return ax


def plot_distributions(snn=None, mode='weights', savefig_dir=None):
    """
    Function to plot the weight or tau distributions of all the layers in a
    neural network.

    :param snn: The neural network to obtain the distributions. Default = None.
    :param mode: Mode for the distribution representation. This argument
    can take the value 'weights' or 'taus'; otherwise, an error will be
    printed and no figure will be generated. Default = 'weights'.
    :param savefig_dir: Directory to save the figure. Default = None,
    nothing is saved.

    :return ax: Axis with the plot.
    """

    # Select parameters and parameter names
    if mode == 'weights':
        params_names = snn.base_params_names
        params = snn.base_params
    elif mode == 'taus':
        params_names = snn.tau_params_names
        params = snn.tau_params

    # Check that mode introduced is valid.
    assert mode == 'weights' or mode == 'taus', \
        '[ERROR] Mode either "weight" or "taus".'

    c = len(params_names)
    fig = plt.figure(figsize=(7, 7))
    for i, name in enumerate(params_names):
        plt.subplot(c, 1, i + 1)
        plot_param(params[i], title=name)
    plt.tight_layout()

    # Save figure
    if savefig_dir is not None:
        model_dir = os.path.join(CHECKPOINT_PATH, savefig_dir, snn.model_name)   
        save_fig(model_dir, f'{mode}_distributions.png')

    ax = plt.gca()

    return ax


def plot_per_neuron(w, n_cols=3, num_channels=1):
    """
    Function to represent a randomly sample weight per neuron

    :param w: Weights to represent.
    :param n_cols: Number of columns/rows for representation. If the argument
    take a value 'n', a total of n^2 samples will be represented. Default = 3.
    :param num_channels: Number of channels (2 for N-MNIST). Default = 1.
    """

    # Select weights
    try:
        w = w.weight.data.cpu().numpy()
    except:
        w = w.data.cpu().numpy()

    # Select samples randomly
    sample_idx = np.random.choice(np.arange(w.shape[0]), n_cols ** 2)
    sample = w[sample_idx, :]

    # Reshape samples
    s = int(w.shape[1] / num_channels)
    a, b = square(s)

    # Generate the plot
    for i, x in enumerate(sample):
        plt.subplot(n_cols, n_cols, i + 1)
        plt.imshow(x[:s].reshape(a, b), cmap='RdBu')


def square(num):
    """
    Function to get the two closest factors of the input integer 'num'.
    These two numbers are used to plot a vector of length 'num' as a
    square-ish matrix in other plot methods (plot_per_neuron)

    :param num: Number to get the closest factors

    :return: The two closest factors for the input integer.
    """

    factor1 = [x for x in range(1, num + 1) if num % x == 0]
    factor2 = [int(num / x) for x in factor1]
    idx = np.argmin(np.abs(np.array(factor2) - np.array(factor1)))
    return factor1[idx], factor2[idx]


# TODO: No hay 'animacion', solo una imagen. Lo estoy probando como:
#  animation(snn.spike_state, snn.layer_names[0], sample=0,
#            make_square=True, cmap='RdBu')
#  ¿Puede ser que en Jupyter no se visualice bien la animacion?
def animation(spike_state, layer_name, sample=0,
              make_square=True, cmap='RdBu'):
    """
    Function to animate a spike_state matrix of one neuron in the network.

    :param spike_state:
    :param layer_name: Layer to be animated.
    :param sample: Sample to be animated. Default = 0.
    :param make_square: Boolean to apply (True) or not (False) reshape to the
    data and obtain a squared representation. Default = True.
    :param cmap: Color map. Default = 'RdBu'.

    :return: Animation.
    """

    # Define animate function
    def animate(frame_num):
        """
        Function to animate

        :param frame_num: Number of frames of the animation.
        """
        im.set_data(x[frame_num, :, :])
        im.axes.set_title(str(frame_num))

    # Get spike state of the layer and optional reshape
    x = spike_state[layer_name][:, sample, :].detach().cpu().numpy()

    if make_square:
        a, b = square(x.shape[1])
        x = x.reshape(x.shape[0], b, a)

    # Set figure configuration
    fig = plt.figure(figsize=(9, 9))
    v_max = np.max(x)
    v_min = np.min(x)
    im = plt.imshow(x[0, :, :], cmap=cmap, vmax=v_max, vmin=v_min)

    # Generate the animation
    interval = 2000 / len(x)  # 2 seconds
    anim = FuncAnimation(fig, animate, frames=len(x), interval=interval)

    return anim

def plot_delays(snn, proj_name, mode='histogram', x_axis='neuron',
                max_subplots=None, savefig_dir=None, show_cmap=False):
    """
    Function to plot the weights of a projection between two layers,
    delay-by-delay.

    :param snn: Neural network whose delays will be plotted.
    :param proj_name:
    :param mode: Mode for the representation. This argument can take the
    value 'histogram' or 'loghist' for histogram plots; or '2D' or '2D_square',
    for 2D heatmap plots. Default = 'histogram'.
    :param x_axis: It can take the value 'neuron', 'delays_pre' or 'delays_pos'
    :param max_subplots:
    :param savefig_dir: Name of the folder where the graph will be saved. If
    this argument take the value 'None', the graph won't be saved. Default =
    None.

    :return: A graph with the delays plotted.
    """

    assert (x_axis in ['neuron', 'delays_pre', 'delays_pos']),\
        '[ERROR] x_axis can only be "neuron"|"delays_pre"|"delays_pos".'

    assert (isinstance(max_subplots, list) or isinstance(max_subplots, int) or max_subplots is None),\
        '[ERROR] max_subplots must be either int, list (of ints), or None.'

    # Flip the delays (this is needed for interpretability)
    real_delays = snn.max_d - 1 - snn.delays
    real_delays = real_delays.numpy()

    # Get weights from projections. Weight mat shape [post neu, pre neu, delay lvl]
    w = get_w_from_proj_name(snn, proj_name)
    print (f"Projection {proj_name}, weight matrix shape {w.shape}")


    # Set the maximum subplots and the list of subplots
    if isinstance(max_subplots, list):     # we re given an explicit list of delays/pre/post graphs to plot
        which_plots = max_subplots
        max_subplots = len(which_plots)
    else:
        if max_subplots is None:           # we re going to plot all graphs for delays/pre/post
            if x_axis == 'neuron':
                max_subplots = len(snn.delays)
            elif x_axis == 'delays_pre':
                max_subplots = w.shape[1]
            elif x_axis == 'delays_pos':
                max_subplots = w.shape[0]
            else:
                raise ValueError("x_axis can only be 'neuron'|'delays_pre'|'delays_pos'") 
       
        if x_axis == 'neuron':     # if we re plotting delay graphs we need the list of strided delays indexing
            which_plots = real_delays[:max_subplots]
        else:                      # .. otherwise the pre/post match exactly the weight mat indexing
            which_plots = list(range(max_subplots)) 
    
    #print('which_plots:', which_plots)


    # Convert the num of plots to a 2d grid of indices for the subplots
    # (reshape in base of the maximum of subplots)
    if '2D' in mode and show_cmap is True:
        a, b = square(max_subplots+1) # +1 for the colormap
    else:
        a, b = square(max_subplots)
    print(f'grid: {a}x{b}')


    # 2D plots: uniform color map and ranges among plots 
    maxabs = np.max([ torch.abs(torch.max( w )).cpu().numpy(), torch.abs(torch.min( w )).cpu().numpy() ]) 
    colormap = 'RdBu'                

    # Plot the data
    if x_axis == 'neuron':
        #fig = plt.figure(figsize=(20, 20))
        for x in range(max_subplots):
            plt.subplot(a, b, x + 1)

            delay_lvl = which_plots[x]                                   # graph idx to delay idx
            w_idx = np.argwhere( real_delays == delay_lvl).flatten()[0]  # delay idx to weight idx
            plot_param(w[:, :, w_idx], mode=mode, title=f'd={delay_lvl}', colormap=colormap, vminmax=maxabs)

            if 'hist' in mode:
                plt.xlabel('Weights')
                plt.ylabel('Frequency')
            else:
                plt.xlabel('Pre neu')
                plt.ylabel('Pos neu')

    elif x_axis == 'delays_pre':
        #fig = plt.figure(figsize=(20, 20))

        for x in range(max_subplots):
            plt.subplot(a, b, x + 1)
            d = torch.flip(w, dims=[2])

            w_idx = which_plots[x]               # graph idx to weight idx
            plot_param(d[:, w_idx, :], mode=mode, title=f'pre_neuron={w_idx}', colormap=colormap, vminmax=maxabs)

            # plt.xticks(snn.delays)
            if 'hist' in mode:
                plt.xlabel('Fanout Weights')
                plt.ylabel('Frequency')
            else:
                plt.xlabel('Delay lvl')
                plt.ylabel('Pos neu')

    elif x_axis == 'delays_pos':
        #fig = plt.figure(figsize=(20, 20))

        for x in range(max_subplots):
            plt.subplot(a, b, x + 1)
            d = torch.flip(w, dims=[2])

            w_idx = which_plots[x]              # graph idx to weight idx
            plot_param(d[w_idx, :, :], mode=mode, title=f'pos_neuron={w_idx}', colormap=colormap, vminmax=maxabs)

            # plt.xticks(snn.delays)
            if 'hist' in mode:
                plt.xlabel('Fanin Weights')
                plt.ylabel('Frequency')
            else:
                plt.xlabel('Delay lvl')
                plt.ylabel('Pre neu')

    if '2D' in mode and show_cmap is True:
        gradient = np.linspace(-maxabs, maxabs, 256).reshape(-1,1)
        gradient = np.flip(gradient)
        gradient = np.hstack([gradient,]*25)
        plt.subplot(a,b,max_subplots+1)
        plt.imshow(gradient, vmin=-maxabs, vmax=maxabs, extent=[0,maxabs/4,-maxabs, maxabs], cmap=colormap)
        plt.tick_params( axis='x', which='both', bottom=False, top=False, labelbottom=False)

    plt.tight_layout()

    # Save figure
    if savefig_dir is not None:
        model_dir = os.path.join(CHECKPOINT_PATH, savefig_dir, snn.model_name)
        save_fig(model_dir, f'{proj_name}_{mode}_delays.png')

    return plt.gca()


# TODO: Cuando se usa mode = 'synapse', las gráficas salen vacías.
#  + La linea donde se usa el num_channels esta comentada, ¿se puede quitar?
#  + Documentar.
def plot_per_neuron_delays(snn, proj_name, mode='image', num_channels=1):
    """
    Function to plot information about randomly sample weight per N neuron
    taking into account the delays.

    :param snn: The neural network whose information will be plotted.
    :param proj_name: The name of the projection whose delays will be plotted.
    :param mode: Mode of the representation. It can take the values 'image'
    (...), 'avg' () or 'synapse' (). Default = 'image'.
    :param num_channels:Number of channels (2 for N-MNIST). Default = 1.

    Alberto:
    randomly sample weight per Neuron
    if mode == d, plot all delays in order,else pick random
    w.shape = (output_neurons, input_neurons, delays)
    """

    # Get the weights and the maximum absolute weight
    w = get_w_from_proj_name(snn, proj_name)
    w = w.cpu().numpy()

    v = np.max(np.abs(w))

    # Get the number of neurons and the number of delays
    n_neurons = w.shape[0]
    num_d = len(snn.delays)
    # s = int(w.shape[1] / num_channels)

    # Reshape
    c, d = square(w.shape[1])
    subplot_a, subplot_b = square(n_neurons)

    # Flip the delays (needed for interpretability)
    real_delays = snn.max_d - 1 - snn.delays

    # Plot
    plt.figure(figsize=(20, 20))

    if mode == 'image':

        ## plot per post neu per delay level the incoming weights heatmap

        # Set the number of rows to max 10
        n_rows = min(10, len(snn.delays))

        # Redefine the number of neurons
        if n_neurons >= 10:
            n_neurons = 10

        for i in range(n_neurons):
            for r in range(num_d):
                if r < n_rows:
                    xx = w[i, :, :]
                    plt.subplot(n_rows, n_neurons, i + r * n_neurons + 1)
                    plt.imshow(xx[:, r].reshape(c, d), cmap='RdBu')
                    plt.title(f'Neuron = {i}, d = {real_delays[r]}')
                    # snn.sample[i].append(xx[:, r].reshape(c, d))

    elif mode == 'avg':

        ## plot per post neu for all pre neu the avg pos/neg weights across all delay levels 

        for i in range(n_neurons):
            xx = w[i, :, :]
            pos = xx * (xx > 0.0)
            neg = xx * (xx < 0.0)

            plt.subplot(subplot_a, subplot_b, i + 1)
            plt.plot(np.mean(np.abs(pos), axis=1),
                     color='b', label='Positive')
            plt.plot(np.mean(np.abs(neg), axis=1),
                     color='r', label='Negative')
            plt.title(f'Neuron = {i}')
            plt.ylim(0, v)
            # plt.ylim(0, v/3)

        plt.legend()

    elif mode == 'synapse':

        ## plot per post neu for a rand selected pre neu the weights across all delay levels

        for i in range(n_neurons):
            xx = w[i, :, :]
            pre = np.random.randint(w.shape[1])

            plt.subplot(subplot_a, subplot_b, i + 1)
            plt.plot(real_delays, xx[pre, :])
            plt.title(f'Pre = {pre}, Post = {i}')
            plt.ylim(-v, v)
            # plt.ylim(-v/2, v/2)

    plt.tight_layout

    return plt.gca()


def visualize_activity(snn, layer, neuron=None, sample=None):
    """
    Function to visualize a single neuron's activity in the snn's layer
    of choice during the time that a sample of the batch is being processed.
    The membrane potential, the spikes and the reference threshold voltage
    are represented.

    :param snn: The neural network whose neuron activity will be plotted.
    :param layer: The layer of the network whose neuron activity will be
    plotted.
    :param neuron: Neuron of the layer whose activity is plotted.
    :param sample: Sample of the batch through which the activity is plotted.
    """

    # Check debug mode
    assert snn.debug == True, \
        "[ERROR]: Debug mode and at least one batch propagation are needed."
    
    th = 0.3

    # Take the number of neurons in the layer
    num_neurons = snn.spike_state[layer].shape[-1]

    # Pick a random neuron or sample the batch if some of them are not defined
    if sample is None:
        sample = np.random.randint(snn.batch_size)
    if neuron is None:
        neuron = np.random.randint(num_neurons)

    # Get spike state and threshold
    spk = snn.spike_state[layer][:,sample,neuron]
    evt_spk = spk.cpu().detach().numpy().nonzero()
    th_spk = th * np.ones(len(evt_spk[0]))

    # Get membrane potential
    if layer != 'input':
        mem = snn.mem_state[layer][:,sample,neuron]
        mem = mem.cpu().detach().numpy()
    else:
        mem = np.zeros(snn.win,)

    # Plot

    

    plt.hlines(th, 0, snn.win, 'g', 'dashed', label='threshold')
    plt.plot(mem, label='membrane potential')
    # plt.eventplot(spk.cpu().detach().numpy().nonzero())
    plt.scatter(evt_spk, th_spk, c='k', marker=10, label = 'spikes')
    plt.title('Dynamics of neuron = ' + str(neuron) +
              ' of the layer ' + str(layer))
    plt.ylabel('Membrane potential (mV)')
    plt.xlabel('Time (ms)')
    plt.legend()

    return plt.gca()


# TODO: mode = 'traces' no funciona, no existe el atributo snn.postsyn_traces
def plot_raster(snn, layer, n, mode='spikes', colorbar = False):
    """
    Function to plot the raster plot of a layer.

    :param snn: Neural network whose raster plot will be obtained.
    :param layer: Layer of the network whose raster plot will be obtained.
    :param n: Number of samples from batch to be drawn in the raster plot.
    :param mode: Mode of the representation. It can be take the values
    'spikes' (for spiking raster plot), 'mems' (for membrane potential),  or
    'traces' (for traces raster plot). Default = 'spikes'.

    :return: The graph with the representation selected.
    """

    if type(n) == int:
        index = list(range(n))
    elif type(n) == list:
        assert type(n[0]) == int, "[ERROR] make sure n is an int or list of int" 
        index = n

    print(index)
    
    N = len(index)

    # Check debug mode
    assert snn.debug == True, \
        "[ERROR]: Debug mode and at least one batch propagation are needed."

    # Take the number of neurons in the layer
    if mode == 'spikes' or mode == 'traces':
        num_neurons = snn.spike_state[layer].shape[-1]
    elif mode == 'mems':
        num_neurons = snn.mem_state[layer].shape[-1]

    # Plot the raster plot
    if mode == 'spikes':
        spk = snn.spike_state[layer][:, index, :]
        spk = spk.cpu().detach().numpy().T.reshape(num_neurons, snn.total_time * N)
        # plt.figure(figsize=(5, n))
        plt.imshow(spk, cmap='Greys')

    elif mode == 'mems':
        mem = snn.mem_state[layer][:, index, :]
        mem = mem.cpu().detach().numpy().T.reshape(num_neurons, snn.total_time * N)
        # v = 2*snn.thresh
        # plt.imshow(mem, vmin= -v, vmax=v, cmap = 'RdBu')
        plt.imshow(mem, cmap='RdBu')
        if colorbar:
            plt.colorbar(location='right')

    elif mode == 'traces':
        tr = snn.postsyn_traces[layer][:, index, :]
        tr = tr.cpu().detach().numpy().T.reshape(num_neurons, snn.total_time * N)
        # v = 2*snn.thresh
        # plt.imshow(mem, vmin= -v, vmax=v, cmap = 'RdBu')
        plt.imshow(tr, cmap='RdBu')

    plt.ylabel('Neuron')
    plt.xlabel('Time (ms)')

    return plt.gca()


def frame_2_image(snn, sample='random'):
    '''
    this generates something like time-surfaces
    valid only for image datasets
    '''
    
    if sample == 'random':
        sample = np.random.randint(0,snn.batch_size)

    dim = int(np.sqrt(snn.num_input/2))
    frames = snn.spike_state['input'][:,sample,:][:, :snn.num_input//2].view(snn.win,dim, dim )

    tau = snn.win
    composed_image = np.zeros((dim, dim))
    for rev_t, frame in enumerate(frames):
        t = snn.win-rev_t
        composed_image += np.exp(-t/tau)*frame.detach().cpu().numpy()
    return composed_image


def plot_taus(snn, label = 'taus', mode='discrete'):

    '''
    mode: real or discrete
    '''

    delta_t = snn.dataset_dict.get('time_ms', 0)/snn.win

    num_subplots = len(snn.tau_m_h)

    plt.title(f'Distribution of taus, {mode} time')
    for i, pseudo_tau_m in enumerate(snn.tau_m_h):

        real_tau = -delta_t/torch.log(torch.sigmoid(pseudo_tau_m))

        if mode=='real':
            plt.subplot(num_subplots, 1, i+1)
            plot_param(real_tau, mode='histogram', label=label, distribution='kde')
            if i==num_subplots-1:
                plt.xlabel('time (ms)')

        elif mode=='discrete':
            plt.subplot(num_subplots, 1, i+1)
            #plot_param(real_tau/snn.win, mode='histogram')    
            plot_param(real_tau/delta_t, mode='histogram', label=label, distribution='kde')
            plt.axvline(x=snn.win, color='red', linestyle='--', linewidth=2)
    #        plt.xlim(0, snn.win)
            if i==num_subplots-1:
                plt.xlabel('simulation timestep')

        else:
            raise ValueError(f"Unsupported: {mode}. Choose from 'real', 'discrete'.")

    return plt.gca()


def plot_taus_refact(snn, label = 'taus', mode='discrete'):

    '''
    mode: real or discrete
    '''

    delta_t = snn.dataset_dict.get('time_ms', 0)/snn.win

    tau_m_params = [param for name, param in snn.named_parameters() if 'tau' in name]

    num_subplots = len(tau_m_params)

    max_tau = 0.0

    plt.title(f'Distribution of taus, {mode} time')
    for i, pseudo_tau_m in enumerate(tau_m_params):

        real_tau = -delta_t/torch.log(torch.sigmoid(pseudo_tau_m))

        if torch.max(real_tau).item()>max_tau:
            max_tau = torch.max(real_tau).item()

        if mode=='real':
            plt.subplot(num_subplots, 1, i+1)
            plot_param(real_tau, mode='histogram', label=label, distribution='kde')
            if i==num_subplots-1:
                plt.xlabel('time (ms)')

        elif mode=='discrete':
            plt.subplot(num_subplots, 1, i+1)
            #plot_param(real_tau/snn.win, mode='histogram')    
            ax = plot_param(real_tau/delta_t, mode='histogram', label=label, distribution='kde')
            ax.axvline(x=snn.win, color='red', linestyle='--', linewidth=2)
            ax.set_xlim(0, max_tau/delta_t)
            if i==num_subplots-1:
                plt.xlabel('simulation timestep')

        else:
            raise ValueError(f"Unsupported: {mode}. Choose from 'real', 'discrete'.")



# def plot_taus_refact(snn, label='taus', mode='discrete'):
#     '''
#     mode: real or discrete
#     '''
#     delta_t = snn.dataset_dict.get('time_ms', 0)/snn.win
#     tau_m_params = [param for name, param in snn.named_parameters() if 'tau' in name]
#     num_subplots = len(tau_m_params)

#     plt.figure(figsize=(8, 2*num_subplots))  # Ensure proper figure size
#     plt.title(f'Distribution of taus, {mode} time')
    
#     for i, pseudo_tau_m in enumerate(tau_m_params):
#         real_tau = -delta_t/torch.log(torch.sigmoid(pseudo_tau_m))
        
#         plt.subplot(num_subplots, 1, i+1)
        
#         if mode == 'real':
#             plot_param(real_tau, mode='histogram', label=label, distribution='kde')
#             if i == num_subplots-1:
#                 plt.xlabel('time (ms)')
#                 plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))  # Control ticks
                
#         elif mode == 'discrete':
#             plot_param(real_tau/delta_t, mode='histogram', label=label, distribution='kde')
#             plt.axvline(x=snn.win, color='red', linestyle='--', linewidth=2)
#             plt.xlim(0, 10*snn.win)
#             if i == num_subplots-1:
#                 plt.xlabel('simulation timestep')
#                 plt.gca().xaxis.set_major_locator(plt.MultipleLocator(snn.win))  # Ticks at window multiples
                
#         else:
#             raise ValueError(f"Unsupported: {mode}. Choose from 'real', 'discrete'.")
    
#     plt.tight_layout()  # Fix overlapping elements


def plot_add_task(output, reference, axes=None, name=''):
    ref = reference
    out = output

    if out.ndim < 3:  # Ensure output has at least 3 dimensions
        raise ValueError("Expected output to have at least 3 dimensions")

    diff = ref - out[:, :, 0]

    print(np.mean(diff))

    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(5, 10))  # Create a new figure if axes not provided

    axes[0].imshow(ref, vmin=0, vmax=2)
    axes[0].set_title('Reference '+name)
    axes[0].set_ylabel('Time')

    axes[1].imshow(out[:, :, 0], vmin=0, vmax=2)
    axes[1].set_title('Output')
    axes[1].set_ylabel('Time')

    axes[2].imshow(diff, vmin=-1, vmax=1)
    axes[2].set_title('Difference')
    axes[2].set_ylabel('Time')
    axes[2].set_xlabel('Training Sample')

    return axes  # Return the axes to be used in an external figure

def plot_add_task2(output, labels):
    mean_out = np.mean(output, axis=0)
    real_labels = labels[:, 0, 0]

    plt.plot(real_labels, label='Real Labels')
    plt.plot(mean_out, label='Mean Out')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()

    return plt.gca()

def plot_add_task3(snn):
    plt.imshow(snn.mem_state['output'].detach().cpu().numpy(), vmin=0, vmax=2)
    plt.title('output')
    plt.ylabel('time')
    plt.xlabel('training sample')

    return plt.gca()

def plot_add_task4(snn, N):
    #N = np.random.randint(snn.batch_size)
    reference = torch.sum(snn.spike_state['input'][:,N,1]* snn.spike_state['input'][:,N,0]).item()
    visualize_activity(snn, 'output', sample=N)
    plt.plot(snn.spike_state['input'][:,N,1].cpu().numpy())
    plt.axhline(y=reference, color='r', linestyle='--')
    plt.axvspan(0.9*snn.win,snn.win, color='gray', alpha=0.5)

    return plt.gca()

from snn_delays.utils.train_utils import to_plot

def plot_membrane_evolution(snn, axes=None):

    num_plots = len(snn.mem_state.keys())

    if axes is None:
        fig, axes = plt.subplots(num_plots, 1, figsize=(5, 10))  # Create a new figure if axes not provided    

    for i, layer in enumerate(sorted(snn.mem_state.keys())):
        axes[i].plot(to_plot(snn.mem_state[layer][:, 0, :]))
        axes[i].set_title(layer)
        if i == num_plots-1:
            axes[i].set_xlabel("Timestep")

    return axes


def plot_losses(nested_loss_lists, label='Mean loss', color='blue', linestyle='-'):

    '''
    plot and compare losses for ablation study
    '''

    # Example data: replace `nested_loss_lists` with your actual data
    #nested_loss_lists = tstloss_d['f_d_2l_hm_ft']

    # Ensure all lists have the same length and epoch indices
    epochs = [entry[0] for entry in nested_loss_lists[0]]  # Epochs
    all_losses = [np.array([entry[1] for entry in lst]) for lst in nested_loss_lists]

    # Calculate average and standard deviation
    mean_losses = np.mean(all_losses, axis=0)
    std_losses = np.std(all_losses, axis=0)

    # Plot the average loss curve with error bars
    #plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_losses, label=label, color=color, linestyle=linestyle)
    #plt.fill_between(epochs, mean_losses - std_losses, mean_losses + std_losses, color=color, alpha=0.2, label='±1 Std Dev')
    plt.fill_between(epochs, mean_losses - std_losses, mean_losses + std_losses, color=color, alpha=0.2)
    #plt.title("Average Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    return plt.gca()