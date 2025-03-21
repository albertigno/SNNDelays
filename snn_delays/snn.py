from snn_delays.config import CHECKPOINT_PATH, DATASET_PATH
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import sys
import json
import numpy as np


class ActFunBase(torch.autograd.Function):
    """
    Base activation function class

    The class implement the forward pass using a Heaviside function as
    activation function. This is the usually choose for spiking neural
    networks. The backward pass is only initialized, this method will be
    rewritten with the surrogate gradient function in the child classes.
    """

    @staticmethod
    def forward(ctx, input_data, scale_factor):
        """
        Forward pass

        Take as input the tensor input_data (in general, the membrane
        potential - threshold) and return a tensor with the same dimension
        as input_data whose elements are 1.0 if the corresponding element in
        input_data is greater than 0.0, and 0.0 otherwise.

        The input parameter ctx is a context object that can be used to stash
        information for backward computation.
        """
        ctx.save_for_backward(input_data, scale_factor)
        return input_data.gt(0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass (this method will be rewritten with the surrogate
        gradient function)
        """
        pass


class ActFunFastSigmoid(ActFunBase):
    """
    Fast-sigmoid activation function class

    It inherits methods from the ActFunBase class and rewrite the backward
    method to include a surrogate gradient function.

    Surrogate gradient function: Normalized negative part of a fast sigmoid
    function (Reference: Zenke & Ganguli (2018))
    """
    def __init__(self):
        """
        Initialization of the activation function
        """
        super(ActFunBase, self).__init__()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass

        Surrogate gradient function: Normalized negative part of a fast
        sigmoid function

        The parameter 'scale' controls steepness of surrogate gradient.
        """
        # scale = 10.0

        input_data, scale = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad = grad_input / (scale*torch.abs(input_data) + 1.0) ** 2
        return grad, None

class Training:
    """
    Training SNN class

    This class includes the methods used to train and evaluate the SNNs, focusing on BPTP and T-BPTP
    """

    def propagate(self, images, labels):
        """
        Function to make the propagation of a single batch. It will depend on
        the loss function used
        :param images: Samples
        :param labels: Targets of the samples
        :param threshold: Apply a threshold to convert the original samples into spikes
        """

        # Resize and reformat of images and labels
        if self.input2spike_th is not None:
            images = images > self.input2spike_th

        # handle incomplete last batch for reproducible tests
        labels = labels.to(self.device)
        if len(labels)<self.batch_size:
            padding_lb = torch.zeros((self.batch_size - len(labels),) + labels.shape[1:]).to(self.device)
            labels = torch.cat([labels, padding_lb], dim=0)

        images = images.to(self.device)
        if len(images)<self.batch_size:
            padding_im = torch.zeros((self.batch_size - len(images),) + images.shape[1:]).to(self.device)
            images = torch.cat([images, padding_im], dim=0)

        ### zero-padding the inputs along the temporal dimension
        if self.time_win<self.win:
            zero_t = torch.zeros(self.batch_size, self.win-self.time_win, images.size(2)*images.size(3), dtype=images.dtype, device=self.device)

            if self.use_amp:
                images = torch.cat([images.view(self.batch_size, self.time_win, -1), zero_t], dim=1).half().to(self.device)
            else:
                images = torch.cat([images.view(self.batch_size, self.time_win, -1), zero_t], dim=1).float().to(self.device)

        elif self.time_win == self.win:
            if self.use_amp:
                images = images.view(self.batch_size, self.win, -1).half().to(self.device)
            else:
                images = images.view(self.batch_size, self.win, -1).float().to(self.device)          

        else:
            raise Exception("propagation time below data timesteps not implemented yet!")

        # Squeeze to eliminate dimensions of size 1    
        if len(images.shape)>3:    
            images = images.squeeze()

        labels = labels.float().squeeze().to(self.device)

        l_f = self.loss_fn

        all_o_mems, all_o_spikes = self(images)

        if l_f == 'mem_last':
            _, labels = torch.max(labels.data, 1)
            outputs = F.softmax(all_o_mems[-1], dim=1)

        elif l_f == 'mem_sum':
            outputs = torch.zeros(
            self.batch_size, self.num_output, device=self.device)
            _, labels = torch.max(labels.data, 1)
            for o_mem in all_o_mems:
                outputs = outputs + F.softmax(o_mem, dim=1)

        elif l_f == 'mem_mot':
            # as in the zenke tutorial
            _, labels = torch.max(labels.data, 1)
            m,_=torch.max(torch.stack(all_o_mems, dim=1), 1)
            outputs = F.softmax(m, dim=1)

        elif l_f == 'spk_count':
            # outputs = outputs/self.win   #normalized         
            outputs = torch.sum(torch.stack(all_o_spikes, dim=1), dim = 1)/self.win

        elif l_f == 'mem_prediction':
            perc = 0.9
            start_time = int(perc * self.win)
            a_o_m = all_o_mems[start_time:]
            outputs = torch.mean(torch.stack(a_o_m, dim=1), dim = 1)
            #outputs, _ = torch.max(torch.stack(a_o_m, dim=1), dim = 1)
            
            #outputs = torch.stack(a_o_m, dim=1).squeeze()

        return outputs, labels


    # TODO: Documentacion
    def train_step(self, train_loader=None, optimizer=None, scheduler= None, 
                   spk_reg=0.0, l1_reg=0.0, dropout=0.0, verbose=True):
        """
        Function for the training of one epoch (over the whole dataset)

        :param train_loader: A pytorch dataloader (default = None)
        :param optimizer: A pytorch optimizer. It can take the values: ...
        (default = None)
        :param spk_reg: Penalty for spiking activity (default = 0.0)
        :param l1_reg: l1 regularizer ¿? (default = 0.0)
        :param dropout: Percentage of randomly dropped spikes (applied to the
        input) (default = 0.0)
        :param verbose: ¿? (default = True)
        """

        # Initializing simulation values to track
        total_loss_train = 0
        running_loss = 0
        total = 0

        # Setting simulation parameters
        num_iter = self.num_train_samples // self.batch_size
        #sr = spk_reg / self.win

        # Training loop over the train dataset
        for i, (images, labels) in enumerate(train_loader):

            # Reset gradients
            self.zero_grad()
            optimizer.zero_grad()

            # # Dropout [REVIEW]
            # images = self.dropout(images.float())

            with amp.autocast(enabled=self.use_amp):

                # Propagate data
                outputs, reference = self.propagate(images, labels)

                #total spike count (for spike regularization)
                spk_count = self.h_sum_spike / (self.batch_size * sum(self.num_neurons_list) * self.win)
                
                #  For L1 loss
                l1_score = 0
                if l1_reg != 0.0:
                    p = self.base_params[1:] if self.delay_type=='ho' else self.base_params
                    for weights in p:
                        weights_sum = torch.sum(torch.abs(weights)) / (weights.shape[0] * weights.shape[1])
                        l1_score = l1_score + weights_sum
                
                #Update simulation values to track
                loss = self.criterion(outputs[:labels.size(0)], reference[:labels.size(0)]) + \
                    spk_reg * spk_count + l1_reg*l1_score

                running_loss += loss.detach().item()
                total_loss_train += loss.detach().item()
                total += labels.size(0)

            # Calculate gradients and optimize to update weights
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            #scheduler.step()
            scheduler.step(self.epoch + i / num_iter)

            # Apply verbose
            # TODO: try with episodic learning and improve syntax
            if verbose:
                if num_iter >= 3:
                    if (i + 1) % int(num_iter / 3.0) == 0:
                        print('Step [%d/%d], Loss: %.5f'
                              % (
                                  i + 1,
                                  self.num_train_samples // self.batch_size,
                                  running_loss), flush=True)
                        print(f'l1_score: {l1_score}')
                else:
                    if (i + 1) % int(num_iter) == 0:
                        print('Step [%d/%d], Loss: %.5f'
                              % (
                                  i + 1,
                                  self.num_train_samples // self.batch_size,
                                  running_loss), flush=True)

            # Reset running loss
            running_loss = 0

        # Update parameters
        self.epoch = self.epoch + 1
        self.train_loss.append([self.epoch, total_loss_train / num_iter])


    def test(self, test_loader=None, dropout=0.0, only_one_batch=False):
        """
        Function to run a test of the neural network over all the samples in
        test dataset


        :param test_loader: Test dataset (default = None)
        :param dropout: Percentage of randomly dropped spikes (applied to the
        input) (default = 0.0)
        """

        # Initializing simulation values to track
        correct = 0
        total = 0
        total_loss_test = 0
        total_spk_count = 0
        total_spk_count_per_layer = np.zeros(self.num_layers)

        # Initialization to store predictions and references
        all_preds = list()
        all_refs = list()        

        # Testing loop over the test dataset
        for i, (images, labels) in enumerate(test_loader):
            
            # # Dropout
            #images = self.dropout(images.float())

            # # Propagate data
            with torch.no_grad():
                with amp.autocast(enabled=self.use_amp):
                    self.eval()
                    outputs, reference = self.propagate(images, labels)
                    self.train()

            # crop results to the labels size (for incomplete batch)
            if type(outputs) == list:
                outputs = [output[:labels.size(0)] for output in outputs]
            else: 
                outputs = outputs[:labels.size(0)]

            reference = reference[:labels.size(0)]        

            # total spike count
            spk_count = self.h_sum_spike / sum(self.num_neurons_list)

            # per layer spike count
            spk_count_layer = torch.tensor([sum_spk/num_neurons for (sum_spk, num_neurons)
             in zip(self.h_sum_spikes_per_layer, self.num_neurons_list)])
            
            #spk_count_layer = torch.tensor(0) # CONV!!

            total_spk_count_per_layer += spk_count_layer.cpu().numpy()

            # Update simulation values to track
            loss = self.criterion(outputs, reference)

            if type(outputs) == list:
                _, predicted = torch.max(outputs[0].data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            _, reference = torch.max(labels.data, 1)

            all_preds = all_preds + list(predicted.cpu().numpy())
            all_refs = all_refs + list(reference.cpu().numpy())

            total += labels.size(0) 

            correct += float((predicted == reference.to(self.device)).sum())

            total_loss_test += loss.detach().item()
            total_spk_count += spk_count.detach()

            if only_one_batch:
                break

        # Calculate accuracy
        acc = 100. * float(correct) / float(total)

        # update accuracy history
        if self.acc[-1][0] <= self.epoch:
            self.acc.append([self.epoch, acc])
            self.test_loss.append([self.epoch, total_loss_test / (i+1)])
            self.test_spk_count.append([self.epoch, total_spk_count.detach().item() / total])
            # quitar penultimo acc, test_loss si coinciden las epocas o si se entrena por primera vez  
            if self.acc[-2][0] == self.epoch or self.acc[-2][1] == None:
                self.acc.pop(-2)
                self.test_loss.pop(-2)
                self.test_spk_count.pop(-2)

        spk_count_layer_neuron = list((total_spk_count_per_layer/ total))

        # Print information about test
        print('Test Loss: {}'.format(total_loss_test / (i+1)))
        print('Avg spk_count per neuron for all {} time-steps {}'.format(
            self.win, total_spk_count / total))
        print('Avg spk per neuron per layer {}'.format(spk_count_layer_neuron))
        print('Test Accuracy of the model on the test samples: %.3f' % acc)
        print('', flush=True)

        return all_refs, all_preds

class SNN(Training, nn.Module):
    """
    Spiking neural network (SNN) class.

    Common characteristic and methods for a spiking neural network with or
    without delays. It inherits from nn.Module.
    """

    def __init__(self, dataset_dict, structure=(256, 2),
                 connection_type='r', delay=None, delay_type='ho',
                 reset_to_zero=True, tau_m='normal', win=50,
                 loss_fn='mem_sum',
                 batch_size=256, device='cuda', debug=False):
        """
        Implementation of an SNN with flexible structure and that can include
        delays of different types.

        :param dataset_dict: Dictionary containing basic info of the dataset.

        :param structure: There are two ways to specify the network structure:
            (1) fixed: tuple -> (neurons_per_hidden_layer, number_of_layers)
            (2) flexible: list -> [n1, n2, n3, ...] with each n being the
            number of neurons per layers 1, 2, 3, ...
        Default = (256, 2).

        :param connection_type: The type of deepnet hidden layer, it can take
        the values 'f' (feedforward layer) or 'r' (recurrent layer) and define
        the update function to be used. Default = 'r'.

        :param delay: A tuple to specify the delays for the network. The tuple
        has the structure (depth, stride). Default = None, equivalent to (1,1).

        :param delay_type: Where delays are applied. It can accept the
        values: 'i' for input delays, 'h' for hidden delays and 'o' for
        output delays, or a combination of them. Default = 'ho', meaning that
        all the layer except the input layer have delays.

        :param thresh: Neuron's threshold. Default = 0.3.

        :param reset_to_zero: Boolean to reset the voltage neuron to zero
        after spike. Default = True.

        :param tau_m: Value to control de adaptability of the time constant of
        the neurons. It can take the values: 'normal' for trainable tau_m;
        or a float number to make it fixed, un-trainable. If it has a fixed value
        of 0.0 then there will be no decay (IF neuron model). Default = 'normal'.

        :param win: Number of time-steps per sample. It must be setting with
        the same value as the one used in the dataset. Default = 50.

        :param surr: Surrogate gradient function used. It can take the values:
        'step' (step function), 'fs' (fast sigmoid), or 'mg' (multi-gaussian).
        Default = 'step'.

        :param loss_fn: Loss function. It can take the values: spk_count (MSELoss
        of spike counts), spk_count_smax (CE of softmaxed spike counts), 'mem_mot'
        (CE of softmaxed maximum Vmem across all tsteps), 'mem_sum'
        (CE of softmaxed cumulative Vmem over all tsteps), 'mem_last' (CE of softmaxed
        Vmem at the last tstep), mem_suppressed (LS loss of spike counts across timesteps,
        with regularised Vmems to emulate WTA). In all 'mem_' losses Vmems may decay
        but the output neurons have oo thress (dont fire)!
        Default = 'mem_sum'.

        :param batch_size: Number of samples in each batch. Default = 256.

        :param device: Device to run the training. It can take the values:
        'cuda' or 'cpu'. Default = 'cuda'.

        :param debug: Boolean for debug. Set True if you want to record
        internal states for all layers (membrane and spikes). Default = False.
        """

        super(SNN, self).__init__()
        
        # Gather keyword arguments for reproducibility in loaded models
        self.kwargs = locals()

        # Set attributes from inputs
        self.dataset_dict = dataset_dict
        self.structure = structure
        self.connection_type = connection_type
        self.delay = delay
        self.reset_to_zero = reset_to_zero
        self.tau_m = tau_m
        self.win = win        
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.device = device
        self.debug = debug

        # Setting dropout
        # self.dropout = torch.nn.Dropout(p=1.0)

        self.time_win = win  # win: the time of data, time_win the time of training

        # important parameters which are left fixed
        self.thresh = 0.3
        self.mean_tau = 20.0 # perez-nieves
        self.surr = 'fs'
        self.surr_scale = torch.tensor([10.0], requires_grad=False,
                                         device=self.device)
        self.bias = False
        self.multi_proj = None # must be defined externally for 'mf' connection type
        self.kwargs['multi_proj'] = self.multi_proj

        # By default, inputs are binarized according to this threshold (see
        # training/propagate). Set this to None if you want to allow floating
        # inputs
        self.input2spike_th = None

        # Asserts to check input arguments
        assert 'num_training_samples' in dataset_dict and \
               'num_input' in dataset_dict and 'num_output' in dataset_dict, \
            "[ERROR] Dataset dictionary don't have the right keys."
        assert type(structure) == tuple or type(structure) == list, \
            "[ERROR] The structure must be given with a list or a tuple."
        assert connection_type == 'f' or connection_type == 'r' or connection_type == 'mf', \
            "[ERROR] The connection_type must take the values 'f', 'r' or 'mf."
        assert delay is None or type(delay) == tuple, \
            "[ERROR] Delay must be a tuple or take the value None."
        assert reset_to_zero is True or reset_to_zero is False, \
            "[ERROR] Reset_to_zero argument must take the value True or False."
        assert type(win) == int, \
            "[ERROR] Window must be an integer."

        # Define the structure of the network
        if type(structure) == tuple:
            self.num_neurons_list = [structure[0] for _ in range(structure[1])]
        elif type(structure) == list:
            self.num_neurons_list = structure
        self.num_layers = len(self.num_neurons_list)

        # Set attributes for training metrics
        self.epoch = 0          # Number of epochs, initialized as 0
        self.acc = [[self.epoch, None]]     # Stores accuracy every time test()
                                            # method is called
        self.train_loss = []    # Stores loss during training
        self.test_loss = [[self.epoch, None]]   # Stores loss during testing
        self.test_spk_count = [[self.epoch, None]]  # Store spiking count
                                                    # during testing
        self.info = {}  # Store some useful info that you want to record for
                        # the save functionality (save_to_numpy method)

        # Set information about dataset, loss function, surrogate gradient,
        # criterion and optimizer functions (attributes initialized as None)
        self.num_train_samples = None
        self.num_input = None
        self.num_output = None
        self.act_fun = None
        self.criterion = None
        self.output_thresh = None
        self.optimizer = None
        self.stored_grads = None

        # Set propagation attributes (attributes initialized as None)
        self.h_sum_spike = None
        self.h_sum_spikes_per_layer = None

        # Set functions used for updating membrane potential (attributes
        # initialized as None)
        self.update_mem_fn = None 
        self.alpha_fn = None
        self.th_reset = None

        # Set model name (attributes initialized as None)
        self.model_name = None
        self.last_model_name = None
        self.last_max_model_name = None

        # Initialization of the layer and projections (weights) names
        self.layer_names = list()
        self.proj_names = list()

        # Set delays (attributes initialized as None)
        self.max_d = None
        self.stride = None
        self.delays = None
        self.delay_type = delay_type
        # self.input_names = None

        # Set parameters of hidden layers (attributes initialized as None)
        self.h_layers = None
        self.tau_m_h = None

        # Set features of the network
        self.define_metaparameters()
        self.define_model_name()
        self.to(self.device)

        # automatic mixed precision
        #self.use_amp = False
        self.use_amp = True # This only works in GPU
        self.scaler = amp.GradScaler()        

    def set_network(self):
        '''
        Initially, this was done during the initialization, but in order to
        add extra functionality which inherits from the base SNN and modifies
        how layers are configured, etc, it is better to call this after the __init__
        
        '''
        self.set_layers()
        self.set_tau_m()
        self.set_layer_lists()
        self.to(self.device)

    @staticmethod
    def weighted_mse_loss(output, target):
        weights = torch.abs(target - 1.0) + 1.0  # Higher weight for targets near 0 and 2
        return torch.mean(weights * (output - target) ** 2)

    def define_metaparameters(self):
        """
        Method to set up the number of input/outputs and train samples of
        the dataset specified; the surrogate gradient function, the loss
        function, the neuron update function, the delays and other
        user-specified parameter/function for the spiking neural network.
        """

        # Set parameters of the dataset
        self.num_train_samples = self.dataset_dict['num_training_samples']
        self.num_input = self.dataset_dict['num_input']
        self.num_output = self.dataset_dict['num_output']

        # Set surrogate gradient function
        if self.surr == 'fs':
            self.act_fun = ActFunFastSigmoid.apply

        # set loss function
        if self.loss_fn == 'spk_count':
            self.criterion = nn.MSELoss()
            self.output_thresh = self.thresh
        elif self.loss_fn == 'mem_mot' or self.loss_fn == 'mem_sum' or self.loss_fn == 'mem_last':
            self.criterion = nn.CrossEntropyLoss()
            self.output_thresh = 1e6  # Output neurons never fire
        elif self.loss_fn == 'mem_prediction':
            #self.criterion = nn.MSELoss()
            self.criterion = self.weighted_mse_loss
            self.output_thresh = 1e6  # Output neurons never fire

        # Set update function
        if self.connection_type == 'f':
            self.update_mem_fn = self.update_mem
        elif self.connection_type == 'r':
            self.update_mem_fn = self.update_mem_rnn
        elif self.connection_type == 'mf':
            self.update_mem_fn = self.update_mem_multi_proj

        # Set alpha function (sigmoid or exponential) for update membrane
        # potential
        if self.tau_m == 'IF':
            self.alpha_fn = self.alpha_none             # no decay, IF neuron model
        else:
            self.alpha_fn = self.alpha_sigmoid

        # Set reset
        if self.reset_to_zero:
            self.th_reset = self.thresh
        else:
            self.th_reset = 1e6

        # Set maximum delay, stride and delays
        if self.delay is None:  # No delays option
            self.delay = (1,1)

        self.stride = self.delay[1]     # Define stride

        # Define delays
        self.delays = torch.tensor(range(0, self.delay[0], self.stride))
        
        self.max_d = self.delays[-1] + 1    # Define maximum delay
        
        # Change for the new convention of 'delay_type'
        if self.delay_type == 'only_hidden':
            self.delay_type = 'ho'
            print(f'\n[INFO] Delay_type changed to {self.delay_type}')
        elif self.delay_type == 'only_input':
            self.delay_type = 'i'
            print(f'\n[INFO] Delay_type changed to {self.delay_type}')
        elif self.delay_type == 'all':
            self.delay_type = 'iho'
            
        assert (self.delay_type != 'o' and self.delay_type != 'io'), \
            "[ERROR] Output delays should always be together with hidden " \
            "delays"

        self.delays_i = self.delays
        self.delays_h = self.delays
        self.delays_o = self.delays

        if 'i' not in self.delay_type:
            self.delays_i = torch.tensor([0])
        if 'h' not in self.delay_type:
            self.delays_h = torch.tensor([0])
        if 'o' not in self.delay_type:
            self.delays_o = torch.tensor([0])    

        # Print information about delays
        print('\n[INFO] Delays: ' + str(self.delays))
        print('\n[INFO] Delays i: ' + str(self.delays_i))
        print('\n[INFO] Delays h: ' + str(self.delays_h))
        print('\n[INFO] Delays o: ' + str(self.delays_o))

        # Define hidden layer names
        self.layer_names = ['f' + str(x + 1) for x in range(self.num_layers)]

    # TODO: Adjust this to the new changes
    def define_model_name(self):
        """
        Function to define the model name based in the architecture of the
        neural network
        """

        dn = self.dataset_dict['dataset_name']

        self.model_name = \
            '{}{}_l{}_{}d{}.t7'.format(
                dn,
                self.win,
#                str(type(self)).split('.')[2][:-2], #?¿
                self.num_layers,
                self.delay[0],
                self.delay[1])

    def set_layers(self):
        """
        Function to set input, hidden and output layers as Linear layers. If the
        propagation mode include recurrence (self.connection_type = 'r'),
        additional layers (self.r_name) are created.
        """

        # Set bias
        bias = self.bias

        num_first_layer = self.num_neurons_list[0]

        # if delays is None, len(self.delays) = 1

        setattr(self, 'f0_f1', nn.Linear(self.num_input*len(self.delays_i),
                                    num_first_layer, bias=bias))            

        # Set linear layers dynamically for the l hidden layers
        for lay_name_1, lay_name_2, num_pre, num_pos in zip(self.layer_names[:-1],
         self.layer_names[1:], self.num_neurons_list[:-1], self.num_neurons_list[1:]):

            # This only if connection is recurrent
            if self.connection_type == 'r':
                name = lay_name_1 + '_' + lay_name_1
                # if self.mask is not None:
                #     setattr(self, name, MaskedLinear(
                #         num_pre* len(self.delays_h), num_pre, mask=self.mask))
                # else:
                #     setattr(self, name, nn.Linear(
                #         num_pre* len(self.delays_h), num_pre, bias=bias))
                setattr(self, name, nn.Linear(
                    num_pre* len(self.delays_h), num_pre, bias=bias))
                
                self.proj_names.append(name)

                # # Apply the mask to the weights after initialization
                # with torch.no_grad():
                #     getattr(self, name).weight *= self.mask                

            # Normal layer
            ### FIX this
            if self.connection_type == 'mf':
                if self.multi_proj is not None:
                    n_multi_proj = self.multi_proj
                else:
                    n_multi_proj = 3
            else:
                n_multi_proj = len(self.delays_h)

            name = lay_name_1 + '_' + lay_name_2
            setattr(self, name, nn.Linear(num_pre * n_multi_proj,
                                        num_pos, bias=bias))             
            self.proj_names.append(name)

        if self.connection_type == 'r':
            name = self.layer_names[-1] + '_' + self.layer_names[-1]
            setattr(self, name, nn.Linear(
                self.num_neurons_list[-1]* len(self.delays_h), self.num_neurons_list[-1], bias=bias))                              
            self.proj_names.append(name)

        # output layer
        name = self.layer_names[-1]+'_o'
        setattr(self, name, nn.Linear(self.num_neurons_list[-1] * len(self.delays_o),
                                    self.num_output, bias=bias ))
            
        self.proj_names.append(name)

    # TODO: Documentar y testar
    def set_tau_m(self):
        """
        Function to define the membrane time constants (taus). If tau_m is a float value
        then it is fixed to that value (non-trainable) for all neurons. If tau_m is 'gamma'
        they are randomly initialised from a gamma distribution and if 'normal' they are
        initialized from a Gaussian distribution. If they are randomly initialized then
        they are also left to be volatile during training (trainable).
        """

        logit = lambda x: np.log(x/(1-x))

        mean_tau = self.mean_tau # mean tau 20ms (Perez-Nieves)
        time_ms = self.dataset_dict.get('time_ms', 0)
        print(time_ms)

        if time_ms != 0:
            delta_t = time_ms/self.win
            print(f"Delta t: {delta_t} ms")
        else:
            raise Exception("Please define time_ms in dataset_dic.")

        if type(self.tau_m) == float:
            for i in range(self.num_layers):
                name = 'tau_m_' + str(i + 1)
                x = logit(np.exp(-delta_t/self.tau_m))
                setattr(self, name, nn.Parameter(x*torch.ones(self.num_neurons_list[i])))
                setattr(self, 'tau_m_o', nn.Parameter(x*torch.ones(self.num_output)))

        elif self.tau_m == 'normal':
            mean = logit(np.exp(-delta_t/mean_tau))
            print(f"mean of normal: {mean}")
            std = 1.0
            for i in range(self.num_layers):
                name = 'tau_m_' + str(i + 1)
                setattr(self, name, nn.Parameter(
                        torch.distributions.normal.Normal(
                            mean * torch.ones(self.num_neurons_list[i]),
                            std * torch.ones(self.num_neurons_list[i])).sample()))
            setattr(self, 'tau_m_o', nn.Parameter(
                torch.distributions.normal.Normal(
                            mean * torch.zeros(self.num_output),
                            std * torch.ones(self.num_output)).sample()))


    # TODO: Terminar documentacion y # Set ¿tau?
    def set_layer_lists(self):
        """
        Function to set layer lists.

        This function creates two lists, self.h_layers and self.tau_m_h, with
        the names of all the layers (connections, including hidden and
        recurrent layers) and ... ¿tau?
        """

        # Set list with all the projections (hidden and recurrent) for easier iteration
        self.h_layers = [nn.Identity()]
        for name in self.proj_names:
            self.h_layers.append(getattr(self, name))

        # Set list with all the taus for easier iteration
        # if self.tau_m == 'normal' or self.tau_m == 'gamma':
        self.tau_m_h = [getattr(self, name) for name in
                        ['tau_m_' + str(i + 1)
                            for i in range(self.num_layers)]]
        self.tau_m_h.append(self.tau_m_o)

    def init_state(self, input):
        """
        Function to set the initial state of the network. It initializes the
        membrane potential, the spikes and the spikes extended with the delays
        of all the neurons in hidden and output layer to zero. Also, the dictionary
        to log these parameters for debug is initialized with its parameters as
        zeros.

        :return: A tuple with the values of the extended spikes,
        the membrane potential of the hidden layer, the spikes of the hidden
        layer, the membrane potential of the output layer and the spikes of
        the output layer.
        """
        mems = dict()
        spikes = dict()
        extended_spikes = dict()
        setattr(self, 'mem_state', dict())
        setattr(self, 'spike_state', dict())

        if 'i' in self.delay_type:
            extended_input = torch.zeros(self.batch_size, self.win+self.max_d,
                                        self.num_input, device=self.device)
            extended_input[:, self.max_d:, :] = input
        else:
            extended_input = input

        # Initialization of membrane potential and spikes for hidden layers
        for name, num_hidden in zip(self.layer_names, self.num_neurons_list):

            if 'h' in self.delay_type:
                extended_spikes[name] = torch.zeros(
                    self.batch_size, self.win+self.max_d,
                    num_hidden, device=self.device)
            mems[name] = torch.zeros(
                self.batch_size, num_hidden, device=self.device)
            spikes[name] = torch.zeros(
                self.batch_size, num_hidden, device=self.device)


            # Initialization of the dictionary to log the state of the
            # network if debug is activated
            if self.debug:
                self.spike_state['input'] = torch.zeros(
                    self.win, self.batch_size,
                    self.num_input, device=self.device)
                self.mem_state[name] = torch.zeros(
                    self.win, self.batch_size,
                    num_hidden, device=self.device)
                self.spike_state[name] = torch.zeros(
                    self.win, self.batch_size,
                    num_hidden, device=self.device)
                self.mem_state['output'] = torch.zeros(
                    self.win, self.batch_size,
                    self.num_output, device=self.device)
                self.spike_state['output'] = torch.zeros(
                    self.win, self.batch_size,
                    self.num_output, device=self.device)

        # Initialization of membrane potential and spikes of output layer
        o_mem = torch.zeros(
            self.batch_size, self.num_output, device=self.device)
        o_spike = torch.zeros(
            self.batch_size, self.num_output, device=self.device)

        return extended_input, extended_spikes, mems, spikes, o_mem, o_spike

    def update_logger(self, *args):
        """
        Function to log the parameters if debug is activated. It creates a
        dictionary with the state of the neural network, recording the values
        of the spikes and membrane voltage for the input, hidden and output
        layers.

        This function takes as arguments the parameters of the network to log.
        """

        # Create the dictionary for logging
        if self.debug:
            x, mems, spikes, o_mem, o_spike = args

            self.spike_state['input'][self.step, :, :] = x
            for name in self.layer_names:
                self.mem_state[name][self.step, :, :] = mems[name]
                self.spike_state[name][self.step, :, :] = spikes[name]
            self.mem_state['output'][self.step, :, :] = o_mem
            self.spike_state['output'][self.step, :, :] = o_spike

    # TODO: Set attributes tau_idx and w_idx
    def update_mem(self, i_spike, o_spike, mem, thresh, _=None):

        """
        Function to update the membrane potential of the output layer. It takes
        into account the spikes coming from the hidden layer.

        :param i_spike: Input spike of the neuron.
        :param o_spike: Output spike of the neuron.
        :param mem: Membrane potential of the neuron.

        :return: A tuple with the membrane potential and output spike updated.
        """

        # Set alpha value to membrane potential decay
        alpha = self.alpha_fn(self.tau_m_h[self.tau_idx]).to(self.device)
        #alpha = torch.exp(-1. / self.tau_m_h[self.tau_idx]).to(self.device)

        # Calculate the new membrane potential and output spike
        mem = mem * alpha * (1 - o_spike) + self.h_layers[self.w_idx](i_spike)

        # o_spike = self.act_fun(mem-thresh)
        # mem = mem*(mem < self.th_reset)    

        # Update attributes
        self.tau_idx = self.tau_idx + 1
        self.w_idx = self.w_idx + 1

        return self.activation_function(mem, thresh)

    # TODO: Set attributes tau_idx and w_idx
    def update_mem_multi_proj(self, i_spike, o_spike, mem, thresh, _=None):

        """
        Function to update the membrane potential of the output layer. It takes
        into account the spikes coming from the hidden layer.

        :param i_spike: Input spike of the neuron.
        :param o_spike: Output spike of the neuron.
        :param mem: Membrane potential of the neuron.

        :return: A tuple with the membrane potential and output spike updated.
        """

        # Set alpha value to membrane potential decay
        alpha = self.alpha_fn(self.tau_m_h[self.tau_idx]).to(self.device)
        #alpha = torch.exp(-1. / self.tau_m_h[self.tau_idx]).to(self.device)

        # Calculate the new membrane potential and output spike
        if self.w_idx == 0:
            mem = mem * alpha * (1 - o_spike) + self.h_layers[self.w_idx](i_spike)
        else:
            mem = mem * alpha * (1 - o_spike) + self.h_layers[self.w_idx](i_spike.repeat(1, self.multi_proj))

        # o_spike = self.act_fun(mem-thresh)
        # mem = mem*(mem < self.th_reset)    

        # Update attributes
        self.tau_idx = self.tau_idx + 1
        self.w_idx = self.w_idx + 1

        return self.activation_function(mem, thresh)


    # TODO: Set attributes tau_idx and w_idx
    def update_mem_rnn(self, i_spike, o_spike, mem, thresh, extended_o_spikes):
        """
        Function to update the membrane potential of the hidden layer. It
        takes into account the spikes coming from the input layer and the
        output spikes from the own hidden layer because of the recurrence.

        :param i_spike: Input spike of the neuron.
        :param o_spike: Output spike of the neuron.
        :param mem: Membrane potential of the neuron.

        :return: A tuple with the membrane potential and output spike updated.
        """

        # Set alpha value to membrane potential decay
        alpha = self.alpha_fn(self.tau_m_h[self.tau_idx]).to(self.device)
        # alpha = torch.sigmoid(self.tau_m_h[self.tau_idx]).to(self.device)

        # Calculate the new membrane potential and output spike
        a = self.h_layers[self.w_idx](i_spike)  # From input spikes
        b = self.h_layers[self.w_idx+1](extended_o_spikes)    # From recurrent spikes
        c = mem * alpha * (1-o_spike)   # From membrane potential decay
        mem = a + b + c

        # o_spike = self.act_fun(mem-thresh)
        # mem = mem*(mem < self.th_reset)   

        # Update attributes
        self.tau_idx = self.tau_idx+1
        self.w_idx = self.w_idx + 2

        return self.activation_function(mem, thresh)


    def activation_function(self, mem, thresh):

        '''
        The activation function is defined here
        '''

        if self.th_reset < thresh:
            th_reset = thresh
        else:
            th_reset = self.th_reset

        o_spike = self.act_fun(mem-thresh, self.surr_scale)
        mem = mem*(mem < th_reset)

        #rand_th = th_reset + 0.3*torch.randn_like(mem)
        #mem = mem*(mem < rand_th)    
        #print(rand_th)

        return mem, o_spike


    @staticmethod
    def alpha_sigmoid(tau):
        return torch.sigmoid(tau)

    @staticmethod
    def alpha_exp(tau):
        return torch.exp(-1/tau)

    @staticmethod
    def alpha_none(tau):
        return torch.ones_like(tau)

    def forward(self, input):

        extended_input, extended_spikes, mems, spikes, o_mem, o_spike = self.init_state(input)
        self.o_sumspike = torch.zeros(
            self.batch_size, self.num_output, device=self.device)
        
        self.h_sum_spike = torch.tensor(0.0)  # for spike-regularization
        self.h_sum_spikes_per_layer = torch.zeros(self.num_layers)

        all_o_mems = []
        all_o_spikes = []    

        for step in range(self.win):

            self.step = step

            if 'i' in self.delay_type:
                delayed_x = extended_input[:, step + self.delays, :]
                prev_spikes = self.f0_f1(delayed_x.transpose(1, 2).reshape(
                    self.batch_size, -1))  # input layer is propagated (with delays)
            else:
                prev_spikes = self.f0_f1(input[:, step, :].view(self.batch_size, -1))

            self.w_idx = 0
            self.tau_idx = 0

            #self.dropout(prev_spikes)

            for i, layer in enumerate(self.layer_names):

                # # Uncomment for recurrent+delays layer
                # if self.connection_type == 'r' and 'h' in self.delay_type:
                #     r_ext_spk = extended_spikes[layer][:, step + self.delays, :].reshape(self.batch_size, -1)
                # else:
                #     r_ext_spk = spikes[layer]
                r_ext_spk = spikes[layer]

                mems[layer], spikes[layer] = self.update_mem_fn(
                    prev_spikes.reshape(self.batch_size, -1), spikes[layer], mems[layer], self.thresh, r_ext_spk)

                if 'h' in self.delay_type:
                    extended_spikes[layer][:, step+self.max_d,
                                        :] = spikes[layer].clone()  # possibly detach()
                    prev_spikes = extended_spikes[layer][:, step + self.delays, :].transpose(1, 2).clone()
                else:
                    prev_spikes = spikes[layer]

                #self.dropout(prev_spikes)

                self.h_sum_spike = self.h_sum_spike + spikes[layer].sum()

                # calculate avg spikes per layer
                self.h_sum_spikes_per_layer[i] = self.h_sum_spikes_per_layer[i] + spikes[layer].sum()

            if 'o' not in self.delay_type:
                prev_spikes = spikes[layer]
            o_mem, o_spike = self.update_mem(
                prev_spikes.reshape(self.batch_size, -1), o_spike, o_mem, self.output_thresh)
  
            self.update_logger(input[:, step, :].view(self.batch_size, -1), mems, spikes, o_mem, o_spike)

            all_o_mems.append(o_mem)
            all_o_spikes.append(o_spike)            

            self.o_sumspike = self.o_sumspike + o_spike


        self.h_sum_spike = self.h_sum_spike / self.num_layers

        return all_o_mems, all_o_spikes


    def save_model(self, model_name='rsnn', directory='default'):
        """
        Function to save model

        :param model_name: Name of the model (default = 'rsnn')
        :param directory: Directory to save the model (relative to
        CHECKPOINT_PATH) (default = 'default')
        """

        self.kwargs.pop('__class__', None)
        self.kwargs.pop('self', None)
        self.kwargs.pop('device', None)
        self.kwargs.pop('debug', None)

        # Create the state dictionary
        state = {
            'type': type(self),
            'net': self.state_dict(),
            'epoch': self.epoch,
            'acc_record': self.acc,
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'test_spk': self.test_spk_count,
            'model_name': self.model_name,
            'self.info': self.info,
            'kwargs': self.kwargs
        }

        # Define the path to save the model
        model_path = os.path.join(CHECKPOINT_PATH, directory)

        # If the directory do not exist, it is created
        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        # Save the model
        torch.save(state,
                   os.path.join(model_path, model_name),
                   _use_new_zipfile_serialization=False)
        print('Model saved in ', model_path)


    def remove_model(self, model_name='rsnn', directory='default'):
        model_path = os.path.join(CHECKPOINT_PATH, directory, model_name)
        os.remove(model_path)

class ModelLoader:
    """
    Model Loader class.

    Load a neural network previously trained and saved.

    arguments = model_name, location, batch_size, device, debug
    """

    def __new__(cls, *args, **kwargs):
        model_name, location, batch_size, device, debug = args

        params = torch.load(
            os.path.join(CHECKPOINT_PATH, location, model_name),
            map_location=torch.device('cpu'))

        params['kwargs']['batch_size'] = batch_size
        params['kwargs']['device'] = device
        params['kwargs']['debug'] = debug

        kwargs = params['kwargs']

        # For backwards compatibility
        if kwargs['tau_m'] == 'adp':
            print('[WARNING] Loading an old version, tau_m changed to gamma.')
            kwargs['tau_m'] = 'gamma'

        if 'dataset' in kwargs.keys():
            print('[WARNING] Loading an old version, check arguments below.')
            d = kwargs['dataset']
            del kwargs['dataset'] # Delete it from the stuff
            kwargs['dataset_dict'] = cls.__get_dict_old_way(cls, d)
            print(kwargs)
        
        # this work for the framework with snn_delays.models.snn.SNN
        #snn = params['type']
        #snn = snn(**kwargs) 
        
        snn = SNN(**kwargs)
        snn.to(device)
        snn.load_state_dict(params['net'], strict= False) # be careful with this
        snn.epoch = params['epoch']
        snn.acc = params['acc_record']
        snn.train_loss = params['train_loss']
        snn.test_loss = params['test_loss']
        snn.test_spk_count = params['test_spk']

        # For backwards compatibility
        if 'model_name' not in params.keys():
            print('[WARNING] Loading and old version, model_name changed '
                  'to default.')
            snn.model_name = 'default'

        print('Instance of {} loaded successfully'.format(params['type']))

        return snn
    
    def __get_dict_old_way(cls, dataset_name):

        dict_path = os.path.join(DATASET_PATH, 'dataset_configs',
                                dataset_name + '.json')

        if os.path.isfile(dict_path):
            with open(dict_path, 'r') as f:
                data_dict = json.load(f)

        else:
            sys.exit('[ERROR] The dictionary of the dataset used does not '
                    'exit. create the dictionary in dataset_configs')
            
        data_dict['num_training_samples'] = data_dict['num_train_samples']
        data_dict['dataset_name'] = dataset_name
        del data_dict['num_train_samples']

        return data_dict