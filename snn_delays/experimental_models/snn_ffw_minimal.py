from snn_delays.config import CHECKPOINT_PATH, DATASET_PATH
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import sys
import json
import numpy as np

class ActFunFastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_data, scale_factor):
        ctx.save_for_backward(input_data, scale_factor)
        # Return a binary spike: 1 if input_data > 0, else 0
        return (input_data > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input_data, scale = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Surrogate gradient: normalized negative part of a fast sigmoid
        grad = grad_input / (scale * torch.abs(input_data) + 1.0) ** 2
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
            images = torch.cat([images.view(self.batch_size, self.time_win, -1), zero_t], dim=1).float().to(self.device)

        elif self.time_win == self.win:
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

        return outputs, labels


    # TODO: Documentacion
    def train_step(self, train_loader=None, optimizer=None, scheduler= None):
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

            # Propagate data
            outputs, reference = self.propagate(images, labels)

            loss = self.criterion(outputs[:labels.size(0)], reference[:labels.size(0)])               

            running_loss += loss.detach().item()
            total_loss_train += loss.detach().item()
            total += labels.size(0)

            # Calculate gradients and optimize to update weights
            #loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            scheduler.step(self.epoch + i / num_iter)

            if (i + 1) % int(num_iter / 3.0) == 0:
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

        # Initialization to store predictions and references
        all_preds = list()
        all_refs = list()        

        # Testing loop over the test dataset
        for i, (images, labels) in enumerate(test_loader):
            
            # # Dropout
            #images = self.dropout(images.float())

            # # Propagate data
            with torch.no_grad():
                self.eval()
                outputs, reference = self.propagate(images, labels)
                self.train()

            # crop results to the labels size (for incomplete batch)
            if type(outputs) == list:
                outputs = [output[:labels.size(0)] for output in outputs]
            else: 
                outputs = outputs[:labels.size(0)]

            reference = reference[:labels.size(0)]        

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

            if only_one_batch:
                break

        # Calculate accuracy
        acc = 100. * float(correct) / float(total)

        # update accuracy history
        if self.acc[-1][0] <= self.epoch:
            self.acc.append([self.epoch, acc])
            self.test_loss.append([self.epoch, total_loss_test / (i+1)])
            if self.acc[-2][0] == self.epoch or self.acc[-2][1] == None:
                self.acc.pop(-2)
                self.test_loss.pop(-2)

        # Print information about test
        print('Test Loss: {}'.format(total_loss_test / (i+1)))
        print('Test Accuracy of the model on the test samples: %.3f' % acc)
        print('', flush=True)

        return all_refs, all_preds

class SNN(Training, nn.Module):
    """
    Spiking neural network (SNN) class.

    Common characteristic and methods for a spiking neural network with or
    without delays. It inherits from nn.Module.
    """

    def __init__(self, dataset_dict, num_neurons_list,
                 tau_m='normal', win=50,
                 loss_fn='mem_sum',
                 batch_size=256, device='cuda'):

        super(SNN, self).__init__()
        
        # Gather keyword arguments for reproducibility in loaded models
        self.kwargs = locals()

        # Set attributes from inputs
        self.dataset_dict = dataset_dict
        self.tau_m = tau_m
        self.win = win        
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.device = device

        self.time_win = win  # win: the time of data, time_win the time of training

        # important parameters which are left fixed
        self.thresh = 0.3
        self.mean_tau = 20.0 # perez-nieves
        self.surr = 'fs'
        self.surr_scale = torch.tensor([10.0], requires_grad=False,
                                         device=self.device)
        self.bias = False

        # By default, inputs are binarized according to this threshold (see
        # training/propagate). Set this to None if you want to allow floating
        # inputs
        self.input2spike_th = None

        self.num_neurons_list = num_neurons_list
        self.num_layers = len(self.num_neurons_list)

        # Set attributes for training metrics
        self.epoch = 0          # Number of epochs, initialized as 0
        self.acc = [[self.epoch, None]]     # Stores accuracy every time test()
                                            # method is called
        self.train_loss = []    # Stores loss during training
        self.test_loss = [[self.epoch, None]]   # Stores loss during testing

        # Set information about dataset, loss function, surrogate gradient,
        # criterion and optimizer functions (attributes initialized as None)
        self.num_train_samples = None
        self.num_input = None
        self.num_output = None
        self.act_fun = None
        self.criterion = None
        self.output_thresh = None
        self.optimizer = None


        # Set functions used for updating membrane potential (attributes
        # initialized as None)
        self.alpha_fn = None
        self.th_reset = None


        # Initialization of the layer and projections (weights) names
        self.layer_names = list()
        self.proj_names = list()

        # Set parameters of hidden layers (attributes initialized as None)
        self.h_layers = None
        self.tau_m_h = None

        # Set features of the network
        self.define_metaparameters()
        self.to(self.device)
    

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

        # Set alpha function for update membrane potential
        self.alpha_fn = self.alpha_sigmoid

        # Set reset
        self.th_reset = self.thresh

        # Define hidden layer names
        self.layer_names = ['f' + str(x + 1) for x in range(self.num_layers)]


    def set_layers(self):
        """
        Function to set input, hidden and output layers as Linear layers. If the
        propagation mode include recurrence (self.connection_type = 'r'),
        additional layers (self.r_name) are created.
        """

        # Set bias
        bias = self.bias

        num_first_layer = self.num_neurons_list[0]

        setattr(self, 'f0_f1', nn.Linear(self.num_input,
                                    num_first_layer, bias=bias))            

        # Set linear layers dynamically for the l hidden layers
        for lay_name_1, lay_name_2, num_pre, num_pos in zip(self.layer_names[:-1],
         self.layer_names[1:], self.num_neurons_list[:-1], self.num_neurons_list[1:]):

            name = lay_name_1 + '_' + lay_name_2
            setattr(self, name, nn.Linear(num_pre, num_pos, bias=bias))             
            self.proj_names.append(name)

        # output layer
        name = self.layer_names[-1]+'_o'
        setattr(self, name, nn.Linear(self.num_neurons_list[-1],
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

    def init_state(self):
        mems = dict()
        spikes = dict()

        # Initialization of membrane potential and spikes for hidden layers
        for name, num_hidden in zip(self.layer_names, self.num_neurons_list):
            mems[name] = torch.zeros(
                self.batch_size, num_hidden, device=self.device)
            spikes[name] = torch.zeros(
                self.batch_size, num_hidden, device=self.device)

        # Initialization of membrane potential and spikes of output layer
        o_mem = torch.zeros(
            self.batch_size, self.num_output, device=self.device)
        o_spike = torch.zeros(
            self.batch_size, self.num_output, device=self.device)

        return mems, spikes, o_mem, o_spike
    

    # TODO: Set attributes tau_idx and w_idx
    def update_mem(self, i_spike, o_spike, mem, thresh):

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

        return mem, o_spike


    @staticmethod
    def alpha_sigmoid(tau):
        return torch.sigmoid(tau)

    def forward(self, input):

        mems, spikes, o_mem, o_spike = self.init_state()

        all_o_mems = []
        all_o_spikes = []    

        for step in range(self.win):

            prev_spikes = self.f0_f1(input[:, step, :].view(self.batch_size, -1))

            self.w_idx = 0
            self.tau_idx = 0

            for i, layer in enumerate(self.layer_names):

                mems[layer], spikes[layer] = self.update_mem(
                    prev_spikes.reshape(self.batch_size, -1), spikes[layer], mems[layer], self.thresh)

                prev_spikes = spikes[layer]
                       
            o_mem, o_spike = self.update_mem(
                prev_spikes.reshape(self.batch_size, -1), o_spike, o_mem, self.output_thresh)

            all_o_mems.append(o_mem)
            all_o_spikes.append(o_spike)            

        return all_o_mems, all_o_spikes