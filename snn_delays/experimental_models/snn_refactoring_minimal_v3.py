from snn_delays.config import CHECKPOINT_PATH, DATASET_PATH
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import sys
import json
import numpy as np

'''
I'll try to make the models more Pytorch-like.
Step 1: simplify to the minimum.

CHANGES:
v3: queued spikes also defined externally, no clone needed, etc
'''


# class ActFunBase(torch.autograd.Function):
#     """
#     Base activation function class

#     The class implement the forward pass using a Heaviside function as
#     activation function. This is the usually choose for spiking neural
#     networks. The backward pass is only initialized, this method will be
#     rewritten with the surrogate gradient function in the child classes.
#     """

#     @staticmethod
#     def forward(ctx, input_data, scale_factor):
#         """
#         Forward pass

#         Take as input the tensor input_data (in general, the membrane
#         potential - threshold) and return a tensor with the same dimension
#         as input_data whose elements are 1.0 if the corresponding element in
#         input_data is greater than 0.0, and 0.0 otherwise.

#         The input parameter ctx is a context object that can be used to stash
#         information for backward computation.
#         """
#         ctx.save_for_backward(input_data, scale_factor)
#         return input_data.gt(0.0).float()

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Backward pass (this method will be rewritten with the surrogate
#         gradient function)
#         """
#         pass


# class ActFunFastSigmoid(ActFunBase):
#     """
#     Fast-sigmoid activation function class

#     It inherits methods from the ActFunBase class and rewrite the backward
#     method to include a surrogate gradient function.

#     Surrogate gradient function: Normalized negative part of a fast sigmoid
#     function (Reference: Zenke & Ganguli (2018))
#     """
#     def __init__(self):
#         """
#         Initialization of the activation function
#         """
#         super(ActFunBase, self).__init__()

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Backward pass

#         Surrogate gradient function: Normalized negative part of a fast
#         sigmoid function

#         The parameter 'scale' controls steepness of surrogate gradient.
#         """
#         # scale = 10.0

#         input_data, scale = ctx.saved_tensors
#         grad_input = grad_output.clone()

#         grad = grad_input / (scale*torch.abs(input_data) + 1.0) ** 2
#         return grad, None


### AI-condensed
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


class AbstractSNNLayer:

    act_fun = ActFunFastSigmoid.apply
    surr_scale = torch.tensor([10.0], requires_grad=False, device="cuda:0")

    @staticmethod
    def alpha_sigmoid(tau):
        return torch.sigmoid(tau)

    def activation_function(self, mem, thresh):

        '''
        The activation function is defined here
        '''

        th_reset = thresh # reset membrane when it reaches thresh

        o_spike = self.act_fun(mem-thresh, self.surr_scale)
        mem = mem * (mem < th_reset)

        return mem, o_spike

class FeedforwardSNNLayer(nn.Module, AbstractSNNLayer):

    '''
    advances a single timestep of a feedforward layer, with or without delays
    '''

    def __init__(self, num_in, num_out, tau_m, batch_size, inf_th = False, device = None, fanin_delays = None):
        '''
        initialize layer type, taus, etc
        prev_spikes: spikes from previous layer
        self.spikes spikes from current layer, and output spikes
        self.mems, mems from current layer
        delayed_spike_queue: to be filled with spike history from previous layer
        '''
        super().__init__()

        self.pre_delays = fanin_delays is not None
        if self.pre_delays:
            self.delays = fanin_delays
            self.max_d = self.delays[-1] + 1
            # self.register_buffer("delayed_spike_queue", torch.zeros(
            #         batch_size, max_d+1, num_in, device=device))
            # self.delayed_spike_queue = torch.zeros(
            #         batch_size, self.max_d+1, num_in, device=device)
            self.linear = nn.Linear(num_in*len(fanin_delays), num_out, bias=False)
        else:
            self.linear = nn.Linear(num_in, num_out, bias=False)
        
        self.num_in = num_in
        self.num_out = num_out

        self.tau_m = tau_m
        self.batch_size = batch_size
        self.device = device

        self.thresh = 1e6 if inf_th else 0.3


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

        # # Set alpha value to membrane potential decay
        # alpha = self.alpha_sigmoid(self.tau_m).to(self.device)

        # # Calculate the new membrane potential and output spike
        # new_mem = mem.clone() * alpha * (1 - o_spike) + self.linear(i_spike)
        # mem_out, o_spike_out = self.activation_function(new_mem, thresh)

        # return mem_out, o_spike_out

        # Set alpha value to membrane potential decay
        alpha = self.alpha_sigmoid(self.tau_m).to(self.device)

        # Calculate the new membrane potential and output spike
        mem = mem * alpha * (1 - o_spike) + self.linear(i_spike)

        return self.activation_function(mem, thresh)

    def forward(self, prev_spikes, own_mems, own_spikes):
        '''
        returns the mem and spike state
        '''

        # propagate previous spikes (wheter on simple or extended form)
        # update_mem(spikes_from_previous_layer, own_spikes, own_mems, threshols)
        mems, spikes = self.update_mem(
            prev_spikes.reshape(self.batch_size, -1), own_spikes, own_mems, self.thresh)
        
        # self.mems = mems
        # self.spikes = spikes

        return mems, spikes


class SNN(Training, nn.Module):
    """
    Spiking neural network (SNN) class.

    Common characteristic and methods for a spiking neural network with or
    without delays. It inherits from nn.Module.
    """

    def __init__(self, dataset_dict, num_hidden = 64, tau_m='normal', win=50,
                 loss_fn='mem_sum', batch_size=256, device='cuda'):
 
        super(SNN, self).__init__()
        
        # Gather keyword arguments for reproducibility in loaded models
        self.kwargs = locals()

        # Set attributes from inputs
        self.dataset_dict = dataset_dict
        self.num_hidden = num_hidden
        self.tau_m = tau_m
        self.win = win        
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.device = device

        self.time_win = win  # win: the time of data, time_win the time of training

        # important parameters which are left fixed
        self.mean_tau = 20.0 # perez-nieves

        # By default, inputs are binarized according to this threshold (see
        # training/propagate). Set this to None if you want to allow floating
        # inputs
        self.input2spike_th = None

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

        # Set features of the network
        self.num_train_samples = self.dataset_dict['num_training_samples']
        self.num_input = self.dataset_dict['num_input']
        self.num_output = self.dataset_dict['num_output']

        # set loss function
        if self.loss_fn == 'spk_count':
            self.criterion = nn.MSELoss()
        elif self.loss_fn == 'mem_mot' or self.loss_fn == 'mem_sum' or self.loss_fn == 'mem_last':
            self.criterion = nn.CrossEntropyLoss()

        self.to(self.device)    


    def set_layers(self):

        num_in = self.num_input # in this example it's 700

        num_h = 64

        num_o = self.num_output # in this example it's 20

        delays = [0, 16, 32]

        # l1 = FeedforwardSNNLayer(num_in, 
        #                          num_h, 
        #                          self.get_tau_m(num_h), 
        #                          self.batch_size, 
        #                          False,
        #                          self.device)
        
        # l2 = FeedforwardSNNLayer(num_h,
        #                          num_h, 
        #                          self.get_tau_m(num_h),
        #                          self.batch_size, 
        #                          False,
        #                          self.device)

        # l3 = FeedforwardSNNLayer(num_h, 
        #                          num_o, 
        #                          self.get_tau_m(num_o),
        #                          self.batch_size,
        #                          True, 
        #                          self.device)

        l1 = FeedforwardSNNLayer(num_in, 
                                 num_h, 
                                 self.get_tau_m(num_h), 
                                 self.batch_size, 
                                 False,
                                 self.device)
        

        l2 = FeedforwardSNNLayer(num_h,
                                 num_h, 
                                 self.get_tau_m(num_h),
                                 self.batch_size, 
                                 False,
                                 self.device, 
                                 delays)

        l3 = FeedforwardSNNLayer(num_h, 
                                 num_o, 
                                 self.get_tau_m(num_o),
                                 self.batch_size,
                                 True, 
                                 self.device)


        # 2 feedforwards
        self.layers = nn.ModuleList([l1, l2, l3])


    def init_state(self):

        '''
        Initially, I let the states to be initialized at __init__
        but they were added to the compute graph, I had to clone().detach()
        them before feeding them to update_mem, 
        and add retain_graph in the backward pass, hurting performance.
        Now, as in the previous version, the states are completely independent of the
        layer graph, they act just as external inputs.
        '''

        mems = dict()
        spikes = dict()
        queued_spikes = dict()

        for i, layer in enumerate(self.layers):
            
            num_neurons = layer.num_out
            num_in = layer.num_in
            name = 'l' + str(i+1)
            mems[name] = torch.zeros(
                self.batch_size, num_neurons, device=self.device)
            spikes[name] = torch.zeros(
                self.batch_size, num_neurons, device=self.device)
            
            if layer.pre_delays:
                max_d = layer.max_d
                queued_spikes[name] = torch.zeros(
                    self.batch_size,  max_d+1, num_in, device=self.device)

        # # Initialization of membrane potential and spikes for hidden layers
        # for name, num_hidden in zip(self.layer_names, self.num_neurons_list):
        #     mems[name] = torch.zeros(
        #         self.batch_size, num_hidden, device=self.device)
        #     spikes[name] = torch.zeros(
        #         self.batch_size, num_hidden, device=self.device)

        # # Initialization of membrane potential and spikes of output layer
        # o_mem = torch.zeros(
        #     self.batch_size, self.num_output, device=self.device)
        # o_spike = torch.zeros(
        #     self.batch_size, self.num_output, device=self.device)

        return mems, spikes, queued_spikes



    def get_tau_m(self, num_neurons):

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
            x = logit(np.exp(-delta_t/self.tau_m))
            return nn.Parameter(x*torch.ones(num_neurons))

        elif self.tau_m == 'normal':
            mean = logit(np.exp(-delta_t/mean_tau))
            print(f"mean of normal: {mean}")
            std = 1.0
            return nn.Parameter(
                    torch.distributions.normal.Normal(
                        mean * torch.ones(num_neurons),
                        std * torch.ones(num_neurons)).sample())

    @staticmethod
    def update_queue(tensor, data):
        '''
        for tensors of dimensions (batch_size, num_timesteps, num_neurons)
        '''
        # nice way
        tensor = torch.cat((tensor[:, -1:, :], tensor[:, :-1, :]), dim=1) # shif to the right
        tensor[:, 0, :] = data

        return tensor

    def forward(self, input):
    
        all_o_mems = []
        all_o_spikes = []    

        mems, spikes, queued_spikes = self.init_state()
        
        for step in range(self.win):

            prev_spikes = input[:, step, :].view(self.batch_size, -1)

            for layer, key in zip(self.layers, spikes.keys()):

                if layer.pre_delays:
                    queued_spikes[key] = self.update_queue(queued_spikes[key], prev_spikes)
                    #prev_spikes = queued_spikes[key][:, layer.delays, :].transpose(1, 2).clone().detach() # is transpose necessary?
                    prev_spikes = queued_spikes[key][:, layer.delays, :].transpose(1, 2)

                mems[key], spikes[key] = layer(prev_spikes, mems[key], spikes[key]) 
                prev_spikes = spikes[key]
            
            # last layer
            all_o_mems.append(mems[key])
            all_o_spikes.append(spikes[key])

        return all_o_mems, all_o_spikes