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
beta version
added structure with delays
added multifeedforward
added random delays
to add: save, load, spike counts, gradients checkpoints, amp
to try: 
'''

### AI-condensed
class ActFunFastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_data):
        ctx.save_for_backward(input_data)
        # Return a binary spike: 1 if input_data > 0, else 0
        return (input_data > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input_data, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Surrogate gradient: normalized negative part of a fast sigmoid
        grad = grad_input / (10.0 * torch.abs(input_data) + 1.0) ** 2
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
        self.incomplete_batch_len = 0
        
        if len(labels)<self.batch_size:
            self.incomplete_batch_len = len(labels)
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

        elif l_f == 'mem_prediction':

            perc = 0.9
            start_time = int(perc * self.win)
            a_o_m = all_o_mems[start_time:]
            # outputs = torch.mean(torch.stack(a_o_m, dim=1), dim = 1)
            outputs = torch.stack(a_o_m, dim=1).squeeze()

            # print(outputs.shape)
            # print(labels.shape)

        return outputs, labels
    

    def tb_synthetic(self):

        last_loss = np.array(self.train_loss)[-1,1]
        min_loss = np.min(np.array(self.train_loss)[:,1])

        # save last model, removing the previous one
        if hasattr(self, 'last_model_name'):
            self.remove_model(self.last_model_name, self.ckpt_dir)
        self.last_model_name = self.model_name+f'_last_{self.epoch+1}epoch'
        self.save_model(self.last_model_name, self.ckpt_dir)

        # save min loss model, removing the previous one
        if last_loss == min_loss:
            if hasattr(self, 'last_min_model_name'):
                self.remove_model(self.last_min_model_name, self.ckpt_dir)
            print(f'saving min loss: {min_loss}')
            self.last_min_model_name = self.model_name+ f'_minloss_{self.epoch+1}epoch'
            self.save_model(self.last_min_model_name, self.ckpt_dir)

    # TODO: Documentacion
    def train_step(self, train_loader=None, optimizer=None, scheduler= None, **kwargs):
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
        # num_iter = (self.num_train_samples // self.batch_size) + 1
        num_iter = len(train_loader)
        #sr = spk_reg / self.win

        if 'gradient_clipping' in kwargs:
            gradient_clipping = kwargs["gradient_clipping"]
        else:
            gradient_clipping = False

        if 'printed_steps' in kwargs:
            printed_steps = kwargs["printed_steps"]
        else:
            printed_steps = 3

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
            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            optimizer.step()
            #scheduler.step()
            scheduler.step(self.epoch + i / num_iter)

            should_print = (
                (num_iter >= 3 and (i + 1) % int(num_iter / printed_steps) == 0) or
                (num_iter < 3 and (i + 1) % num_iter == 0)
            )

            if should_print:
                progress = f"Step [{i + 1}/{self.num_train_samples // self.batch_size}]"
                print(f"{progress}, Loss: {running_loss:.5f}", flush=True)

                #### temporal feature for synthetic datasets...
                if "episodic" in self.dataset_dict["dataset_name"]:
                    self.epoch = i
                    self.train_loss.append([self.epoch, running_loss])        
                    self.tb_synthetic()            
                #print(f"l1_score: {l1_score}")

            # Reset running loss
            running_loss = 0

        if self.incomplete_batch_len == 0:
            num_train_samples = num_iter*self.batch_size
        else:
            num_train_samples = (num_iter-1)*self.batch_size + self.incomplete_batch_len 
        
        #print(num_train_samples)
        norm_iters = num_train_samples / self.batch_size

        if "episodic" not in self.dataset_dict["dataset_name"]:
            # Update parameters
            self.epoch = self.epoch + 1
            self.train_loss.append([self.epoch, total_loss_train / norm_iters])


    def test(self, test_loader=None, only_one_batch=False):
        """
        Function to run a test of the neural network over all the samples in
        test dataset


        :param test_loader: Test dataset (default = None)
        :param dropout: Percentage of randomly dropped spikes (applied to the
        input) (default = 0.0)
        """

        #### save gradient norms
        if self.save_gradients:
            gradient_norms = {
                name: float(f"{param.grad.data.norm(2).item():.2f}")
                for name, param in self.named_parameters()
                if param.grad is not None
            }
            
            print("Gradient norms:", gradient_norms)

        # Initializing simulation values to track
        correct = 0
        total = 0
        total_loss_test = 0
        total_spk_count = 0

        num_iter = len(test_loader)

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

            # total spike count
            spk_count = self.spike_count / (len(self.layers) -1)   

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
            total_spk_count += spk_count

            if only_one_batch:
                break

        if self.incomplete_batch_len == 0:
            num_test_samples = num_iter*self.batch_size
        else:
            num_test_samples = (num_iter-1)*self.batch_size + self.incomplete_batch_len 
        
        print(num_test_samples)
        norm_iters = num_test_samples / self.batch_size

        # Calculate accuracy
        acc = 100. * float(correct) / float(total)

       # update accuracy history
        if self.acc[-1][0] <= self.epoch:
            self.acc.append([self.epoch, acc])
            self.test_loss.append([self.epoch, total_loss_test / norm_iters])
            self.test_spk_count.append([self.epoch, total_spk_count/ total])
            if self.save_gradients:
                self.test_gradients.append([self.epoch, gradient_norms])
            # quitar penultimo acc, test_loss si coinciden las epocas o si se entrena por primera vez  
            if self.acc[-2][0] == self.epoch or self.acc[-2][1] == None:
                self.acc.pop(-2)
                self.test_loss.pop(-2)
                self.test_spk_count.pop(-2)
                if self.save_gradients:
                    self.test_gradients.pop(-2)

        # Print information about test
        print('Test Loss: {}'.format(total_loss_test / (i+1)))
        print('Avg spk_count per neuron for all {} time-steps {}'.format(
            self.win, total_spk_count / total))
        print('Test Accuracy of the model on the test samples: %.3f' % acc)
        print('', flush=True)

        return all_refs, all_preds


class AbstractSNNLayer(nn.Module):

    act_fun = ActFunFastSigmoid.apply

    @staticmethod
    def alpha_sigmoid(tau):
        return torch.sigmoid(tau)

    def activation_function(self, mem, thresh):

        '''
        The activation function is defined here
        '''

        th_reset = thresh # reset membrane when it reaches thresh

        o_spike = self.act_fun(mem-thresh)
        mem = mem * (mem < th_reset)

        return mem, o_spike
    
    def forward(self, prev_spikes, own_mems, own_spikes):
        '''
        returns the mem and spike state
        '''

        # propagate previous spikes (wheter on simple or extended form)
        # update_mem(spikes_from_previous_layer, own_spikes, own_mems, threshols)
        mems, spikes = self.update_mem(
            prev_spikes.reshape(self.batch_size, -1), own_spikes, own_mems, self.thresh)

        return mems, spikes
    
    def update_mem(self, i_spike, o_spike, mem, thresh):
        """Child classes must implement this."""
        pass

class ReciprocalSNNLayer(AbstractSNNLayer):

    '''
    '''

    def __init__(self, num_in_i, num_in_j, num_i, num_j, tau_m_i, tau_m_j, batch_size, 
                 inf_th=False, device=None):
        super().__init__()

        self.linear_i = nn.Linear(num_in_i, num_i, bias=False) # from external stimulus to i
        self.linear_j = nn.Linear(num_in_j, num_j, bias=False) # from external stimulus to j
        self.linear_ij = nn.Linear(num_i, num_j, bias=False) # from j to i
        self.linear_ji = nn.Linear(num_j, num_i, bias=False) # from i to j

        self.num_i = num_i
        self.num_j = num_j

        self.tau_m_i = tau_m_i
        self.tau_m_j = tau_m_j

        self.batch_size = batch_size
        self.device = device
        self.thresh = float('inf') if inf_th else 0.3

        # ## connectivity normalization (double check)
        # with torch.no_grad():

        #     max_scaler_i = torch.sqrt(torch.tensor(num_in_i+num_i))
        #     in_max_i = torch.sqrt(torch.tensor(num_in_i))
        #     rc_max_i = torch.sqrt(torch.tensor(num_i))
        #     self.linear_i.weight *= (in_max_i/max_scaler_i).item()
        #     self.linear_ji.weight *= (rc_max_i/max_scaler_i).item()

        #     max_scaler_j = torch.sqrt(torch.tensor(num_in_j+num_j))
        #     in_max_j = torch.sqrt(torch.tensor(num_in_j))
        #     rc_max_j = torch.sqrt(torch.tensor(num_j))
        #     self.linear_j.weight *= (in_max_j/max_scaler_j).item()
        #     self.linear_ij.weight *= (rc_max_j/max_scaler_j).item()


    def forward(self, prev_spikes_i, prev_spikes_j, own_mems_i, own_mems_j, own_spikes_i, own_spikes_j):
        '''
        returns the mem and spike state
        '''

        # propagate previous spikes (wheter on simple or extended form)
        # update_mem(spikes_from_previous_layer, own_spikes, own_mems, threshols)
        mems_i, mems_j, spikes_i, spikes_j = self.update_mem(
            prev_spikes_i.reshape(self.batch_size, -1), 
            prev_spikes_j.reshape(self.batch_size, -1), 
            own_spikes_i,
            own_spikes_j,
            own_mems_i,
            own_mems_j,
            self.thresh)

        return mems_i, mems_j, spikes_i, spikes_j

    def update_mem(self, i_spike, j_spike, o_spike_i, o_spike_j, mem_i, mem_j, thresh):
            
            """
            Update the membrane potential and output spike for the reciprocal SNN layer.
    
            :param i_spike: Input spikes from the previous layer.
            :param j_spike: Input spikes from the recurrent layer.
            :param o_spike_i: Output spikes from this layer (i).
            :param o_spike_j: Output spikes from the recurrent layer (j).
            :param mem_i: Membrane potential of this layer (i).
            :param mem_j: Membrane potential of the recurrent layer (j).
            :param thresh: Threshold for spike generation.
            """
    
            # Set alpha value to membrane potential decay
            alpha_i = self.alpha_sigmoid(self.tau_m_i).to(self.device)
            alpha_j = self.alpha_sigmoid(self.tau_m_j).to(self.device)
    
            # contributions from external stimulus to i and j
            a_i = self.linear_i(i_spike)
            a_j = self.linear_j(j_spike)
            
            # contributions from i to j and j to i
            b_i = self.linear_ij(o_spike_j)    # From i to j
            b_j = self.linear_ji(o_spike_i)    # From j to i

            #b_i = self.linear_ij(o_spike_i)    # From i to j
            #b_j = self.linear_ji(o_spike_j)    # From j to i

            # decay of membrane potential
            c_i = mem_i * alpha_i * (1-o_spike_i)   # From membrane potential decay
            c_j = mem_j * alpha_j * (1-o_spike_j)   # From membrane potential decay

            mem_i = a_i + b_i + c_i
            mem_j = a_j + b_j + c_j

            mem_i, o_spike_i = self.activation_function(mem_i, thresh)
            mem_j, o_spike_j = self.activation_function(mem_j, thresh)

            return mem_i, mem_j, o_spike_i, o_spike_j


class DelayedSNNLayer(AbstractSNNLayer):
    """
    Base class to handle fan-in delays and the multi-projections for SNN layers.
    This allows sharing delay logic across Feedforward and Recurrent layers.
    num_pruned delays: if 
    """
    def __init__(self, num_in, num_out, fanin_delays=None, pruned_delays=None):
        super().__init__()

        self.num_in = num_in
        self.num_out = num_out

        self.pre_delays = fanin_delays is not None
        
        self.pruned_delays = pruned_delays

        if self.pre_delays:
            self.delays = fanin_delays
            self.max_d = max(fanin_delays) + 1
            self.linear = nn.Linear(num_in * len(fanin_delays), num_out, bias=False)
            if pruned_delays is not None:
                mask = torch.rand(num_out, num_in * len(fanin_delays)) < (pruned_delays / len(self.delays))
                self.register_buffer('pruning_mask', mask)
                # Apply the mask to the weights after initialization
                with torch.no_grad():
                    self.linear.weight *= self.pruning_mask
        else:
            self.linear = nn.Linear(num_in, num_out, bias=False)

class MultiFeedforwardSNNLayer(AbstractSNNLayer):

    '''
    advances a single timestep of a multi-feedforward layer, with or without delays
    '''

    def __init__(self, num_in, num_out, tau_m, batch_size, inf_th=False, device=None, fanin_multifeedforward=None):
        super().__init__()

        self.num_in = num_in
        self.num_out = num_out
        self.pre_delays = False

        self.multif= fanin_multifeedforward
        self.linear = nn.Linear(num_in * fanin_multifeedforward, num_out, bias=False)

        self.tau_m = tau_m
        self.batch_size = batch_size
        self.device = device
        self.thresh = float('inf') if inf_th else 0.3
        self.multi_proj = fanin_multifeedforward

    def update_mem(self, i_spike, o_spike, mem, thresh):

        # Set alpha value to membrane potential decay
        alpha = self.alpha_sigmoid(self.tau_m).to(self.device)

        # Calculate the new membrane potential and output spike
        mem = mem * alpha * (1 - o_spike) + self.linear(i_spike.repeat(1, self.multif))

        return self.activation_function(mem, thresh)


class FeedforwardSNNLayer(DelayedSNNLayer):

    '''
    advances a single timestep of a feedforward layer, with or without delays
    '''

    def __init__(self, num_in, num_out, tau_m, batch_size, inf_th=False, device=None, fanin_delays=None, 
                 pruned_delays=None):
        super().__init__(num_in, num_out, fanin_delays, pruned_delays)
        self.tau_m = tau_m
        self.batch_size = batch_size
        self.device = device
        self.thresh = float('inf') if inf_th else 0.3

        if self.pre_delays:
            with torch.no_grad():
                scale_factor = torch.sqrt(self.max_d / self.pruned_delays).item()
                self.linear.weight *= scale_factor


    def update_mem(self, i_spike, o_spike, mem, thresh):

        # Set alpha value to membrane potential decay
        alpha = self.alpha_sigmoid(self.tau_m).to(self.device)

        # Calculate the new membrane potential and output spike
        mem = mem * alpha * (1 - o_spike) + self.linear(i_spike)

        return self.activation_function(mem, thresh)


class RecurrentSNNLayer(DelayedSNNLayer):

    '''
    advances a single timestep of a recurrent layer, with or without delays
    implemented delays only in the fanin connections, not in the recurrent ones.
    '''

    def __init__(self, num_in, num_out, tau_m, batch_size, inf_th=False, device=None, fanin_delays=None, 
                 pruned_delays=None):
        super().__init__(num_in, num_out, fanin_delays, pruned_delays)
        self.linear_rec = nn.Linear(num_out, num_out, bias=False)
        self.tau_m = tau_m
        self.batch_size = batch_size
        self.device = device
        self.thresh = float('inf') if inf_th else 0.3

        with torch.no_grad():

            mx = torch.sqrt(torch.tensor(num_in+num_out))
            ln_mx = torch.sqrt(torch.tensor(num_in))
            rc_mx = torch.sqrt(torch.tensor(num_out))

            self.linear.weight *= (ln_mx/mx).item()
            self.linear_rec.weight *= (rc_mx/mx).item()


    def update_mem(self, i_spike, o_spike, mem, thresh):

        # Set alpha value to membrane potential decay
        alpha = self.alpha_sigmoid(self.tau_m).to(self.device)

        # Calculate the new membrane potential and output spike
        a = self.linear(i_spike)  # From input spikes
        b = self.linear_rec(o_spike)    # From recurrent spikes
        c = mem * alpha * (1-o_spike)   # From membrane potential decay
        mem = a + b + c

        return self.activation_function(mem, thresh)


class SNN(Training, nn.Module):
    """
    Spiking neural network (SNN) class.

    Common characteristic and methods for a spiking neural network with or
    without delays. It inherits from nn.Module.
    """

    def __init__(self, dataset_dict, structure = [40, 40], tau_m='log-uniform', win=50,
                 loss_fn='mem_sum', batch_size=256, device='cuda', **extra_kwargs):
        
        '''
        structure: either (num_neurons, num_hidden_layers, connection_type)
        or (soon to be implemented) a list with specific configuration e.g d64f_64r
        '''
 
        super(SNN, self).__init__()
        
        # Gather keyword arguments for reproducibility in loaded models
        self.kwargs = locals()

        # Set attributes from inputs
        self.dataset_dict = dataset_dict
        self.structure = structure
        self.tau_m = tau_m
        self.win = win        
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.device = device
        self.debug = False

        self.extra_kwargs = extra_kwargs # options for delays and mf

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

        self.save_gradients = False
        self.test_gradients = [[self.epoch, None]] # Stores gradients during test 
    
        self.info = {}  # Store some useful info that you want to record for
                        # the save functionality (save_to_numpy method)

        # Set information about dataset, loss function, surrogate gradient,
        # criterion and optimizer functions (attributes initialized as None)
        self.act_fun = None
        self.criterion = None
        self.output_thresh = None
        self.optimizer = None

        # Set features of the network
#        self.num_train_samples = self.dataset_dict['num_training_samples']
        self.num_input_i = self.dataset_dict['num_input_i']
        self.num_input_j = self.dataset_dict['num_input_j']

        # # simulation time-step
        # self.step = -1

        self.model_name = ''

        # set loss function
        if self.loss_fn == 'spk_count':
            self.criterion = nn.MSELoss()
        elif self.loss_fn == 'mem_mot' or self.loss_fn == 'mem_sum' or self.loss_fn == 'mem_last':
            self.criterion = nn.CrossEntropyLoss()
            self.nonfiring_output = True
        elif self.loss_fn == 'mem_prediction':
            self.criterion = nn.MSELoss()
            self.nonfiring_output = True  # Output neurons never fire
            
        self.to(self.device)    

    def set_layers(self):
        '''
        quick option: a tuple of (num_hidden, num_layers, layer_type)
        if layer_type == d, fanin delays are placed in the second-to-last layer. 
        e. g: (48, 2, 'd') --> i-48-d-48-o
        '''

        num_in_i = self.num_input_i
        num_in_j = self.num_input_j
        num_i = self.structure[0]
        num_j = self.structure[1]
 
        self.layers = nn.ModuleList()

        # input layer:

        kwargs = {'num_in_i': num_in_i, 
                  'num_in_j': num_in_j,
                  'num_i': num_i,
                  'num_j': num_j,
                  'tau_m_i': self.get_tau_m(num_i),
                  'tau_m_j': self.get_tau_m(num_j),	  
                  'batch_size': self.batch_size, 
                  'inf_th': False,
                  'device': self.device}
        
        self.layers.append(ReciprocalSNNLayer(**kwargs))    


    def init_state(self):

        '''
        '''

        mems_i = dict()
        mems_j = dict()
        spikes_i = dict()
        spikes_j = dict()

        for i, layer in enumerate(self.layers):
            
            num_i = layer.num_i
            num_j = layer.num_j

            name = 'l' + str(i+1)

            mems_i[name] = torch.zeros(
                self.batch_size, num_i, device=self.device)
            spikes_i[name] = torch.zeros(
                self.batch_size, num_i, device=self.device)
            mems_j[name] = torch.zeros(
                self.batch_size, num_j, device=self.device)
            spikes_j[name] = torch.zeros(
                self.batch_size, num_j, device=self.device)

        return mems_i, mems_j, spikes_i, spikes_j


    def get_tau_m(self, num_neurons):

        logit = lambda x: np.log(x/(1-x))

        mean_tau = self.mean_tau # mean tau 20ms (Perez-Nieves)
        time_ms = self.dataset_dict.get('time_ms', 0)
        #print(time_ms)

        if time_ms != 0:
            delta_t = time_ms/self.win
            print(f"Delta t: {delta_t} ms")
        else:
            if self.tau_m == 'normal':
                raise Exception("Please define time_ms in dataset_dic.")

        if type(self.tau_m) == float:
            x = logit(np.exp(-delta_t/self.tau_m))
            return nn.Parameter(x*torch.ones(num_neurons))

        elif self.tau_m == 'normal': 
            #log-normal in reality, the name is kept for compatibility
            mean = logit(np.exp(-delta_t/mean_tau))
            #print(f"mean of normal: {mean}")
            std = 1.0
            return nn.Parameter(
                    torch.distributions.normal.Normal(
                        mean * torch.ones(num_neurons),
                        std * torch.ones(num_neurons)).sample())
        
        elif 'log-uniform' in self.tau_m:
            # Sample U uniformly in log space from 0.1 to max
            # if log-uniform, max = 10*num_timesteps
            # if log-uniform-st, max = 0.1*num_timesteps
            # if only one output neuron, tau=num_timesteps

            if self.tau_m == 'log-uniform':
                max_factor = 10
            elif self.tau_m == 'log-uniform-st':
                max_factor = 0.1

            # if num_neurons==self.num_output:
            #     log_tau_min = np.log(0.9*self.win)
            #     log_tau_max = np.log(1.1*self.win) 
            # else:
            #     log_tau_min = np.log(0.1)
            #     log_tau_max = np.log(max_factor*self.win)

            log_tau_min = np.log(0.1)
            log_tau_max = np.log(max_factor*self.win)

            U = np.random.uniform(log_tau_min, log_tau_max, size=(num_neurons,))
            
            # Compute M = -log(exp(exp(-U)) - 1)
            # exp_neg_U = np.exp(-U, dtype=np.float64)
            # exp_exp_neg_U = np.exp(exp_neg_U, dtype=np.float64)
            # M = -np.log(exp_exp_neg_U - 1.0)
            M = -np.log(np.exp(np.exp(-U)) - 1)

            return nn.Parameter(torch.tensor(M, dtype=torch.float))        

    def forward(self, input_i, input_j):
    
        all_i_mems = torch.zeros(self.win, self.batch_size, self.structure[0], device=self.device)
        all_i_spikes = torch.zeros(self.win, self.batch_size, self.structure[0], device=self.device) 
        all_j_mems = torch.zeros(self.win, self.batch_size, self.structure[1], device=self.device)
        all_j_spikes = torch.zeros(self.win, self.batch_size, self.structure[1], device=self.device)   

        mems_i, mems_j, spikes_i, spikes_j = self.init_state()

        for step in range(self.win):

            prev_spikes_i = input_i[:, step, :].view(self.batch_size, -1)
            prev_spikes_j = input_j[:, step, :].view(self.batch_size, -1)

            for layer, key in zip(self.layers, spikes_i.keys()):

                # prev_spikes_i, prev_spikes_j, own_mems_i, own_mems_j, own_spikes_i, own_spikes_j

                mems_i[key], mems_j[key], spikes_i[key], spikes_j[key] = layer(prev_spikes_i,
                                                                            prev_spikes_j,
                                                                            mems_i[key],
                                                                            mems_j[key],
                                                                            spikes_i[key],
                                                                            spikes_j[key])

                prev_spikes_i = spikes_i[key]
                prev_spikes_j = spikes_j[key]
            
            # # append results to activation list
            all_i_mems[step, :, :] = mems_i[key]
            all_i_spikes[step, :, :] = spikes_i[key]
            all_j_mems[step, :, :] = mems_j[key]
            all_j_spikes[step, :, :] = spikes_j[key]

        return all_i_mems, all_j_mems, all_i_spikes, all_j_spikes
    
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