from snn_delays.config import CHECKPOINT_PATH, DATASET_PATH
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import sys
import json
import numpy as np
from collections import deque
from snn_delays.layers import (Conv2DSNNLayer, Conv3DSNNLayer, FlattenSNNLayer,
                               FeedforwardSNNLayer, RecurrentSNNLayer,
                               MultiFeedforwardSNNLayer, DelayedSNNLayer)

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

        if self.null_category == True:
            labels = labels[:, :self.num_output]  # remove the null category
        
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

        ### TODO: handle different time propagations to that of the trained models
        # ### zero-padding the inputs along the temporal dimension
        # if self.time_win<self.win:
        #     zero_t = torch.zeros(self.batch_size, self.win-self.time_win, images.size(2)*images.size(3), dtype=images.dtype, device=self.device)
        #     images = torch.cat([images.view(self.batch_size, self.time_win, -1), zero_t], dim=1).float().to(self.device)

        # elif self.time_win == self.win:
        #     # old implementaton (no conv, flattening the inputs)
        #     #images = images.view(self.batch_size, self.win, -1).float().to(self.device)      
                
        #     images = images.float().to(self.device)

        # else:
        #     raise Exception("propagation time below data timesteps not implemented yet!")


        images = images.float().to(self.device)

        # Squeeze to eliminate dimensions of size 1    
        if len(images.shape)>3 and self.batch_size>1:    
            images = images.squeeze()

        if self.batch_size>1:
            labels = labels.float().squeeze().to(self.device)
        else:
            labels = labels.float().to(self.device)
            
        ### propagation
        all_o_mems, all_o_spikes = self(images)

        ### loss function implementation
        l_f = self.loss_fn

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
    

    def propagate_live(self, images, spk_count_size = None):
        '''
        this is to propagate and test streaming data.
        there is no need to pass labels.
        '''

        if spk_count_size is None:
            spk_count_size = self.win

        with torch.no_grad():

            self.eval()

            # Resize and reformat of images and labels
            if self.input2spike_th is not None:
                images = images > self.input2spike_th

            images = images.to(self.device)

            #images = images.view(self.batch_size, self.win, -1).float().to(self.device)
            #images = images.view(self.batch_size, -1).float().to(self.device)
            images = images.reshape(self.batch_size, -1).float().to(self.device)          

            # Squeeze to eliminate dimensions of size 1    
            ### careful with this if batch size is 1
            if len(images.shape)>3 and self.batch_size>1:    
                images = images.squeeze()

            l_f = self.loss_fn

            #o_mems, o_spikes = self.forward_live(images)
            mems, spikes = self.forward_live(images)

            # o_mems = mems['output']
            # o_spikes = spikes['output']

            if len(self.mems_fifo['output']) == self.win:
                for key in self.mems_fifo.keys():
                    self.mems_fifo[key].popleft()
                    self.spikes_fifo[key].popleft()

            for key in self.mems_fifo.keys():
                self.mems_fifo[key].append(mems[key])
                self.spikes_fifo[key].append(spikes[key]) 

            if l_f == 'mem_last':
#                _, labels = torch.max(labels.data, 1)
                outputs = F.softmax(self.mems_fifo['output'][-1], dim=1)

            elif l_f == 'mem_sum':
                outputs = torch.zeros(
                self.batch_size, self.num_output, device=self.device)
#                _, labels = torch.max(labels.data, 1)
                for o_mem in self.mems_fifo['output']:

                    outputs = outputs + F.softmax(o_mem, dim=1)

            elif l_f == 'mem_mot':
                # as in the zenke tutorial
#                _, labels = torch.max(labels.data, 1)
                m,_=torch.max(torch.stack(self.mems_fifo['output'], dim=1), 1)
                outputs = F.softmax(m, dim=1)

            elif l_f == 'spk_count':
                outputs = torch.zeros(
                self.batch_size, self.num_output, device=self.device)
                for o_spk in list(self.spikes_fifo['output'])[-spk_count_size:]:
                    outputs = outputs + o_spk
                # outputs = outputs/self.win   #normalized         
                # outputs = torch.sum(outputs, dim = 1)/self.win

            elif l_f == 'mem_prediction':

                perc = 0.9
                start_time = int(perc * self.win)
                a_o_m = self.mems_fifo['output'][start_time:]
                # outputs = torch.mean(torch.stack(a_o_m, dim=1), dim = 1)
                outputs = torch.stack(a_o_m, dim=1).squeeze()

            _, predicted = torch.max(outputs.data, 1)

        return predicted.item()
            # print(predicted)

            # print(outputs.shape)
            # print(labels.shape)

#        return outputs, labels

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
                #self.eval()
                outputs, reference = self.propagate(images, labels)
                #self.train()

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


class SNN(Training, nn.Module):
    """
    Spiking neural network (SNN) class.

    Common characteristic and methods for a spiking neural network with or
    without delays. It inherits from nn.Module.
    """

    def __init__(self, dataset_dict, model_config, tau_m='normal', win=50,
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
        self.model_config = model_config
        self.tau_m = tau_m
        self.win = win
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.device = device
        self.debug = False

        self.extra_kwargs = extra_kwargs # options for delays and mf

        self.time_win = win  # win: the time of data, time_win the time of training
        self.num_simulation_steps = win

        # important parameters which are left fixed
        self.mean_tau = 20.0 # perez-nieves

        # By default, inputs are binarized according to this threshold (see
        # training/propagate). Set this to None if you want to allow floating
        # inputs
        self.input2spike_th = None
        self.null_category = False  # if the dataset has a null category, set it here

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
        self.num_train_samples = None
        self.num_input = None
        self.num_output = None
        self.act_fun = None
        self.criterion = None
        self.output_thresh = None
        self.optimizer = None

        # Set features of the network
        self.num_train_samples = self.dataset_dict['num_training_samples']
        self.num_input = self.dataset_dict['num_input']
        self.num_input_channels = self.dataset_dict.get('num_input_channels', 1)  # Default to 1 if not specified
        self.num_output = self.dataset_dict['num_output']

        # initialize layer list
        self.layers = nn.ModuleList()

        # # simulation time-step
        # self.step = -1

        self.live = False

        self.model_name = ''
        self.last_model_name = None
        self.last_max_model_name = None

        # get delta_t
        time_ms = self.dataset_dict.get('time_ms', 0)
        if time_ms != 0:
            self.delta_t = time_ms/self.win
            print(f"Delta t: {self.delta_t} ms")
        else:
            raise Exception("Please define time_ms in dataset_dic.")

        # set loss function
        if self.loss_fn == 'spk_count':
            self.criterion = nn.MSELoss()
            self.nonfiring_output = False
        elif self.loss_fn == 'mem_mot' or self.loss_fn == 'mem_sum' or self.loss_fn == 'mem_last':
            self.criterion = nn.CrossEntropyLoss()
            self.nonfiring_output = True
        elif self.loss_fn == 'mem_prediction':
            self.criterion = nn.MSELoss()
            self.nonfiring_output = True  # Output neurons never fire
            
        self.to(self.device)    


    def set_layers(self):

        logit = lambda x: np.log(x/(1-x))
        conv_tau_m = 5.0 # Default tau_m for Conv2D layers, can be adjusted (ms)

        current_in_channels = self.num_input_channels
        if current_in_channels == 1:
            current_in_height = self.num_input
            current_in_width = 1
        elif current_in_channels == 2: # assume 2D square input
            current_in_height = int((self.num_input // 2) ** 0.5)
            current_in_width = current_in_height

        for i, config in enumerate(self.model_config):
            layer_type = config['type']
            layer_kwargs = {} # Collect kwargs for the current layer

            # Common SNN layer arguments
            layer_kwargs['batch_size'] = self.batch_size
            layer_kwargs['device'] = self.device

            if layer_type == 'Conv2D':
                # Conv2DSNNLayer specific arguments
                layer_kwargs['in_channels'] = current_in_channels
                layer_kwargs['out_channels'] = config['out_channels']
                layer_kwargs['kernel_size'] = config['kernel_size']
                layer_kwargs['stride'] = config.get('stride', config['kernel_size'])  # Default to kernel_size if not specified
                layer_kwargs['tau_m'] = torch.tensor(logit(np.exp(-self.delta_t/conv_tau_m)), device=self.device)
                layer_kwargs['avg_pooling'] = config.get('avg_pooling', False)  # Default to False if not specified

                layer_instance = Conv2DSNNLayer(**layer_kwargs)
                self.layers.append(layer_instance)

                # Update dimensions for the next layer
                current_in_channels = layer_kwargs['out_channels']

                kernel_h = layer_kwargs['kernel_size']
                kernel_w = layer_kwargs['kernel_size']
                stride_h = layer_kwargs['stride']
                stride_w = layer_kwargs['stride']

                current_in_height = int(np.floor(
                    (current_in_height - kernel_h) / stride_h + 1
                ))
                current_in_width = int(np.floor(
                    (current_in_width - kernel_w) / stride_w + 1
                ))

                if layer_kwargs['avg_pooling']:
                    current_in_height = current_in_height // 2  # Assuming avg pooling halves the height
                    current_in_width = current_in_width // 2

                # current_in_height = current_in_height // layer_kwargs['kernel_size'] # Assuming stride=kernel_size
                # current_in_width = current_in_width // layer_kwargs['kernel_size'] # Assuming stride=kernel_size

                # update output shape information (useful for self.init_state())
                self.layers[-1].output_shape = (current_in_channels, current_in_height, current_in_width)

            elif layer_type == 'Conv3D':
                # Conv3DSNNLayer specific arguments
                layer_kwargs['in_channels'] = current_in_channels
                layer_kwargs['out_channels'] = config['out_channels']
                layer_kwargs['kernel_size'] = config['kernel_size']
                layer_kwargs['temporal_kernel_size'] = config.get('fanin_delays', 1)
                layer_kwargs['avg_pooling'] = config.get('avg_pooling', False)  # Default to False if not specified

                layer_kwargs['tau_m'] = torch.tensor(logit(np.exp(-self.delta_t/conv_tau_m)), device=self.device)

                layer_instance = Conv3DSNNLayer(**layer_kwargs)
                self.layers.append(layer_instance)

                # Update dimensions for the next layer
                current_in_channels = layer_kwargs['out_channels']
                current_in_height = current_in_height // layer_kwargs['kernel_size'] # Assuming stride=kernel_size
                current_in_width = current_in_width // layer_kwargs['kernel_size'] # Assuming stride=kernel_size

                if layer_kwargs['avg_pooling']:
                    current_in_height = current_in_height // 2  # Assuming avg pooling halves the height
                    current_in_width = current_in_width // 2

                # update output shape information (useful for self.init_state())
                self.layers[-1].output_shape = (current_in_channels, current_in_height, current_in_width)

            elif layer_type == 'Flatten':
                layer_instance = FlattenSNNLayer(batch_size=self.batch_size)
                self.layers.append(layer_instance)
                
                # Update dimensions for the next layer (flattened size)
                current_in_channels = current_in_channels * current_in_height * current_in_width
                # current_in_height = 1 # No spatial dimensions after flatten
                # current_in_width = 1 # No spatial dimensions after flatten

                # update output shape information (useful for self.init_state())
                self.layers[-1].output_shape = (current_in_channels,)

            elif layer_type == 'Feedforward':
                # num_in is the current_in_channels (from previous layer's output)
                layer_kwargs['num_in'] = current_in_channels
                layer_kwargs['num_out'] = config['num_out']
                layer_kwargs['tau_m'] = self.get_tau_m(config['num_out'])

                # Specific kwargs for DelayedSNNLayer (if Feedforward is a DelayedSNNLayer)
                if 'fanin_delays' in config:
                    max_delay = config['fanin_delays']['max_delay']
                    stride = config['fanin_delays']['stride']
                    pruned_delays= config['fanin_delays'].get('pruning', None)
                    layer_kwargs['fanin_delays'] = torch.tensor(range(0, max_delay, stride))
                    if pruned_delays is not None:
                        layer_kwargs['pruned_delays'] = pruned_delays

                layer_instance = FeedforwardSNNLayer(**layer_kwargs)
                self.layers.append(layer_instance)
                
                # Update dimensions for the next layer
                current_in_channels = layer_kwargs['num_out']

                # update output shape information (useful for self.init_state())
                self.layers[-1].output_shape = (current_in_channels,)

            elif layer_type == 'Recurrent':
                # num_in is the current_in_channels
                layer_kwargs['num_in'] = current_in_channels
                layer_kwargs['num_out'] = config['num_out']
                layer_kwargs['tau_m'] = self.get_tau_m(config['num_out'])

                # Specific kwargs for DelayedSNNLayer if Recurrent inherits from it
                if 'fanin_delays' in config:
                    max_delay = config['fanin_delays']['max_delay']
                    stride = config['fanin_delays']['stride']
                    pruned_delays= config['fanin_delays'].get('pruning', None)
                    layer_kwargs['fanin_delays'] = torch.tensor(range(0, max_delay, stride))
                    if pruned_delays is not None:
                        layer_kwargs['pruned_delays'] = pruned_delays

                layer_instance = RecurrentSNNLayer(**layer_kwargs)
                self.layers.append(layer_instance)
                
                # Update dimensions for the next layer
                current_in_channels = layer_kwargs['num_out']

                # update output shape information (useful for self.init_state())
                self.layers[-1].output_shape = (current_in_channels,)

            elif layer_type == 'MultiFeedforward':
                # num_in is the current_in_channels
                layer_kwargs['num_in'] = current_in_channels
                layer_kwargs['num_out'] = config['num_out']
                layer_kwargs['tau_m'] = self.get_tau_m(config['num_out'])
                layer_kwargs['num_fanin_multifeedforward'] = config['num_fanin_multifeedforward'] # Assuming this key is used

                layer_instance = MultiFeedforwardSNNLayer(**layer_kwargs)
                self.layers.append(layer_instance)
                
                # Update dimensions for the next layer
                current_in_channels = layer_kwargs['num_out']

                # update output shape information (useful for self.init_state())
                self.layers[-1].output_shape = (current_in_channels,)

        # Special handling for the last layer: num_out should be self.num_output_classes
        layer_kwargs = {} # Collect kwargs for the current layer
        layer_kwargs['batch_size'] = self.batch_size
        layer_kwargs['device'] = self.device
        layer_kwargs['num_in'] = current_in_channels
        layer_kwargs['num_out'] = self.num_output
        layer_kwargs['tau_m'] = self.get_tau_m(self.num_output)
        layer_kwargs['inf_th'] = self.nonfiring_output
        #layer_kwargs['apply_bn'] = False  # No batch normalization on the output layer
        layer_instance = FeedforwardSNNLayer(**layer_kwargs)
        self.layers.append(layer_instance)

        # update output shape information (useful for self.init_state())
        self.layers[-1].output_shape = (self.num_output,)

        # if self.debug:
        #     self.init_state_logger()


    def set_live_mode(self, inference_window=None):

        # set fifos for the live mode
        if self.live:

            if inference_window is None:
                inference_window = self.win

            self.mems_fifo = dict()
            self.spikes_fifo = dict()
            
            for i in range(len(self.layers)):
                if i == len(self.layers)-1:
                    name = 'output'
                else:
                    name = 'l' + str(i+1)
                self.mems_fifo[name] = deque(maxlen=inference_window)
                self.spikes_fifo[name] = deque(maxlen=inference_window)

            self.reset_state_live()


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

        ### state logger
        if self.debug:
            
            setattr(self, 'mem_state', dict())
            setattr(self, 'spike_state', dict())    

            in_channels = self.num_input_channels
            if in_channels == 1:
                self.spike_state['input'] = torch.zeros(
                    self.win, self.batch_size,
                    self.num_input, device=self.device)
            elif in_channels == 2: # assume 2D square input
                in_height = int((self.num_input // 2) ** 0.5)
                in_width = in_height
                self.spike_state['input'] = torch.zeros(
                    self.win, self.batch_size,
                    in_channels, in_width, in_height, device=self.device)

        for i, layer in enumerate(self.layers):

            output_shape = layer.output_shape  # Get the output shape from the layer's attribute
            
            # Determine the name for the state dictionary (improve this later)
            if i == len(self.layers)-1:
                name = 'output'
            else:
                name = 'l' + str(i)

            # Initialize mems and spikes with the determined output_shape
            mems[name] = torch.zeros(
                self.batch_size, *output_shape, device=self.device)
            spikes[name] = torch.zeros(
                self.batch_size, *output_shape, device=self.device)
            
            if self.debug:
                self.mem_state[name] = torch.zeros(
                self.win, self.batch_size, *output_shape, device=self.device)
                self.spike_state[name] = torch.zeros(
                self.win, self.batch_size, *output_shape, device=self.device)            
                
            if isinstance(layer, DelayedSNNLayer) and layer.pre_delays:
                max_d = layer.max_d
                queued_spikes[name] = torch.zeros(
                    self.batch_size,  max_d+1, layer.num_in, device=self.device)
                
            elif isinstance(layer, Conv3DSNNLayer):

                # get previous layer's output shape (fanin activations)
                if i > 0:
                    prev_layer = self.layers[i-1]
                    input_shape = prev_layer.output_shape
                else:
                    H = int((self.num_input // 2) ** 0.5)
                    W = H
                    input_shape = (self.num_input_channels, H, W)

                max_d = layer.temporal_kernel_size
                queued_spikes[name] = torch.zeros(
                    self.batch_size,  max_d, *input_shape, device=self.device)                

        return mems, spikes, queued_spikes
    

    def reset_state_live(self):
        '''
        same as above, but for live mode. 
        '''

        self.mems = dict()
        self.spikes = dict()
        self.queued_spikes = dict()

        for i, layer in enumerate(self.layers):
            
            num_neurons = layer.num_out
            num_in = layer.num_in
            
            if i == len(self.layers)-1:
                name = 'output'
            else:
                name = 'l' + str(i)

            self.mems[name] = torch.zeros(
                self.batch_size, num_neurons, device=self.device)
            self.spikes[name] = torch.zeros(
                self.batch_size, num_neurons, device=self.device)
            
            if layer.pre_delays:
                max_d = layer.max_d
                self.queued_spikes[name] = torch.zeros(
                    self.batch_size,  max_d+1, num_in, device=self.device)


    # def init_state_logger(self):

    #     # Initialization of the dictionary to log the state of the
    #     # network if debug is activated

    #     with torch.no_grad():

    #         setattr(self, 'mem_state', dict())
    #         setattr(self, 'spike_state', dict())        

    #         if self.debug:
    #             self.spike_state['input'] = torch.zeros(
    #                 self.win, self.batch_size,
    #                 self.num_input, device=self.device)
                
    #         for i, layer in enumerate(self.layers):

    #             num_neurons = layer.num_out

    #             if i == len(self.layers)-1:
    #                 name = 'output'
    #             else:
    #                 name = 'l' + str(i)

    #             self.mem_state[name] = torch.zeros(
    #                 self.win, self.batch_size,
    #                 num_neurons, device=self.device)
    #             self.spike_state[name] = torch.zeros(
    #                 self.win, self.batch_size,
    #                 num_neurons, device=self.device)


    def update_logger(self, *args):
        """
        Function to log the parameters if debug is activated. It creates a
        dictionary with the state of the neural network, recording the values
        of the spikes and membrane voltage for the input, hidden and output
        layers.

        This function takes as arguments the parameters of the network to log.
        """

        if self.debug:
            inpt, mems, spikes, step = args

            self.spike_state['input'][step, ...] = inpt.detach().clone()

            for i in range(len(self.layers)):

                if i == len(self.layers)-1:
                    name = 'output'
                else:
                    name = 'l' + str(i)

                self.mem_state[name][step, ...] = mems[name].detach().clone()
                self.spike_state[name][step, ...] = spikes[name].detach().clone()

    def get_tau_m(self, num_neurons):

        logit = lambda x: np.log(x/(1-x))

        mean_tau = self.mean_tau # mean tau 20ms (Perez-Nieves)

        if type(self.tau_m) == float:
            x = logit(np.exp(-self.delta_t/self.tau_m))
            return nn.Parameter(x*torch.ones(num_neurons))

        elif self.tau_m == 'normal': 
            #log-normal in reality, the name is kept for compatibility
            mean = logit(np.exp(-self.delta_t/mean_tau))
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

            if num_neurons==self.num_output:
                log_tau_min = np.log(0.9*self.win)
                log_tau_max = np.log(1.1*self.win) 
            else:
                log_tau_min = np.log(0.1)
                log_tau_max = np.log(max_factor*self.win)
            U = np.random.uniform(log_tau_min, log_tau_max, size=(num_neurons,))
            
            # Compute M = -log(exp(exp(-U)) - 1)
            # exp_neg_U = np.exp(-U, dtype=np.float64)
            # exp_exp_neg_U = np.exp(exp_neg_U, dtype=np.float64)
            # M = -np.log(exp_exp_neg_U - 1.0)
            M = -np.log(np.exp(np.exp(-U)) - 1)

            return nn.Parameter(torch.tensor(M, dtype=torch.float))        

    @staticmethod
    def update_queue(tensor, data):
        '''
        for tensors of dimensions (batch_size, num_timesteps, num_neurons)
        '''
        # nice way
        # tensor = torch.cat((tensor[:, -1:, :], tensor[:, :-1, :]), dim=1) # shif to the right (Alberto)
        tensor = torch.roll(tensor, shifts=1, dims=1)  # Shift right by 1 (AI suggested, need to benchmark)

        tensor[:, 0, ...] = data

        return tensor

    def forward(self, input):
    
        all_o_mems = []
        all_o_spikes = []    

        mems, spikes, queued_spikes = self.init_state()

        self.spike_count = 0.0
        
        for step in range(self.num_simulation_steps):

            #prev_spikes = input[:, step, :].view(self.batch_size, -1)
            prev_spikes = input[:, step, :]

            for layer, key in zip(self.layers, spikes.keys()):

                if layer.pre_delays:
                    queued_spikes[key] = self.update_queue(queued_spikes[key], prev_spikes)
                    #prev_spikes = queued_spikes[key][:, layer.delays, :].transpose(1, 2).clone().detach() # is transpose necessary?
                    if isinstance(layer, Conv3DSNNLayer):
                        prev_spikes = queued_spikes[key].transpose(1, 2)
                    else:
                        prev_spikes = queued_spikes[key][:, layer.delays, :].transpose(1, 2)

                mems[key], spikes[key] = layer(prev_spikes, mems[key], spikes[key]) 
                prev_spikes = spikes[key]

                self.spike_count += prev_spikes.sum().item()
            
            # remove spikes  of last layer from spike count
            self.spike_count -= prev_spikes.sum().item()

            # store results
            #self.update_logger(input[:, step, :].view(self.batch_size, -1), mems, spikes, step)
            self.update_logger(input[:, step, ...], mems, spikes, step)

            # append results to activation list
            all_o_mems.append(mems[key])
            all_o_spikes.append(spikes[key])

        return all_o_mems, all_o_spikes
    

    def forward_live(self, input):
    
        mems = self.mems
        spikes = self.spikes
        queued_spikes = self.queued_spikes

        self.spike_count = 0.0
        
        prev_spikes = input.view(self.batch_size, -1)

        for layer, key in zip(self.layers, spikes.keys()):

            if layer.pre_delays:
                queued_spikes[key] = self.update_queue(queued_spikes[key], prev_spikes)
                #prev_spikes = queued_spikes[key][:, layer.delays, :].transpose(1, 2).clone().detach() # is transpose necessary?
                prev_spikes = queued_spikes[key][:, layer.delays, :].transpose(1, 2)

            mems[key], spikes[key] = layer(prev_spikes, mems[key], spikes[key]) 
            prev_spikes = spikes[key]

            self.spike_count += prev_spikes.sum().item()
        
        # remove spikes  of last layer from spike count
        self.spike_count -= prev_spikes.sum().item()

        # store results
        #self.update_logger(input[:, step, :].view(self.batch_size, -1), mems, spikes, step)

        #return mems[key].detach().clone(), spikes[key].detach().clone()
        return mems, spikes


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