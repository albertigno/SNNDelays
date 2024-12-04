import torch
import torch.nn as nn
from snn_delays.snn import SNN
import numpy as np

class TH_SNN(SNN):
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
        extra param: mask: a mask for the input layer
        """

        # Pass arguments to the parent class
        super().__init__(
            dataset_dict=dataset_dict, 
            structure=structure,
            connection_type=connection_type,
            delay=delay,
            delay_type=delay_type,
            reset_to_zero=reset_to_zero,
            tau_m=tau_m,
            win=win,
            loss_fn=loss_fn,
            batch_size=batch_size,
            device=device,
            debug=debug
        )
        
        # Gather keyword arguments for reproducibility in loaded models
        self.kwargs = locals()
        self.set_th()
        self.set_layers()

    # TODO: Documentar y testar
    def set_th(self):
        """
        Function to define the trainable thresholds.
        """

        initial_th = 0.3 # all thresholds initialized in 0.3

        for i in range(self.num_layers):
            name = 'th_' + str(i + 1)
            setattr(self, name, nn.Parameter(initial_th*torch.ones(self.num_neurons_list[i])))
        setattr(self, 'th_o', nn.Parameter(initial_th*torch.ones(self.num_output)))

        ### set list of thresholds:
        self.th_h = [getattr(self, name) for name in
                        ['th_' + str(i + 1)
                            for i in range(self.num_layers)]]
        self.th_h.append(self.th_o)        

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

        return self.activation_function(mem, self.th_h[self.tau_idx-1])

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

        return self.activation_function(mem, self.th_h[self.tau_idx-1])
    
    def activation_function(self, mem, thresh):

        '''
        The activation function is defined here
        '''

        th_reset = self.th_reset

        # if mem.shape[-1] == 20:
        #     print(mem)
        #     print(thresh)
        #     print(mem-thresh)

        o_spike = self.act_fun(mem-thresh, self.surr_scale)
        mem = mem*(mem < th_reset)     

        return mem, o_spike