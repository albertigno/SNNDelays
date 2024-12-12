import torch
import torch.nn as nn
from snn_delays.snn import SNN
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

class DelayMaskingLayer(nn.Module):
    def __init__(self, in_features, out_features, n_delays, top_k=3):
        super(DelayMaskingLayer, self).__init__()

        self.n_input = in_features // n_delays
        self.n_delays = n_delays
        self.n_output = out_features
        self.top_k = top_k

        self.linear = nn.Linear(in_features, out_features, bias=False)

        self._masked_projection = None

    def forward(self, x):
        # Reshape the projection matrix to (n_output, n_input, n_delays)
        projection_reshaped = self.linear.weight.view(self.n_output, self.n_input, self.n_delays)
        
        # Compute the mask dynamically based on absolute values
        abs_projection = projection_reshaped.abs()
        top_k_indices = abs_projection.topk(k=self.top_k, dim=2).indices
        
        mask = torch.zeros_like(projection_reshaped, dtype=torch.bool)
        mask.scatter_(2, top_k_indices, True)
        
        # Apply the mask to the projection matrix
        masked_projection = projection_reshaped * mask

        # Reshape back to original shape
        masked_projection = masked_projection.view(self.n_output, self.n_input * self.n_delays)

        self._masked_projection = masked_projection

        # Apply projection to input
        return nn.functional.linear(x, masked_projection)

class FixedMaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask):
        super(FixedMaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Define the linear layer with PyTorch's default initialization
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
        # Register the mask
        self.register_buffer('mask', mask)
        
        # Apply the mask to the weights after initialization
        with torch.no_grad():
            ##### uncomment to enforce negative weights (self inhibition experiment)
            # data = self.linear.weight.data
            # self.linear.weight.data = -1.0*data*(data>0) + data*(data<0)
            self.linear.weight *= self.mask
    
    def forward(self, x):
        # Apply the mask during the forward pass to enforce connectivity

        masked_weight = self.linear.weight * self.mask
        return nn.functional.linear(x, masked_weight)



class FixedDelayMaskingLayer(nn.Module):
    def __init__(self, in_features, out_features, n_delays, top_k=3):
        super(FixedDelayMaskingLayer, self).__init__()
        self.n_input = in_features // n_delays
        self.n_delays = n_delays
        self.n_output = out_features
        self.top_k = top_k

        # Initialize projection weights
        self.linear = nn.Linear(in_features, out_features, bias=False)

        # Precompute mask and register as buffer
        with torch.no_grad():
            projection_reshaped = self.linear.weight.view(self.n_output, self.n_input, self.n_delays)
            abs_projection = projection_reshaped.abs()
            top_k_indices = abs_projection.topk(k=self.top_k, dim=2).indices
            
            mask = torch.zeros_like(projection_reshaped, dtype=torch.bool)
            mask.scatter_(2, top_k_indices, True)

        self.register_buffer("mask", mask)

    def forward(self, x):
        # Apply the precomputed mask to the projection matrix
        projection_reshaped = self.linear.weight.view(self.n_output, self.n_input, self.n_delays)
        masked_projection = projection_reshaped * self.mask

        # Reshape back to original shape
        masked_projection = masked_projection.view(self.n_output, self.n_input * self.n_delays)

        # Apply projection to input
        return nn.functional.linear(x, masked_projection)


class P_DelaySNN(SNN):
    """
    Spiking neural network (SNN) class.

    Common characteristic and methods for a spiking neural network with or
    without delays. It inherits from nn.Module.
    """

    def __init__(self, dataset_dict, structure=(256, 2),
                 connection_type='r', delay=None, n_pruned_delays=1,
                 delay_mask = 'top_k',
                 delay_type='h',
                 reset_to_zero=True, tau_m='normal', win=50,
                 loss_fn='mem_sum',
                 batch_size=256, device='cuda', debug=False):
        """
        This only works for 'h' type delays
        extra params: 
            n_pruned_delays
            delay_mask: 'top_k' or 'random'.
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

        self.n_pruned_delays = n_pruned_delays
        self.delay_mask = delay_mask
        
        self.set_layers()

    def set_layers(self):
        """
        Function to set input, hidden and output layers as Linear layers. If the
        propagation mode include recurrence (self.connection_type = 'r'),
        additional layers (self.r_name) are created.
        """

        # Set bias
        bias = False

        num_first_layer = self.num_neurons_list[0]

        # if delays is None, len(self.delays) = 1

        setattr(self, 'f0_f1', nn.Linear(self.num_input*len(self.delays_i),
                                    num_first_layer, bias=False))            

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
            name = lay_name_1 + '_' + lay_name_2
            # setattr(self, name, nn.Linear(num_pre * len(self.delays_h),
            #                             num_pos, bias=bias))     
            if self.delay_mask == 'top_k':
                setattr(self, name, DelayMaskingLayer(num_pre * len(self.delays_h),
                                            num_pos, len(self.delays_h), self.n_pruned_delays)) 
            elif self.delay_mask == 'random':
                mask = torch.rand(num_pos, num_pre * len(self.delays_h)) < (self.n_pruned_delays / len(self.delays_h))
                mask.to(self.device)
                setattr(self, name, FixedMaskedLinear(num_pre * len(self.delays_h),
                                            num_pos, mask))
            # setattr(self, name, FixedDelayMaskingLayer(num_pre * len(self.delays_h),
            #                             num_pos, len(self.delays_h), self.n_pruned_delays))

            self.proj_names.append(name)

        if self.connection_type == 'r':
            name = self.layer_names[-1] + '_' + self.layer_names[-1]
            setattr(self, name, nn.Linear(
                self.num_neurons_list[-1]* len(self.delays_h), self.num_neurons_list[-1], bias=bias))                              
            self.proj_names.append(name)

        # output layer
        name = self.layer_names[-1]+'_o'
        setattr(self, name, nn.Linear(self.num_neurons_list[-1] * len(self.delays_o),
                                    self.num_output, bias=False))
            
        self.proj_names.append(name)