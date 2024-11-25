import torch
import torch.nn as nn
from snn_delays.snn import SNN

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask):
        super(MaskedLinear, self).__init__()
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


class Masked_SNN(SNN):
    """
    Spiking neural network (SNN) class.

    Common characteristic and methods for a spiking neural network with or
    without delays. It inherits from nn.Module.
    """

    def __init__(self, dataset_dict, structure=(256, 2),
                 connection_type='r', delay=None, mask=None, delay_type='ho',
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
        
        if mask is not None:
            self.register_buffer('mask', mask) # this creates a self.mask = mask
        else:
            self.mask = None        
        
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

        if self.mask is not None:
            setattr(self, 'f0_f1', MaskedLinear(self.num_input*len(self.delays_i),
                                        num_first_layer, mask=self.mask))               
        else:
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
            setattr(self, name, nn.Linear(num_pre * len(self.delays_h),
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
                                    self.num_output, bias=False))
            
        self.proj_names.append(name)

 