import torch
import torch.nn as nn
from snn_delays.snn import SNN
from torch.autograd import Function
import torch.nn.functional as F

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Custom Binarization Function (like the one you provided)
class Binarize(Function):
    @staticmethod
    def forward(ctx, input, bin_mode, inplace=False):
        ctx.inplace = inplace
        #ctx.save_for_backward(input, scale)
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        #scale = output.abs().mean() if allow_scale else 1
        scale = output.abs().mean()

        if bin_mode == 'binary':
            #return (output > 0.0).float().mul(scale) # {0, 1}
            return (output > 0.0).float().mul(scale).mul(0.3) # {0, 1} scaled to 1/3
        elif bin_mode == 'relu':
            return output.clamp(0, 1).mul(scale) # {0, 1} relu-like
        elif bin_mode == 'sign':
            return output.sign().mul(scale) # {-1, 1}
        else: # not deterministic sign
            return output.div(scale).add_(1).div_(2).add_(torch.rand(output.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1).mul(scale)

    # with trainable scale
    # @staticmethod
    # def backward(ctx, grad_output):
    #     input, scale = ctx.saved_tensors
    #     # Straight-Through Estimator (STE) for binary values
    #     grad_input = grad_output.clone()
    #     grad_scale = None
    #     if scale.requires_grad:
    #         # Gradient for scale (simplified)
    #         grad_scale = grad_output.mul(input.sign()).mean()
    #     return grad_input, grad_scale, None, None

    ## without trainable scale
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through Estimator (STE)
        grad_input = grad_output
        return grad_input, None, None, None

class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bin_mode, bias):
        super(BinaryLinear, self).__init__()
        self.bin_mode = bin_mode
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))  # Full-precision weights
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        # Apply binarization to weights using the Binarize function
        #binary_weight = Binarize.apply(self.weight, quant_mode, allow_scale)
        binary_weight = Binarize.apply(self.weight, self.bin_mode)
        return F.linear(x, binary_weight, self.bias)


class BinarySNN(SNN):
    """
    Spiking neural network (SNN) class.

    Common characteristic and methods for a spiking neural network with or
    without delays. It inherits from nn.Module.
    """

    def __init__(self, dataset_dict, structure=(256, 2),
                 connection_type='r', delay=None, delay_type='ho',
                 reset_to_zero=True, tau_m='normal', win=50, 
                 loss_fn='mem_sum', binary_mode='sign', bias = False,
                 batch_size=256, device='cuda', debug=False):
        """
        extra param: binary
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
        self.binary_mode = binary_mode  
        self.bias = bias

    def set_layers(self):
        """
        Function to set input, hidden and output layers as Linear layers. If the
        propagation mode include recurrence (self.connection_type = 'r'),
        additional layers (self.r_name) are created.
        """

        # Set bias
        num_first_layer = self.num_neurons_list[0]

        # if delays is None, len(self.delays) = 1

        setattr(self, 'f0_f1', BinaryLinear(self.num_input*len(self.delays_i),
                                    num_first_layer, self.binary_mode, self.bias))                         

        # Set linear layers dynamically for the l hidden layers
        for lay_name_1, lay_name_2, num_pre, num_pos in zip(self.layer_names[:-1],
         self.layer_names[1:], self.num_neurons_list[:-1], self.num_neurons_list[1:]):

            # This only if connection is recurrent
            if self.connection_type == 'r':
                name = lay_name_1 + '_' + lay_name_1

                setattr(self, name, BinaryLinear(
                    num_pre* len(self.delays_h), num_pre, self.binary_mode, self.bias))
                
                self.proj_names.append(name)       

            # Normal layer
            name = lay_name_1 + '_' + lay_name_2

            setattr(self, name, BinaryLinear(num_pre * len(self.delays_h),
                                    num_pos, self.binary_mode, self.bias))    
            
            self.proj_names.append(name)

        if self.connection_type == 'r':
            name = self.layer_names[-1] + '_' + self.layer_names[-1]

            setattr(self, name, BinaryLinear(
                self.num_neurons_list[-1]* len(self.delays_h), self.num_neurons_list[-1], self.binary_mode, self.bias))                                                
            
            self.proj_names.append(name)

        # output layer
        name = self.layer_names[-1]+'_o'

        setattr(self, name, BinaryLinear(self.num_neurons_list[-1] * len(self.delays_o),
                                    self.num_output, self.binary_mode, self.bias)) 

        self.proj_names.append(name)


class TH_BinarySNN(BinarySNN):

    """
    Spiking neural network (SNN) class.

    Common characteristic and methods for a spiking neural network with or
    without delays. It inherits from nn.Module.
    """

    def __init__(self, dataset_dict, structure=(256, 2),
                 connection_type='r', delay=None, binary_mode='sign', delay_type='ho',
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
            binary_mode=binary_mode,
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