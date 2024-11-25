import torch
import torch.nn as nn
from snn_delays.snn import SNN
from torch.autograd import Function
import torch.nn.functional as F

# Custom Binarization Function (like the one you provided)
class Binarize(Function):
    @staticmethod
    def forward(ctx, input, bin_mode='sign', inplace=False):
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
            #return output.div(scale).sign().mul(scale) # {-1, 1}
            #return output.div(scale).clamp(0, 1).mul(scale) # {0, 1} relu-like
            #return (output > 0.0).float().mul(scale) # {0, 1}
            return (output > 0.0).float().mul(scale).mul(0.3) # {0, 1} halfscale
        elif bin_mode == 'sign':
            return output.div(scale).sign().mul(scale)
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
    def __init__(self, in_features, out_features, bin_mode ='sign', bias=False):
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
                 connection_type='r', delay=None, binary_mode='sign', delay_type='ho',
                 reset_to_zero=True, tau_m='normal', win=50,
                 loss_fn='mem_sum',
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

        setattr(self, 'f0_f1', BinaryLinear(self.num_input*len(self.delays_i),
                                    num_first_layer, self.binary_mode))                         

        # Set linear layers dynamically for the l hidden layers
        for lay_name_1, lay_name_2, num_pre, num_pos in zip(self.layer_names[:-1],
         self.layer_names[1:], self.num_neurons_list[:-1], self.num_neurons_list[1:]):

            # This only if connection is recurrent
            if self.connection_type == 'r':
                name = lay_name_1 + '_' + lay_name_1

                setattr(self, name, BinaryLinear(
                    num_pre* len(self.delays_h), num_pre, self.binary_mode))
                
                self.proj_names.append(name)       

            # Normal layer
            name = lay_name_1 + '_' + lay_name_2

            setattr(self, name, BinaryLinear(num_pre * len(self.delays_h),
                                    num_pos, self.binary_mode))    
            
            self.proj_names.append(name)

        if self.connection_type == 'r':
            name = self.layer_names[-1] + '_' + self.layer_names[-1]

            setattr(self, name, BinaryLinear(
                self.num_neurons_list[-1]* len(self.delays_h), self.num_neurons_list[-1], self.binary_mode))                                                
            
            self.proj_names.append(name)

        # output layer
        name = self.layer_names[-1]+'_o'

        setattr(self, name, BinaryLinear(self.num_neurons_list[-1] * len(self.delays_o),
                                    self.num_output, self.binary_mode)) 

        self.proj_names.append(name)