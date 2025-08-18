import torch
import torch.nn as nn
import torch.nn.functional as F

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


class TDBatchNorm1d(nn.Module):
    """
    Threshold-Dependent Batch Normalization (tdBN) for dense (fully connected) layers,
    using nn.BatchNorm1d.
    This module normalizes pre-activations to have a mean of 0 and a variance of (gamma * V_th)^2.
    """
    def __init__(self, num_features: int):
        super().__init__()

        # hard coded values (initial weight or gamma is 1.0)
        eps = 1e-5
        momentum = 0.1

        # BatchNorm1d expects input of shape (N, C) or (N, C, L)
        # For dense layers, our input will be (batch_size, num_features), fitting (N, C).
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=True)
        self.bn.bias = None # We do not need a bias term as we will scale the output with gamma * V_th.

    def forward(self, x: torch.Tensor, v_th: torch.Tensor) -> torch.Tensor:
        """
        Applies Threshold-Dependent Batch Normalization to the input pre-activations.

        Args:
            x (torch.Tensor): The input tensor of pre-activations.
                              Expected shape: (batch_size, num_features).
            v_th (torch.Tensor): The neuron firing threshold. This can be:
                                 - A scalar: Applied universally to all features.
                                 - A tensor of shape (num_features,): A specific threshold for each feature/neuron.
                                 PyTorch's broadcasting rules will handle the multiplication.

        Returns:
            torch.Tensor: The normalized pre-activations, targeting N(0, (gamma * V_th)^2).
                          Shape will be the same as input `x`: (batch_size, num_features).
        """

        # self.bn(x) is gamma * N(0,1).
        # This makes the distribution of `scaled_x` target N(0, (gamma * V_th)^2).
        scaled_x = self.bn(x) * v_th

        return scaled_x

class AbstractSNNLayer(nn.Module):

    act_fun = ActFunFastSigmoid.apply
    output_shape = None # to be defined when child classes are instantiated
    pre_delays = False # default false, when delayed child classes are instantiated this turns to True

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

    def __init__(self, num_in, num_out, tau_m, batch_size, inf_th=False, device=None, num_fanin_multifeedforward=None):
        super().__init__()

        self.num_in = num_in
        self.num_out = num_out
        self.pre_delays = False

        self.multif= num_fanin_multifeedforward
        self.linear = nn.Linear(num_in * num_fanin_multifeedforward, num_out, bias=False)

        self.tau_m = tau_m
        self.batch_size = batch_size
        self.device = device
        self.thresh = float('inf') if inf_th else 0.3
        self.multi_proj = num_fanin_multifeedforward

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

        if self.pre_delays and self.pruned_delays is not None:
            with torch.no_grad():
                scale_factor = torch.sqrt(self.max_d / self.pruned_delays).item()
                self.linear.weight *= scale_factor


    def update_mem(self, i_spike, o_spike, mem, thresh):

        # Set alpha value to membrane potential decay
        alpha = self.alpha_sigmoid(self.tau_m).to(self.device)

        # weighed output from previous spikes
        w_out = self.linear(i_spike)

        # batch normalization
        # if self.apply_bn:
        #     w_out = self.tdbn(w_out, thresh)

        # Calculate the new membrane potential and output spike
        mem = mem * alpha * (1 - o_spike) + w_out

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

## AI-generated 
class Conv2DSNNLayer(AbstractSNNLayer):
    """
    Advances a single timestep of a 2D convolutional SNN layer.
    This layer does not have explicit delays.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, tau_m, batch_size, 
                 inf_th=False, avg_pooling=False, device=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.tau_m = tau_m
        self.batch_size = batch_size
        self.device = device
        self.thresh = float('inf') if inf_th else 0.3
        self.avg_pooling = avg_pooling

        #### hard-coded params
        # stride = kernel_size # no overlapping
        padding = 0
        dilation = 1
        groups = 1 # all channels are convolved with the same filter

        # stride=1, padding=0, dilation=1, groups=1 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, 
                              groups=groups, bias=False)

    def update_mem(self, i_spike, o_spike, mem, thresh):
        """
        Updates the membrane potential and generates spikes for a Conv2D SNN layer.

        Args:
            i_spike (torch.Tensor): Input spikes from the previous layer. Expected shape (batch_size, channels, height, width).
            o_spike (torch.Tensor): Previous output spikes of this layer. Expected shape (batch_size, out_channels, out_height, out_width).
            mem (torch.Tensor): Previous membrane potential of this layer. Expected shape (batch_size, out_channels, out_height, out_width).
            thresh (float): Threshold for spiking.

        Returns:
            tuple: A tuple containing:
                - mem (torch.Tensor): Updated membrane potential.
                - o_spike (torch.Tensor): New output spikes.
        """
        # Set alpha value to membrane potential decay
        alpha = self.alpha_sigmoid(self.tau_m).to(self.device)

        # Apply convolutional operation to input spikes
        conv_output = self.conv(i_spike)

        if self.avg_pooling:
            conv_output = F.avg_pool2d(conv_output, kernel_size=2)

        # Calculate the new membrane potential
        mem = mem * alpha * (1 - o_spike) + conv_output

        return self.activation_function(mem, thresh)

    def forward(self, prev_spikes, own_mems, own_spikes):
        '''
        Returns the mem and spike state for Conv2DSNNLayer.
        The input prev_spikes are expected to be in (batch_size, channels, height, width) format.
        '''
        mem_out, spikes_out = self.update_mem(
            prev_spikes, own_spikes, own_mems, self.thresh)

        return mem_out, spikes_out


# AI-Generated
class Conv3DSNNLayer(AbstractSNNLayer):
    """
    Advances a single timestep of a 3D "static" convolutional SNN layer,
    processing input with a temporal dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, temporal_kernel_size, 
                 tau_m, batch_size, inf_th=False, avg_pooling=False, device=None):
        super().__init__()

        self.pre_delays = True  # The temporal dimension is treated as a delay

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size # This now refers to the spatial kernel (H, W)
        self.tau_m = tau_m
        self.batch_size = batch_size
        self.device = device
        self.thresh = float('inf') if inf_th else 0.3
        self.avg_pooling = avg_pooling

        # New temporal dimensions
        self.temporal_kernel_size = temporal_kernel_size # The 'D' dimension of Conv3d kernel

        #### hard-coded params for SPATIAL dimensions (H, W)
        # non-overlapping spatial patches
        spatial_stride = kernel_size # no overlapping
        spatial_padding = 0
        spatial_dilation = 1 # Assuming default for spatial
        spatial_groups = 1   # Assuming default for spatial
        
        temporal_stride = 1 # No temporal stride, as we want to process all time channels
        temporal_padding = 0 
        temporal_dilation = 1

        self.conv = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(self.temporal_kernel_size, self.kernel_size, self.kernel_size), # (D_kernel, H_kernel, W_kernel)
            stride=(temporal_stride, spatial_stride, spatial_stride), # (D_stride, H_stride, W_stride)
            padding=(temporal_padding, spatial_padding, spatial_padding), # (D_padding, H_padding, W_padding)
            dilation=(temporal_dilation, spatial_dilation, spatial_dilation), # (D_dilation, H_dilation, W_dilation)
            groups=spatial_groups,
            bias=False
        )

    def update_mem(self, i_spike, o_spike, mem, thresh):
        """
        Updates the membrane potential and generates spikes for a 3D Conv SNN layer.

        Args:
            i_spike (torch.Tensor): Input spikes buffer from the previous layer.
                                    Expected shape (batch_size, time_channels, channels, height, width).
            o_spike (torch.Tensor): Previous output spikes of this layer.
                                    Expected shape (batch_size, out_channels, out_height, out_width).
            mem (torch.Tensor): Previous membrane potential of this layer.
                                Expected shape (batch_size, out_channels, out_height, out_width).
            thresh (float): Threshold for spiking.

        Returns:
            tuple: A tuple containing:
                - mem (torch.Tensor): Updated membrane potential.
                - o_spike (torch.Tensor): New output spikes.
        """
        # Set alpha value to membrane potential decay
        alpha = self.alpha_sigmoid(self.tau_m).to(self.device)

        # Apply 3D convolutional operation to input spikes
        # Input to conv must be (N, C_in, D_in, H_in, W_in)
        # Your i_spike is (batch_size, time_channels, channels, height, width)
        # So we need to permute dimensions: (batch_size, channels, time_channels, height, width)
        
        # Original shape (B, D, C, H, W) -> Permute to (B, C, D, H, W) for Conv3d
        #i_spike_permuted = i_spike.permute(0, 2, 1, 3, 4) # (batch, channels, time_channels, height, width)
        #conv_output = self.conv(i_spike_permuted)

        conv_output = self.conv(i_spike)

        # After Conv3d with temporal_kernel_size=time_channels and temporal_stride=1, temporal_padding=0,
        # the output `conv_output` will have a temporal dimension of 1.
        # Shape of conv_output: (batch_size, out_channels, 1, out_height, out_width)
        # We need to squeeze this singleton temporal dimension.
        conv_output_2d = conv_output.squeeze(2) # Shape: (batch_size, out_channels, out_height, out_width)

        if self.avg_pooling:
            conv_output_2d = F.avg_pool2d(conv_output_2d, kernel_size=2)

        # Calculate the new membrane potential
        mem = mem * alpha * (1 - o_spike) + conv_output_2d

        return self.activation_function(mem, thresh)

    def forward(self, prev_spikes_buffer, own_mems, own_spikes):
        '''
        Returns the mem and spike state for Conv2DSNNLayer.
        The input prev_spikes_buffer is expected to be in (batch_size, time_channels, channels, height, width) format.
        '''
        mem_out, spikes_out = self.update_mem(
            prev_spikes_buffer, own_spikes, own_mems, self.thresh)

        return mem_out, spikes_out


class FlattenSNNLayer(AbstractSNNLayer):
    """
    A layer to flatten the spatial dimensions of spikes and membrane potentials
    for transition from convolutional to dense layers.
    It does not have SNN dynamics (tau_m, threshold, activation).
    """
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        # No tau_m, device, thresh as it's purely a reshaping layer
        # Override update_mem and activation_function as they are not used
        self.update_mem = self._flatten_update_mem
        self.activation_function = self._dummy_activation

    # Dummy activation function - not used, but AbstractSNNLayer expects it
    def _dummy_activation(self, mem, thresh):
        return mem, (mem > thresh).float() # Just to satisfy the signature if needed

    def _flatten_update_mem(self, i_spike, o_spike, mem, thresh):
        """
        This layer just passes through the input and flattens it.
        The o_spike and mem are assumed to be already flattened if they were
        passed through this layer previously.
        """
        # i_spike is the previous layer's output (spikes_out), so it's what we need to flatten
        flattened_i_spike = i_spike.reshape(self.batch_size, -1)
        
        return flattened_i_spike, flattened_i_spike

    def forward(self, prev_spikes, own_mems, own_spikes):
        '''
        Flattens the input spikes and membrane potentials.
        '''
        
        flattened_spikes = prev_spikes.reshape(self.batch_size, -1)

        return flattened_spikes, flattened_spikes 