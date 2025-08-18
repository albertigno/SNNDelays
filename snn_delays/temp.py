class Conv2DSNNLayer(AbstractSNNLayer):
    """
    Advances a single timestep of a 2D convolutional SNN layer.
    This layer does not have explicit delays.
    """
    def __init__(self, in_channels, out_channels, kernel_size, tau_m, batch_size, 
                 inf_th=False, device=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.tau_m = tau_m
        self.batch_size = batch_size
        self.device = device
        self.thresh = float('inf') if inf_th else 0.3

        #### hard-coded params
        stride = kernel_size # no overlapping
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