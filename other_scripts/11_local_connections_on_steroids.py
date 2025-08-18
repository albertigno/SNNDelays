import torch
import torch.nn as nn
import torch.nn.functional as F

class LocallyConnected2D(nn.Module):
    def __init__(self, input_size, filter_size, num_parallel, in_channels=2, linearize=False):
        """
        Args:
            input_size (int): Spatial dimension of input (2^n).
            filter_size (int): Size of filters (power of 2).
            num_parallel (int): Number of parallel projections (fanout filters).
            in_channels (int): Number of input channels (default=2).
            linearize (bool): If True, linearize output; else group by block (default=False).
        """
        super().__init__()
        assert input_size % filter_size == 0, "Filter size must divide input size evenly"
        
        self.input_size = input_size
        self.filter_size = filter_size
        self.num_parallel = num_parallel
        self.in_channels = in_channels
        self.linearize = linearize
        
        # Calculate number of blocks per dimension
        self.num_blocks = input_size // filter_size
        self.num_blocks_total = self.num_blocks * self.num_blocks
        
        # Initialize weights: separate filter for each block, channel, and parallel projection
        # Shape: (num_parallel, in_channels, num_blocks, num_blocks, filter_size, filter_size)
        self.weight = nn.Parameter(
            torch.Tensor(num_parallel, in_channels, self.num_blocks, self.num_blocks, filter_size, filter_size))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, input_size, input_size, in_channels)
        
        Returns:
            Output tensor either:
                - Linearized: (N, num_parallel * num_blocks * num_blocks * in_channels)
                - Grouped: (N, num_parallel, num_blocks, num_blocks, in_channels)
        """
        N, H, W, C = x.shape
        assert H == self.input_size and W == self.input_size, "Input spatial dimensions mismatch"
        assert C == self.in_channels, "Input channel dimension mismatch"
        
        # Rearrange to channels-first: (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Extract non-overlapping blocks
        # Output: (N, C, num_blocks, num_blocks, filter_size, filter_size)
        x_blocks = x.unfold(2, self.filter_size, self.filter_size).unfold(3, self.filter_size, self.filter_size)
        
        # Reshape for broadcasting: (N, 1, C, num_blocks, num_blocks, filter_size, filter_size)
        x_blocks = x_blocks.unsqueeze(1)
        
        # Element-wise multiplication and sum over spatial dimensions
        # Output: (N, num_parallel, C, num_blocks, num_blocks)
        out = torch.sum(x_blocks * self.weight, dim=(-1, -2))
        
        # Rearrange dimensions to group by block position
        # Output: (N, num_parallel, num_blocks, num_blocks, C)
        out = out.permute(0, 1, 3, 4, 2)
        
        if self.linearize:
            # Flatten all dimensions except batch
            out = out.reshape(N, -1)
        
        return out
    
input_size = 16  # 2^4
filter_size = 4   # Power of 2
num_parallel = 3  # 3 parallel projections
in_channels = 2   # Input channels

# Create layer
layer = LocallyConnected2D(input_size, filter_size, num_parallel, in_channels, linearize=False)

# Input tensor: (batch=8, 16x16 spatial, 2 channels)
x = torch.randn(8, input_size, input_size, in_channels)

# Forward pass
output = layer(x)
print(output.shape)  # Output shape: (8, 3, 4, 4, 2) [grouped by block]