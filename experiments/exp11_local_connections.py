import torch
import numpy as np
import matplotlib.pyplot as plt

def create_local_connection_mask(input_size, kernel_size, stride, channels):
    """
    Create a mask for local connections.
    
    Parameters:
        input_size (int): The spatial size of the input (assumes square, e.g., 100 for 100x100).
        kernel_size (int): The size of the local receptive field (e.g., 5 for 5x5).
        stride (int): The stride for moving the local receptive field (e.g., 5 for non-overlapping).
        channels (int): The number of input channels (e.g., 2 for 2-channel images).
    
    Returns:
        torch.Tensor: A binary mask with shape (output_neurons, input_neurons).
    """
    output_size = (input_size - kernel_size) // stride + 1
    mask = torch.zeros((output_size**2, input_size**2 * channels), dtype=torch.float32)

    for oy in range(output_size):
        for ox in range(output_size):
            output_idx = oy * output_size + ox
            
            # Top-left corner of the kernel in the input
            start_y = oy * stride
            start_x = ox * stride
            
            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    for c in range(channels):
                        input_y = start_y + ky
                        input_x = start_x + kx
                        input_idx = c * input_size**2 + input_y * input_size + input_x
                        mask[output_idx, input_idx] = 1

    return mask

# Parameters
input_size = 100
kernel_size = 5
stride = 5
channels = 2

# Create the mask
mask = create_local_connection_mask(input_size, kernel_size, stride, channels)

print(f"Mask shape: {mask.shape}")  # Should be (400, 20000)

plt.figure(figsize=(10,10))
plt.imshow(mask)
plt.show()