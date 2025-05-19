from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional
import torch
import numpy as np
import time
from torch.utils.data import Dataset

'''
alberto: this is a modification of tonic's MemoryCachedDataset to handle numpy-format samples when caching device=gpu
'''

# @dataclass
# class MemoryCachedDataset:
#     """MemoryCachedDataset caches the samples to memory to substantially improve data loading
#     speeds. However you have to keep a close eye on memory consumption while loading your samples,
#     which can increase rapidly when converting events to rasters/frames. If your transformed
#     dataset doesn't fit into memory, yet you still want to cache samples to speed up training,
#     consider using `DiskCachedDataset` instead.

#     Parameters:
#         dataset:
#             Dataset to be cached to memory.
#         device:
#             Device to cache to. This is preferably a torch device. Will cache to CPU memory if None (default).
#         transform:
#             Transforms to be applied on the data
#         target_transform:
#             Transforms to be applied on the label/targets
#         transforms:
#             A callable of transforms that is applied to both data and labels at the same time.
#     """

#     dataset: Iterable
#     device: Optional[str] = None
#     transform: Optional[Callable] = None
#     target_transform: Optional[Callable] = None
#     transforms: Optional[Callable] = None
#     samples_dict: dict = field(init=False, default_factory=dict)

#     def __getitem__(self, index):
#         try:
#             data, targets = self.samples_dict[index]
#         except KeyError as _:
#             data, targets = self.dataset[index]
#             if self.device is not None:
#                 data = self.to_device(data)
#                 targets = self.to_device(targets)                
#             self.samples_dict[index] = (data, targets)

#         if self.transform is not None:
#             data = self.transform(data)
#         if self.target_transform is not None:
#             targets = self.target_transform(targets)
#         if self.transforms is not None:
#             data, targets = self.transforms(data, targets)
#         return data, targets

#     def __len__(self):
#         return len(self.dataset)


#     def to_device(self, data):
#         """
#         Converts data to the specified device, handling both NumPy arrays and PyTorch tensors.
        
#         Args:
#             data: Input data (NumPy array or PyTorch tensor)
        
#         Returns:
#             Device-moved tensor
#         """
#         # If it's a NumPy array, convert to PyTorch tensor first
#         if isinstance(data, np.ndarray):
#             data = torch.from_numpy(data)
        
#         # If it's already a PyTorch tensor, move to device
#         if isinstance(data, torch.Tensor):
#             return data.to(self.device)
        
#         # For other types, raise an error or handle as needed
#         raise TypeError(f"Unsupported data type: {type(data)}")
    

# AI-version
@dataclass
class GPUMemoryCachedDataset:
    """Highly optimized GPU-cached dataset for consistent-shaped samples.
    
    Features:
    - Single contiguous GPU memory allocation
    - Batched host-to-device transfer
    - Pinned memory for fastest transfers
    - Pre-computed transforms when possible
    """

    dataset: Dataset
    device: Optional[str] = None
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None
    transforms: Optional[Callable] = None
    _data_tensor: torch.Tensor = None
    _target_tensor: torch.Tensor = None

    def __post_init__(self):
        self._initialize_cache()

    def _initialize_cache(self):
        if self._data_tensor is not None:  # Already initialized
            return

        start_time = time.time()
        print(f"Initializing GPU cache for {len(self.dataset)} samples...")

        # Get first sample to determine shapes and dtypes
        sample_data, sample_target = self.dataset[0]
        
        # Convert to tensor if numpy array
        if isinstance(sample_data, np.ndarray):
            sample_data = torch.from_numpy(sample_data)
        if isinstance(sample_target, np.ndarray):
            sample_target = torch.from_numpy(sample_target)

        # Create pinned memory buffers
        data_shape = (len(self.dataset), *sample_data.shape)
        target_shape = (len(self.dataset), *sample_target.shape)
        
        cpu_data = torch.empty(data_shape, dtype=sample_data.dtype, pin_memory=True)
        cpu_target = torch.empty(target_shape, dtype=sample_target.dtype, pin_memory=True)

        # Batch load into pinned memory
        for i in range(len(self.dataset)):
            data, target = self.dataset[i]
            
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target)
                
            cpu_data[i] = data
            cpu_target[i] = target

        # Single transfer to GPU
        self._data_tensor = cpu_data.to(self.device, non_blocking=True)
        self._target_tensor = cpu_target.to(self.device, non_blocking=True)

        print(f"Cache initialized in {time.time() - start_time:.2f} seconds")

    def __getitem__(self, index):
        data = self._data_tensor[index]
        target = self._target_tensor[index]

        # Apply transforms if needed
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            data, target = self.transforms(data, target)

        return data, target

    def __len__(self):
        return len(self.dataset)