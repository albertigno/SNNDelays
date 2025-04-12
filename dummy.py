
import torch
from torch.utils.data import Dataset
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional
import torch
import numpy as np

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



@dataclass
class MemoryCachedDataset:
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


class SyntheticDataset(Dataset):

    def __init__(self, seq_length=50, dataset_size=128, randomness=False):

        super(SyntheticDataset, self).__init__()

        self.seq_length = seq_length
        self.mem_length = int(0.1*seq_length)
        self.dataset_size = dataset_size
        self.randomness = randomness     

    def __getitem__(self, idx):
        """
        Get a sample of the dataset. If the sample index is higher than the
        number of samples in dataset, it returns an error and stop the
        execution

        :param idx: Index of the sample to be returned
        :return: A tuple with the original (sample) and the target (label)
        sequence
        """
        _x, _y = self.create_sample(
            self.seq_length, self.mem_length, idx, self.randomness)
        return _x, _y

    def __len__(self):
        """
        The number of samples in the dataset

        :return: Dataset size
        """
        return self.dataset_size


class CopyMemoryDataset(SyntheticDataset):

    ### V4: COPY TASK WITH multiple output neurons
    ### V5: all sequence is random numbers
    @staticmethod
    def create_sample(seq_length, mem_length, idx, rnd):

        # Set seed for repeated batches ir rnd=False
        if not rnd:
            torch.manual_seed(idx)

        max_noise = 0.2
        seq = max_noise*torch.rand([seq_length, 3], dtype=torch.float)
        #seq[:,0] = torch.randint(1, 10, (seq_length, 1)) / 10.0 # random numbers from 0.1 to 0.9

        seq[:,0] = torch.randint(1, 10, (seq_length,)) / 10.0 # random numbers from 0.1 to 0.9
        
        # the time at which the number to memorize appears
        start_time = torch.randint(high=seq_length//2, size=(1,)).item()
        
        # marker for the sequence to remember
        seq[start_time:start_time + mem_length, 1] = torch.ones([mem_length])

        label = torch.zeros(mem_length, 1)
        label[:,0] = seq[start_time:start_time + mem_length, 0].T.clone().detach()
        
        seq[seq_length-mem_length:, 2] = torch.ones([mem_length])
        
        label = label.expand(-1, mem_length).T

        return seq.clone().detach(), label.clone().detach()

    def get_train_attributes(self):
        """
        Function to get these three attributes which are necessary for a
        correct initialization of the SNNs: num_training samples, num_input,
        etc. All Dataset should have this, if possible.
        """
        train_attrs = {'num_input': 3,
                       'num_training_samples': self.dataset_size,
                       'num_output': self.mem_length}

        return train_attrs


class MultiAddtaskDataset(SyntheticDataset):

    ### TWO SETS
    @staticmethod
    def create_sample(seq_length, mem_length, idx, rnd):

        # Set seed for repeated batches ir rnd=False
        if not rnd:
            torch.manual_seed(idx)

        max_noise = 0.2
        seq = max_noise*torch.rand([seq_length, 3], dtype=torch.float)
        seq[:,0] = torch.randint(1, 10, (seq_length,)) / 10.0 # random numbers from 0.1 to 0.9

        # the time at which the number to memorize appears
        half_seq = int(0.8*seq_length/2) - mem_length
        end_seq = int(0.8*seq_length) - mem_length
        start_time_1 = torch.randint(high=half_seq, size=(1,)).item()
        start_time_2 = torch.randint(low=half_seq+mem_length, high=end_seq, size=(1,)).item()

        # marker for the two sequence to add
        seq[start_time_1:start_time_1 + mem_length, 1] = torch.ones([mem_length])
        seq[start_time_2:start_time_2 + mem_length, 1] = torch.ones([mem_length])

        # marker for the queue at the end of the task
        seq[seq_length-mem_length:, 2] = torch.ones([mem_length])
        
        # Sum all elements of 'labels' to create a new label, normalize so the max is 2 (as in add task)
        operand1 = seq[start_time_1:start_time_1 + mem_length, 0]
        operand2 = seq[start_time_2:start_time_2 + mem_length, 0]
        lbl = torch.sum(operand1 + operand2)/(0.9*mem_length)
        label = lbl.item() * torch.ones([mem_length, 1])

        return seq.clone().detach(), label.clone().detach()


if __name__=='__main__':

    total_time = 50
    batch_size = 128
    dataset_size = batch_size*3000

    train_dataset = CopyMemoryDataset(seq_length=total_time,
                                    dataset_size=dataset_size,
                                    randomness=True)
    test_dataset = CopyMemoryDataset(seq_length=total_time,
                                    dataset_size=batch_size,
                                    randomness=False)    

    train_dataset = MemoryCachedDataset(train_dataset, device="cuda:0")
    test_dataset = MemoryCachedDataset(test_dataset, device="cuda:0")

if __name__ == '__main__':
    import time
    import torch.utils.data as data

    total_time = 50
    batch_size = 128
    dataset_size = batch_size * 3000
    num_workers = 4  # Adjust based on your CPU cores
    prefetch_factor = 2  # Number of batches loaded in advance

    # Create datasets
    train_dataset = CopyMemoryDataset(seq_length=total_time,
                                    dataset_size=dataset_size,
                                    randomness=True)
    test_dataset = CopyMemoryDataset(seq_length=total_time,
                                    dataset_size=batch_size,
                                    randomness=False)

    # Cache to GPU
    print("Caching training data to GPU...")
    start_cache = time.time()
    train_dataset = MemoryCachedDataset(train_dataset, device="cuda:0")
    test_dataset = MemoryCachedDataset(test_dataset, device="cuda:0")
    print(f"Caching completed in {time.time() - start_cache:.2f} seconds")

    # Create DataLoaders with optimized settings
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Already on GPU
        prefetch_factor=prefetch_factor,
        persistent_workers=True  # Maintains worker pool
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )

    # Benchmark function
    def benchmark(dataloader, name, num_batches=100):
        print(f"\nBenchmarking {name}...")
        start = time.time()
        batch_times = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            batch_start = time.time()
            data, target = batch
            # Simulate training step
            torch.cuda.synchronize()  # Wait for GPU ops to complete
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
        total_time = time.time() - start
        avg_batch_time = sum(batch_times) / len(batch_times) * 1000
        print(f"Processed {len(batch_times)} batches")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average batch time: {avg_batch_time:.2f} ms")
        print(f"Throughput: {len(batch_times)/total_time:.2f} batches/sec")

    # Run benchmarks
    benchmark(train_loader, "Training Data")
    benchmark(test_loader, "Test Data")

    # Full epoch timing
    print("\nTiming full training epoch...")
    start_epoch = time.time()
    for i, (data, target) in enumerate(train_loader):
        if i % 100 == 0:
            print(f"Processed batch {i}/{len(train_loader)}")
    epoch_time = time.time() - start_epoch
    print(f"Full epoch time: {epoch_time:.2f} seconds")
    print(f"Average batch time: {epoch_time/len(train_loader)*1000:.2f} ms")