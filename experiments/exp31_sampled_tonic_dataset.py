import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from collections import defaultdict
import pickle
from snn_delays.config import DATASET_PATH
from snn_delays.utils.train_utils import get_device
import os
from tonic import MemoryCachedDataset

device = get_device()

class SubDataset(Dataset):
    def __init__(self, base_dataset, samples_per_class, target_classes, save_path='indexes'):

        save_indices_path= os.path.join(DATASET_PATH, save_path)
        self.base_dataset = base_dataset
        self.samples_per_class = samples_per_class
        self.target_classes = target_classes
        self.num_classes = len(target_classes)

        if os.path.exists(save_indices_path):
            with open(save_indices_path, 'rb') as f:
                indices = pickle.load(f)
        
        else:
            class_counts = defaultdict(int)
            indices = []
            
            for i, (_, label) in enumerate(base_dataset):
                class_idx = np.argmax(label)
                if class_idx in target_classes and class_counts[class_idx] < samples_per_class:
                    indices.append(i)
                    class_counts[class_idx] += 1
            with open(save_indices_path, 'wb') as f:
                pickle.dump(indices, f)
    
        self.filtered_dataset = Subset(base_dataset, indices)

    def __len__(self):
        return len(self.filtered_dataset)
    
    def __getitem__(self, idx):
        if idx >= len(self.filtered_dataset):
            raise IndexError("Index out of range for SubDataset")
        
        img, label = self.filtered_dataset[idx]
        img = torch.tensor(img, dtype=torch.float).to(device, non_blocking=True)
        label = torch.tensor(label, dtype=torch.long).to(device, non_blocking=True)
        return img, label
    
from snn_delays.snn import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils import train, get_device
from snn_delays.utils.test_behavior import tb_save_max_last_acc
import torch

device = get_device()
torch.manual_seed(10)

dataset = 'ssc'
total_time = 50
batch_size = 256

# DATASET
DL = DatasetLoader(dataset=dataset,
                   caching='memory',
                   num_workers=0,
                   batch_size=batch_size,
                   total_time=total_time,
                   crop_to=1e6)

_, __, dataset_dict = DL.get_dataloaders()

classes = 20
samp_per_class = 100

target_classes = [x for x in range(classes)]
test_dataset = DL._dataset.test_dataset
train_dataset = DL._dataset.train_dataset

sub_train_dataset = SubDataset(train_dataset, samp_per_class, target_classes, f'{dataset}_s{samp_per_class}_c{classes}_train')
sub_test_dataset = SubDataset(test_dataset, samp_per_class, target_classes, f'{dataset}_s{samp_per_class}_c{classes}_test')

train_dataset = MemoryCachedDataset(sub_train_dataset, device=device)
test_dataset = MemoryCachedDataset(sub_test_dataset, device=device)

train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False,
                            #pin_memory=True,
                            num_workers=0)

test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False,
                            #pin_memory=True,
                            num_workers=0)

for x, y in train_loader:
    print(x.shape)

ckpt_dir = 'default' 
dataset_dict['num_training_samples'] = classes*samp_per_class
dataset_dict['time_ms'] = 50

snn = SNN(dataset_dict=dataset_dict, structure=(64, 2), connection_type='f',
    delay=(48,16), delay_type='h', tau_m = 'normal',
    win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
    debug=False)

snn.set_network()

snn.to(device)

num_epochs = 100

train(snn, train_loader, test_loader, 1e-3, num_epochs, dropout=0.0, 
    test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(100, 0.95), test_every=1, streamlit=True)