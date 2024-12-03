from snn_delays.datasets.custom_datasets import ConcatenatedDataset
import torch
import time
from snn_delays.snn import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils import train, get_device
from snn_delays.utils.test_behavior import tb_save_max_last_acc
from torch.utils.data import DataLoader
from tonic import MemoryCachedDataset

'''
stmnist dataset as in ablation study
'''

device = get_device()

# for reproducibility
torch.manual_seed(10)

dataset = 'nmnist'
#total_time = 25
total_time = 1
batch_size = 128

# DATASET
DL = DatasetLoader(dataset=dataset,
                  caching='memory',
                  num_workers=0,
                  batch_size=batch_size,
                  total_time=total_time,
                  sensor_size_to=64,
                  crop_to=3e6)

_, _, dataset_dict = DL.get_dataloaders()

target_classes = [1, 3, 8]
test_dataset = DL._dataset.test_dataset
train_dataset = DL._dataset.train_dataset

num_seq = 4

conc_test_dataset = ConcatenatedDataset(test_dataset, num_seq, target_classes)
conc_train_dataset = ConcatenatedDataset(train_dataset, num_seq, target_classes)

train_dataset = MemoryCachedDataset(conc_train_dataset)
test_dataset = MemoryCachedDataset(conc_test_dataset)

train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=0)

test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=0)

dataset_dict["num_output"] = conc_train_dataset.total_combinations
dataset_dict["num_training_samples"] = len(conc_train_dataset)

print(dataset_dict["num_training_samples"])

num_epochs = 100

lr = 1e-3

taimu1 = time.time()

tau_m = 'normal'
#delay = (48*2,16*2)
#delay = (96,32)
delay = None
#delay = (45, 15)
ckpt_dir = 'exp19_GC' 

snn = SNN(dataset_dict=dataset_dict, structure=(64, 4), connection_type='f',
    delay=delay, delay_type='h', tau_m = tau_m,
    win=total_time*num_seq, loss_fn='mem_sum', batch_size=batch_size, device=device,
    debug=False)

snn.set_network()

snn.model_name = f'dvs5_seq{num_seq}_{delay}'+ snn.model_name

snn.to(device)
train(snn, train_loader, test_loader, lr, num_epochs, dropout=0.0, 
    test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(100, 0.95), test_every=1)

print(f'[INFO] TIEMPO: {time.time() - taimu1}', flush=True)