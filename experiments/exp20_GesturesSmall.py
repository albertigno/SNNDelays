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

dataset = 'ibm_gestures'
total_time = 5
batch_size = 128

# DATASET
DL = DatasetLoader(dataset=dataset,
                  caching='memory',
                  num_workers=0,
                  batch_size=batch_size,
                  total_time=total_time,
                  sensor_size_to=64,
                  crop_to=3e6)

train_loader, test_loader, dataset_dict = DL.get_dataloaders()

num_epochs = 100

lr = 1e-3

taimu1 = time.time()

tau_m = 'normal'
#delay = (96,32)
delay = None
ckpt_dir = 'exp20_Gestures_small' 

#n_hidden = 8
n_hidden = 64

snn = SNN(dataset_dict=dataset_dict, structure=(n_hidden, 2), connection_type='f',
    delay=delay, delay_type='h', tau_m = tau_m,
    win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
    debug=False)

snn.model_name = f'{n_hidden}_{delay}'+ snn.model_name

snn.set_network()

snn.to(device)
train(snn, train_loader, test_loader, lr, num_epochs, dropout=0.0, 
    test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(100, 0.95), test_every=1)

print(f'[INFO] TIEMPO: {time.time() - taimu1}', flush=True)