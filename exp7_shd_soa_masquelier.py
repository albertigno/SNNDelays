import torch
import time
from snn_delays.snn import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils import train, get_device
from snn_delays.utils.test_behavior import tb_save_max_last_acc

'''
SHD dataset as in Hammouramy, Masquelier, where they get 95%
I get: 83% (batch size 128), 86% (batch size 256)
'''

device = get_device()

# for reproducibility
torch.manual_seed(10)

dataset = 'shd'
total_time = 100
batch_size = 256

# DATASET
DL = DatasetLoader(dataset=dataset,
                  caching='memory',
                  num_workers=0,
                  batch_size=batch_size,
                  total_time=total_time,
                  crop_to=1e6,
                  sensor_size_to = 140)
train_loader, test_loader, dataset_dict = DL.get_dataloaders()
          
num_epochs = 100

lr = 1e-3
# SNN CON DELAYS
taimu1 = time.time()

tau_m = 'normal'
delay = (25,1)
ckpt_dir = 'exp7_shd_soa_masquelier'  # donde se guardará

snn = SNN(dataset_dict=dataset_dict, structure=(256, 2), connection_type='f',
    delay=delay, delay_type='ho', tau_m = tau_m,
    win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
    debug=False)

snn.to(device)
train(snn, train_loader, test_loader, lr, num_epochs, dropout=0.0, lr_scale=(5.0, 2.0), 
    test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(100, 0.95), test_every=1)

print(f'[INFO] TIEMPO: {time.time() - taimu1}', flush=True)