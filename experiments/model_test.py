import torch
import time
from snn_delays.snn import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils import train, get_device
from snn_delays.utils.test_behavior import tb_save_max_last_acc

device = get_device()

# for reproducibility
torch.manual_seed(10)

dataset = 'shd'
total_time = 250
batch_size = 64

# DATASET
DL = DatasetLoader(dataset=dataset,
                   caching='memory',
                   num_workers=0,
                   batch_size=batch_size,
                   total_time=total_time)
train_loader, test_loader, dataset_dict = DL.get_dataloaders()

# SNN CON DELAYS
taimu1 = time.time()

tau_m = 'normal'
delay = (150, 6)

snn1 = SNN(dataset_dict=dataset_dict, structure=(48, 2), connection_type='f',
          delay=delay, delay_type='ho', tau_m = tau_m,
          win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
          debug=True)

snn2 = SNN(dataset_dict=dataset_dict, structure=(48, 2), connection_type='f',
          delay=None, delay_type='', tau_m = tau_m,
          win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
          debug=True)

snn3 = SNN(dataset_dict=dataset_dict, structure=(48, 2), connection_type='r',
          delay=delay, delay_type='ho', tau_m = tau_m,
          win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
          debug=True)

snn4 = SNN(dataset_dict=dataset_dict, structure=(48, 2), connection_type='r',
          delay=None, delay_type='', tau_m = tau_m,
          win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
          debug=True)

snn1.to(device)
snn2.to(device)
snn3.to(device)
snn4.to(device)

ckpt_dir = 'exp_default'  # donde se guardará
train(snn1, train_loader, test_loader, 0.25*1e-3, 50, dropout=0.0, lr_scale=(5.0, 2.0), 
      test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=5)
train(snn2, train_loader, test_loader, 1e-3, 50, dropout=0.0, lr_scale=(5.0, 2.0), 
      test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=5)
train(snn3, train_loader, test_loader, 1e-3, 50, dropout=0.0, lr_scale=(5.0, 2.0), 
      test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=5)
train(snn4, train_loader, test_loader, 1e-3, 50, dropout=0.0, lr_scale=(5.0, 2.0), 
      test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=5)

print('[INFO] TIEMPO: ', time.time() - taimu1)