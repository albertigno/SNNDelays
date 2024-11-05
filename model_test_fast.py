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
total_time = 30
batch_size = 1024  # lr=1e-4

# DATASET
DL = DatasetLoader(dataset=dataset,
                   caching='memory',
                   num_workers=0,
                   batch_size=batch_size,
                   total_time=total_time)
train_loader, test_loader, dataset_dict = DL.get_dataloaders()

# SNN CON DELAYS
taimu1 = time.time()

tau_m = 3.0

snn1 = SNN(dataset_dict=dataset_dict, structure=(48, 2), connection_type='f',
          delay=(10, 1), delay_type='ho', tau_m = tau_m,
          win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
          debug=True)

snn2 = SNN(dataset_dict=dataset_dict, structure=(48, 2), connection_type='f',
          delay=None, delay_type='', tau_m = tau_m,
          win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
          debug=True)

snn3 = SNN(dataset_dict=dataset_dict, structure=(48, 2), connection_type='r',
          delay=(10, 1), delay_type='ho', tau_m = tau_m,
          win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
          debug=True)

snn4 = SNN(dataset_dict=dataset_dict, structure=(48, 2), connection_type='r',
          delay=None, delay_type='', tau_m = tau_m,
          win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
          debug=True)

snn1.to(device) # 30 - 30
snn2.to(device) # 11 - 24
snn3.to(device) # 30 - 4
snn4.to(device) # 10 - 35

ckpt_dir = 'exp_default'  # donde se guardará
train(snn1, train_loader, test_loader, 4*1e-3, 5, dropout=0.0, lr_scale=(5.0, 2.0), 
      test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=5)
train(snn2, train_loader, test_loader, 4*1e-3, 5, dropout=0.0, lr_scale=(5.0, 2.0), 
      test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=5)
train(snn3, train_loader, test_loader, 4*1e-3, 5, dropout=0.0, lr_scale=(5.0, 2.0), 
      test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=5)
train(snn4, train_loader, test_loader, 4*1e-3, 5, dropout=0.0, lr_scale=(5.0, 2.0), 
      test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=5)

print('[INFO] TIEMPO: ', time.time() - taimu1)