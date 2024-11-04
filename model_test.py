import torch
import time
from snn_delays.snn import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils import train, get_device
from snn_delays.utils.test_behavior import tb_save_max_last_acc

#device = get_device()

device = 'cpu'

# for reproducibility
torch.manual_seed(10)

dataset = 'shd'
total_time = 30
batch_size = 1024  # lr=1e-4

# DATASET
DL = DatasetLoader(dataset=dataset,
                   caching='disk',
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
.
snn1.to(device)
snn2.to(device)
snn3.to(device)
snn4.to(device)

ckpt_dir = 'exp_default'  # donde se guardará
train(snn1, train_loader, test_loader, 4*1e-3, 5, dropout=0.0, lr_scale=(5.0, 2.0), 
      test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=5, lsm=True)
train(snn2, train_loader, test_loader, 4*1e-3, 5, dropout=0.0, lr_scale=(5.0, 2.0), 
      test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=5, lsm=True)
train(snn3, train_loader, test_loader, 4*1e-3, 5, dropout=0.0, lr_scale=(5.0, 2.0), 
      test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=5, lsm=True)
train(snn4, train_loader, test_loader, 4*1e-3, 5, dropout=0.0, lr_scale=(5.0, 2.0), 
      test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=5, lsm=True)

print('[INFO] TIEMPO: ', time.time() - taimu1)