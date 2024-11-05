import torch
import time
from snn_delays.snn import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils import train, get_device, copy_snn
from snn_delays.utils.test_behavior import tb_save_max_last_acc

device = get_device()

# for reproducibility
torch.manual_seed(10)

dataset = 'shd'
total_time = 50
batch_size = 256

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
#delay = (150, 6)
#delay = (10, 1)
delay = (20, 5)
ckpt_dir = 'exp_dymbatch2'  # donde se guardará

snn1 = SNN(dataset_dict=dataset_dict, structure=(64, 1), connection_type='f',
          delay=delay, delay_type='ho', tau_m = tau_m,
          win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
          debug=True)

snn1.to(device)
train(snn1, train_loader, test_loader, 1e-3, 15, dropout=0.0, lr_scale=(5.0, 2.0), 
      test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 1.0), test_every=5)

# DATASET
DL = DatasetLoader(dataset=dataset,
                   caching='memory',
                   num_workers=0,
                   batch_size=1024,
                   total_time=total_time)
train_loader, test_loader, dataset_dict = DL.get_dataloaders()

snn2 = copy_snn(snn1, 1024)
snn2.to(device)
train(snn2, train_loader, test_loader, 1e-3, 85, dropout=0.0, lr_scale=(5.0, 2.0), 
      test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 1.0), test_every=5)

print('[INFO] TIEMPO: ', time.time() - taimu1)