import torch
import time


#from snn_delays.snn_refactored_capo_backup import SNN
from snn_delays.snn_refactored_backup import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils_refact_minimal import train, get_device
from snn_delays.utils.test_behavior import tb_save_max_acc_refac

device = get_device()

# for reproducibility
torch.manual_seed(10)

dataset = 'shd'
total_time = 250
batch_size = 256  # lr=1e-4

# DATASET
DL = DatasetLoader(dataset=dataset,
                   caching='memory',
                   num_workers=0,
                   batch_size=batch_size,
                   total_time=total_time,
                   crop_to=1e6)
train_loader, test_loader, dataset_dict = DL.get_dataloaders()

# SNN CON DELAYS

tau_m = 'log-uniform-st'

# extra_kwargs = {'delay_range':(40, 1),
#                 'pruned_delays': 3}

extra_kwargs = {'delay_range':(150, 6),
                'pruned_delays': 6}

structure = (64, 2, 'd')  # delay structure

snn = SNN(dataset_dict=dataset_dict, structure=structure, tau_m=tau_m, win=50, loss_fn='mem_sum', 
          batch_size=batch_size, device=device, **extra_kwargs)
snn.set_layers()

snn.to(device) # 30 - 30

print(snn)

ckpt_dir = 'exp_default'  # donde se guardará
train(snn, train_loader, test_loader, 1e-3, 50, lr_tau= 0.1, 
      test_behavior=tb_save_max_acc_refac, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=1, freeze_taus=True)




