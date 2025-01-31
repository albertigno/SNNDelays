import torch
from snn_delays.snn import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.test_behavior import tb_save_max_last_acc
from snn_delays.utils.train_utils import train, get_device
import matplotlib.pyplot as plt

device = get_device()

# For reproducibility
torch.manual_seed(10)

### DATASET

# Parameters
dataset = 'nmnist'
total_time = 50
batch_size = 1024

# Resized dataloader
# DATASET
DL = DatasetLoader(dataset=dataset,
                   caching='gpu',
                   num_workers=0,
                   batch_size=batch_size,
                   total_time=total_time, 
                   sensor_size_to=17)

# DL = DatasetLoader(dataset=dataset,
#                    caching='memory',
#                    num_workers=0,
#                    batch_size=batch_size,
#                    total_time=total_time)

train_loader, test_loader, dataset_dict = DL.get_dataloaders()

structure = (64, 1)

#dataset_dict["time_ms"] = 2e3 # first version (mean alpha: 0.144)
dataset_dict["time_ms"] = 1 # second version (mean alpha: 0.998)

loss_fn = 'mem_sum'

snn = SNN(dataset_dict=dataset_dict, structure=structure, connection_type='f',
          delay=None, delay_type='', tau_m='normal',
          reset_to_zero=True, win=total_time,
          loss_fn=loss_fn, batch_size=batch_size, device=device,
          debug=True)

snn.set_network()

print(torch.mean(torch.sigmoid(snn.tau_m_1)))

snn.to(device)

snn.model_name = f'full_weights_17_alpha1_{structure}' + snn.model_name

num_epochs = 10
lr = 1e-3
ckpt_dir = 'exp_snn2chip_nmnist'

train(snn, train_loader, test_loader, lr, num_epochs, dropout=0.0, 
    test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=1, freeze_taus=True)