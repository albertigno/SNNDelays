import torch
import time
from snn_delays.snn import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils import train, get_device
from snn_delays.utils.test_behavior import tb_save_max_last_acc

'''
stmnist dataset as in ablation study
'''

device = get_device()

# for reproducibility
torch.manual_seed(10)

dataset = 'psmnist'
total_time = 196
batch_size = 1024

# DATASET
DL = DatasetLoader(dataset=dataset,
                   caching='memory',
                   num_workers=0,
                   batch_size=batch_size,
                   downsample=True)

train_loader, test_loader, dataset_dict = DL.get_dataloaders()
          
num_epochs = 20

lr = 1e-3
# SNN CON DELAYS
taimu1 = time.time()

tau_m = 'normal'
delay = (180,60)
ckpt_dir = 'exp9_mt_psmnist' 

dataset_dict["time_ms"] = 1e3 # assuming a total time equal to 1 second

snn = SNN(dataset_dict=dataset_dict, structure=(64, 2), connection_type='f',
    delay=delay, delay_type='h', tau_m = tau_m,
    win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
    debug=False)

# 48: 3s/epoca
snn.input2spike_th = None # 48 -> (81%)

snn.to(device)
train(snn, train_loader, test_loader, lr, num_epochs, dropout=0.0, 
    test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(100, 0.95), test_every=1)

print(f'[INFO] TIEMPO: {time.time() - taimu1}', flush=True)