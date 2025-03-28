import torch
import time
from snn_delays.experimental_models.snn_mask import Masked_SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils import train, get_device
from snn_delays.utils.test_behavior import tb_save_max_last_acc
from snn_delays.utils.hw_aware_utils import create_local_connection_mask

'''
Doesn't improve performance either in feedforward or with delays
'''

device = get_device()

# for reproducibility
torch.manual_seed(10)

dataset = 'ibm_gestures'
total_time = 100
batch_size = 128

# DATASET
DL = DatasetLoader(dataset=dataset,
                  caching='memory',
                  num_workers=0,
                  batch_size=batch_size,
                  total_time=total_time,
                  sensor_size_to=64,
                  crop_to=1e6)
train_loader, test_loader, dataset_dict = DL.get_dataloaders()
          
num_epochs = 100

lr = 1e-3
# SNN CON DELAYS
taimu1 = time.time()

tau_m = 'normal'
delay = (99,33)
ckpt_dir = 'exp3_shd50_rnn' 

# Parameters
input_size = 64
kernel_size = 4
stride = 4
channels = 2

mask = create_local_connection_mask(input_size, kernel_size, stride, channels)

print(mask.shape)

snn = Masked_SNN(dataset_dict=dataset_dict, structure=(256, 4), connection_type='f',
    delay=delay, delay_type='h', tau_m = tau_m,
    win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
    debug=False, mask=mask)

snn.set_network()

snn.input2spike_th = None 

snn.to(device)
train(snn, train_loader, test_loader, lr, num_epochs, dropout=0.0, 
    test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(100, 0.95), test_every=1)