import torch
import time
#from snn_delays.experimental_models.snn_binary import BinarySNN
from snn_delays.experimental_models.snn_delay_prun import P_DelaySNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.model_loader import ModelLoader
from snn_delays.utils.train_utils import train, get_device, transfer_weights_taus, copy_snn
from snn_delays.utils.test_behavior import tb_save_max_last_acc

'''
22/11/2024
anything over 81% would be fine. Got just 72% after 50 epochs....
With gradual delays: 81%?

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

#tau_m = 'normal'
#delay = (48,16)
#delay = (100,1)
ckpt_dir = 'exp14_shd50_pruned3' 

# snn = P_DelaySNN(dataset_dict=dataset_dict, structure=(64, 2), connection_type='f',
#     delay=delay, delay_type='h', n_pruned_delays=3, tau_m = tau_m,
#     win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
#     debug=False)

#snn =  ModelLoader('DynamicGradualPruninDVSGest_ibm_gestures100_l2_100d1.t7_7992424242424242_max_73epoch',ckpt_dir, batch_size, device, False)
### 79.92% to 79.92%

snn =  ModelLoader('DynamicGradualPruninDVSGest_ibm_gestures100_l2_100d1.t7_7121212121212122_last_98epoch',ckpt_dir, batch_size, device, False)
### 71.21% to 77.27% (dynamic mask)
### 71.21% to --% (fixed mask)

snn.model_name = 'DynamicGradualPruninDVSGest_finetuned_'

snn.set_network()

snn.to(device)

# pruning gradually from 100 to 3 delays
snn.f1_f2.top_k = 3
print("pruned to: " + str(snn.f1_f2.top_k))
train(snn, train_loader, test_loader, lr, 50, dropout=0.0, 
    test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=1)
