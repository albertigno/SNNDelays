import torch
import time
from snn_delays.experimental_models.snn_binary import BinarySNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.model_loader import ModelLoader
from snn_delays.utils.train_utils import train, get_device, transfer_weights_taus
from snn_delays.utils.test_behavior import tb_save_max_last_acc

'''
22/11/2024

with delays. base 85%
surprisingly good results with binary! almost 65%!
impressive results with sign! over 80%!
caveat: it was using trainable taus....

without trainable taus:

with recurrent. base 81% to 64% (sign) the true binary one doesn't learn!



'''

device = get_device()

# for reproducibility
torch.manual_seed(10)

dataset = 'shd'
total_time = 50
batch_size = 1024

# DATASET
DL = DatasetLoader(dataset=dataset,
                   caching='memory',
                   num_workers=0,
                   batch_size=batch_size,
                   total_time=total_time,
                   crop_to=1e6)
train_loader, test_loader, dataset_dict = DL.get_dataloaders()
          
num_epochs = 100

lr = 1e-3
# SNN CON DELAYS
taimu1 = time.time()

tau_m = 'normal'
delay = (48,16)
ckpt_dir = 'exp13_shd50_binary' 

# snn = BinarySNN(dataset_dict=dataset_dict, structure=(64, 2), connection_type='f',
#     delay=delay, delay_type='h', tau_m = tau_m,
#     win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
#     debug=False, binary_mode='sign')

snn = BinarySNN(dataset_dict=dataset_dict, structure=(64, 2), connection_type='r',
    delay=None, delay_type='h', tau_m = tau_m,
    win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
    debug=False, binary_mode='binary')

snn.set_network()

#base_snn =  ModelLoader('f_d_2l_ht_tt_rpt2_8529151943462898_max_42epoch','abl1_shd50', batch_size, device, False)
base_snn =  ModelLoader('r_nd_2l_ht_tt_rpt4_8193462897526501_max_45epoch','abl1_shd50', batch_size, device, False)

base_snn.to(device)
base_snn.test(test_loader)

#snn = transfer_weights_taus(base_snn, snn)

weight_taus = [(name, w) for name, w  in snn.named_parameters() if 's' not in name]

for (name_src, param_src), (name_dst, param_dst) in zip(base_snn.named_parameters(), weight_taus):
    assert name_src == name_dst, f"Parameter mismatch: {name_src} != {name_dst}"
    param_dst.data.copy_(param_src.data)

snn.to(device)
snn.test(test_loader)

train(snn, train_loader, test_loader, lr, num_epochs, dropout=0.0, 
    test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(100, 0.95), test_every=1)