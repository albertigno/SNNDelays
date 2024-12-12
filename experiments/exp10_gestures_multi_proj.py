import torch
import time
from snn_delays.snn import SNN
from snn_delays.experimental_models.snn_delay_prun import P_DelaySNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils import train, get_device
from snn_delays.utils.test_behavior import tb_save_max_last_acc

'''
SHD dataset as in ablation study
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

tau_m = 'normal'
ckpt_dir = 'exp10_gestures' 
structure = (64, 2)

num_epochs = 100

lr = 1e-3
# SNN CON DELAYS
taimu1 = time.time()


########################3

config = 'd_top3_gradual'

########################3

if config == 'mf':
    connection_type = config
    delay = None
    delay_type = ''
elif config == 'f4':
    structure=(64, 4)
    connection_type = 'f'
    delay = None
    delay_type = ''
elif config == 'dc': # delay coarse
    connection_type = 'f'
    delay = (48*2, 16*2)
    delay_type = 'h'
elif config == 'da': # delay coarse
    connection_type = 'f'
    delay = (32*2, 1)
    delay_type = 'h'
elif config == 'd_top3': # delay coarse
    connection_type = 'f'
    delay = (32*2, 1)
    delay_type = 'h'
    n_delays = 3
    num_epochs = 50
elif config == 'd_top3_gradual':
    connection_type = 'f'
    delay = (32*2, 1)
    delay_type = 'h'
    n_delays = 3
    num_epochs = 50
# snn = SNN(dataset_dict=dataset_dict, structure=(64, 2), connection_type='mf',
#     delay=None, delay_type='', tau_m = tau_m,
#     win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
#     debug=False)

snn = SNN(dataset_dict=dataset_dict, structure=structure, connection_type=connection_type,
    delay=delay, delay_type=delay_type, tau_m = tau_m,
    win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
    debug=False)
snn.multi_proj = 3

snn.set_network()

snn.to(device)
train(snn, train_loader, test_loader, lr, num_epochs, dropout=0.0, 
    test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(100, 0.95), test_every=1)

if 'top' in config:
    p_snn = P_DelaySNN(dataset_dict=dataset_dict, structure=structure, connection_type=connection_type,
        delay=delay, delay_type=delay_type, n_pruned_delays=n_delays, tau_m = tau_m,
        win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
        debug=False)

    p_snn.set_network()
    p_snn.to(device)

    for (name_src, param_src), (name_dst, param_dst) in zip(snn.named_parameters(), p_snn.named_parameters()):
        assert name_src.split('.')[0] == name_dst.split('.')[0], f"Parameter mismatch: {name_src} != {name_dst}"
        param_dst.data.copy_(param_src.data)    

    if config == 'd_top3':
        train(p_snn, train_loader, test_loader, lr, num_epochs, dropout=0.0, 
            test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(100, 0.95), test_every=1)
    elif config == 'd_top3_gradual':
        for n_delays in range(60,2, -1):
            p_snn.f1_f2.top_k = n_delays
            print("pruned to: " + str(n_delays))
            train(p_snn, train_loader, test_loader, lr, 5, dropout=0.0, 
                test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=1)
        train(p_snn, train_loader, test_loader, lr, 50, dropout=0.0, 
                test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=1)
