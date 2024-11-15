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
batch_size = 64

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
delay = (48, 16)
ckpt_dir = 'exp_dymbatch3'  # donde se guardará

# snn1 = SNN(dataset_dict=dataset_dict, structure=(32, 2), connection_type='f',
#           delay=delay, delay_type='h', tau_m = tau_m,
#           win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
#           debug=True)

snn1 = SNN(dataset_dict=dataset_dict, structure=(32, 2), connection_type='r',
          delay=None, delay_type='h', tau_m = tau_m,
          win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
          debug=True)


# few epochs regime: batch size 64: 127 updates * 10 epochs
# f

# Avg spk_count per neuron for all 50 time-steps 5.505441665649414
# Avg spk per neuron per layer [10.445795605123674, 11.575971731448764]
# Test Accuracy of the model on the test samples: 47.482
# Gradient norm for 'tau_m_1': 0.0035
# Gradient norm for 'tau_m_2': 0.0053
# Gradient norm for 'tau_m_o': 0.2776
# Gradient norm for 'f0_f1.weight': 0.4359
# Gradient norm for 'f1_f2.weight': 0.7679
# Gradient norm for 'f2_o.weight': 3.6980

# f+d
# Avg spk_count per neuron for all 50 time-steps 6.310981750488281
# Avg spk per neuron per layer [10.281595075088338, 14.96233160335689]
# Test Accuracy of the model on the test samples: 75.309
# Gradient norm for 'tau_m_1': 0.0011
# Gradient norm for 'tau_m_2': 0.0069
# Gradient norm for 'tau_m_o': 0.2777
# Gradient norm for 'f0_f1.weight': 0.2646
# Gradient norm for 'f1_f2.weight': 0.8358
# Gradient norm for 'f2_o.weight': 3.5676

# r
# Avg spk_count per neuron for all 50 time-steps 8.367349624633789
# Avg spk per neuron per layer [15.306123012367491, 18.163275728798588]
# Test Accuracy of the model on the test samples: 54.638
# Gradient norm for 'tau_m_1': 0.0034
# Gradient norm for 'tau_m_2': 0.0062
# Gradient norm for 'tau_m_o': 0.4148
# Gradient norm for 'f0_f1.weight': 0.2033
# Gradient norm for 'f1_f1.weight': 0.0977
# Gradient norm for 'f1_f2.weight': 0.8959
# Gradient norm for 'f2_f2.weight': 0.8815
# Gradient norm for 'f2_o.weight': 7.7288

snn1.to(device)
train(snn1, train_loader, test_loader, 0.25*1e-3, 30, dropout=0.0, lr_scale=(5.0, 2.0), 
      test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(100, 0.9), test_every=5)

print('[INFO] TIEMPO: ', time.time() - taimu1)