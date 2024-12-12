import torch
import time
from snn_delays.snn import SNN
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils import train, get_device
from snn_delays.utils.test_behavior import tb_save_max_last_acc

print("packages loaded", flush=True)

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

for lr in [1e-4, 1e-3, 5e-3, 1e-2]: 
      for lr_tau in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0]:         
            #num_epochs = (batch_size*96) // 256
            num_epochs = 100

            # SNN CON DELAYS
            taimu1 = time.time()

            tau_m = 'normal'
            #delay = (150,6)
            delay = (48,16)
            ckpt_dir = 'exp7_gestures100_b128_sweep_learnig_rate_also_tau'  # donde se guardará

            snn = SNN(dataset_dict=dataset_dict, structure=(64, 2), connection_type='f',
                  delay=delay, delay_type='h', tau_m = tau_m,
                  win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
                  debug=False)
            
            snn.set_network()
            
            snn.model_name = "lr" + str(lr) + "_lrt" + str(lr_tau) + "_" + snn.model_name

            snn.to(device)
            train(snn, train_loader, test_loader, lr, num_epochs, dropout=0.0, lr_tau=lr_tau, 
                  test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(100, 0.95), test_every=1)

            print(f'[INFO] TIEMPO: {time.time() - taimu1}', flush=True)

for lr in [1e-4, 1e-3, 5e-3, 1e-2]: 
      for lr_tau in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0]:       
            #num_epochs = (batch_size*96) // 256
            num_epochs = 100

            # SNN CON DELAYS
            taimu1 = time.time()

            tau_m = 'normal'
            #delay = (150,6)
            delay = None
            ckpt_dir = 'exp7_gestures100_b128_sweep_learnig_rate_also_tau'  # donde se guardará

            snn = SNN(dataset_dict=dataset_dict, structure=(64, 2), connection_type='f',
                  delay=delay, delay_type='h', tau_m = tau_m,
                  win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
                  debug=False)
            
            snn.set_network()
            
            snn.model_name = "ffw_lr" + str(lr) + "_lrt" + str(lr_tau) + "_" + snn.model_name

            snn.to(device)
            train(snn, train_loader, test_loader, lr, num_epochs, dropout=0.0, lr_tau=lr_tau, 
                  test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(100, 0.95), test_every=1)

            print(f'[INFO] TIEMPO: {time.time() - taimu1}', flush=True)


for lr in [1e-4, 1e-3, 5e-3, 1e-2]: 
      for lr_tau in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0]: 

            #num_epochs = (batch_size*96) // 256
            num_epochs = 100

            # SNN CON DELAYS
            taimu1 = time.time()

            tau_m = 'normal'
            #delay = (150,6)
            delay = None
            ckpt_dir = 'exp7_gestures100_b128_sweep_learnig_rate_also_tau'  # donde se guardará

            snn = SNN(dataset_dict=dataset_dict, structure=(64, 2), connection_type='r',
                  delay=delay, delay_type='h', tau_m = tau_m,
                  win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
                  debug=False)
            
            snn.set_network()
            
            snn.model_name = "rec_lr" + str(lr) + "_lrt" + str(lr_tau) + "_" + snn.model_name

            snn.to(device)
            train(snn, train_loader, test_loader, lr, num_epochs, dropout=0.0, lr_tau=lr_tau, 
                  test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(100, 0.95), test_every=1)

            print(f'[INFO] TIEMPO: {time.time() - taimu1}', flush=True)