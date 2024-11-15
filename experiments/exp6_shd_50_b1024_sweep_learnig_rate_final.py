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

dataset = 'shd'
total_time = 50
batch_size = 1024

# DATASET
DL = DatasetLoader(dataset=dataset,
                  caching='memory',
                  num_workers=0,
                  batch_size=batch_size,
                  total_time=total_time)
train_loader, test_loader, dataset_dict = DL.get_dataloaders()

for lr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2]: 
           
      #num_epochs = (batch_size*96) // 256
      num_epochs = 100

      # SNN CON DELAYS
      taimu1 = time.time()

      tau_m = 'normal'
      #delay = (150,6)
      delay = (64,16)
      ckpt_dir = 'exp_soa50_1024_learning_rate_final'  # donde se guardará

      snn = SNN(dataset_dict=dataset_dict, structure=(64, 2), connection_type='f',
            delay=delay, delay_type='h', tau_m = tau_m,
            win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
            debug=False)
      
      snn.input2spike_th = None
      
      snn.model_name = "delay_lr" + str(lr) + "_" + snn.model_name

      snn.to(device)
      train(snn, train_loader, test_loader, lr, num_epochs, dropout=0.0, lr_scale=(5.0, 2.0), 
            test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(100, 0.95), test_every=1)

      print(f'[INFO] TIEMPO: {time.time() - taimu1}', flush=True)

for lr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2]: 
           
      #num_epochs = (batch_size*96) // 256
      num_epochs = 100

      # SNN CON DELAYS
      taimu1 = time.time()

      tau_m = 'normal'
      #delay = (150,6)
      delay = None
      ckpt_dir = 'exp_soa50_1024_learning_rate_final'  # donde se guardará

      snn = SNN(dataset_dict=dataset_dict, structure=(64, 2), connection_type='f',
            delay=delay, delay_type='h', tau_m = tau_m,
            win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
            debug=False)
      
      snn.input2spike_th = None
      
      snn.model_name = "ffw_lr" + str(lr) + "_" + snn.model_name

      snn.to(device)
      train(snn, train_loader, test_loader, lr, num_epochs, dropout=0.0, lr_scale=(5.0, 2.0), 
            test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(100, 0.95), test_every=1)

      print(f'[INFO] TIEMPO: {time.time() - taimu1}', flush=True)


for lr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2]: 
           
      #num_epochs = (batch_size*96) // 256
      num_epochs = 100

      # SNN CON DELAYS
      taimu1 = time.time()

      tau_m = 'normal'
      #delay = (150,6)
      delay = None
      ckpt_dir = 'exp_soa50_1024_learning_rate_final'  # donde se guardará

      snn = SNN(dataset_dict=dataset_dict, structure=(64, 2), connection_type='r',
            delay=delay, delay_type='h', tau_m = tau_m,
            win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
            debug=False)
      
      snn.input2spike_th = None
      
      snn.model_name = "rec_  lr" + str(lr) + "_" + snn.model_name

      snn.to(device)
      train(snn, train_loader, test_loader, lr, num_epochs, dropout=0.0, lr_scale=(5.0, 2.0), 
            test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(100, 0.95), test_every=1)

      print(f'[INFO] TIEMPO: {time.time() - taimu1}', flush=True)