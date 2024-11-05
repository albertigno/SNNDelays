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

for batch_size in [256, 512, 1024, 8, 16, 32, 64, 128, 2, 1]:

      dataset = 'shd'
      total_time = 250
      #batch_size = 2
      bs_factor = batch_size/256.0

      num_epochs = (batch_size*96) // 256

      if num_epochs < 1:
            num_epochs = 1

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
      delay = (150,6)
      ckpt_dir = 'exp_dymbatch5'  # donde se guardará

      snn = SNN(dataset_dict=dataset_dict, structure=(32, 2), connection_type='r',
            delay=None, delay_type='h', tau_m = tau_m,
            win=total_time, loss_fn='mem_sum', batch_size=batch_size, device=device,
            debug=True)
      
      snn.model_name = snn.model_name + "_b" + str(batch_size)

      snn.to(device)
      train(snn, train_loader, test_loader, bs_factor*1e-3, num_epochs, dropout=0.0, lr_scale=(5.0, 2.0), 
            test_behavior=tb_save_max_last_acc, ckpt_dir=ckpt_dir, scheduler=(10, 0.95), test_every=5)

      print(f'[INFO] TIEMPO: {time.time() - taimu1}', flush=True)