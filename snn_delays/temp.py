from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils_refact_minimal import train
from snn_delays.snn_refactored import SNN

dataset = 'ibm_gestures'
total_time = 50
batch_size = 64

# DATASET
DL = DatasetLoader(dataset=dataset,
                  caching='memory',
                  num_workers=0,
                  batch_size=batch_size,
                  total_time=total_time,
                  crop_to=1e6)
train_loader, test_loader, dataset_dict = DL.get_dataloaders()

snn = SNN(dataset_dict=dataset_dict,
    structure=(64, 2, 'f'),
    tau_m='normal',
    win=total_time,
    loss_fn='mem_sum',
    batch_size=batch_size,
    device='cuda:0',
    debug=True)

snn.set_layers()
snn.to(snn.device)


train(snn, train_loader, test_loader, 1e-3, 10)