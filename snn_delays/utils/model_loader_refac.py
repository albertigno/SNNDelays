import os
import torch
import json
from snn_delays.config import CHECKPOINT_PATH
from snn_delays.config import DATASET_PATH
import sys 

class ModelLoader:
    """
    Model Loader class.

    Load a neural network previously trained and saved.

    arguments = model_name, location, batch_size, device, debug
    """

    def __new__(cls, *args, **kwargs):
        model_name, location, batch_size, device = args

        params = torch.load(
            os.path.join(CHECKPOINT_PATH, location, model_name),
            map_location=torch.device('cpu'))

        params['kwargs']['batch_size'] = batch_size
        params['kwargs']['device'] = device

        kwargs = params['kwargs']

        snn = params['type']
        snn = snn(**kwargs)
        snn.set_layers()
        snn.to(device)
        snn.load_state_dict(params['net'], strict= False) # be careful with this
        snn.epoch = params['epoch']
        snn.acc = params['acc_record']
        snn.train_loss = params['train_loss']
        snn.test_loss = params['test_loss']
        snn.test_spk_count = params['test_spk']

        print('Instance of {} loaded successfully'.format(params['type']))

        return snn

