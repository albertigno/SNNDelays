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
        model_name, location, batch_size, device, debug = args

        params = torch.load(
            os.path.join(CHECKPOINT_PATH, location, model_name),
            map_location=torch.device('cpu'))

        params['kwargs']['batch_size'] = batch_size
        params['kwargs']['device'] = device
        params['kwargs']['debug'] = debug

        kwargs = params['kwargs']

        # For backwards compatibility
        if kwargs['tau_m'] == 'adp':
            print('[WARNING] Loading an old version, tau_m changed to gamma.')
            kwargs['tau_m'] = 'gamma'

        # For backwards compatibility
        if kwargs['loss_fn'] == 'sum':
            print('[WARNING] Loading an old version, loss_fn=sum changed to spk_sum.')
            kwargs['loss_fn'] = 'spk_count'
        elif kwargs['loss_fn'] == 'mot':
            print('[WARNING] Loading an old version, loss_fn=mot changed to mem_sum.')
            kwargs['loss_fn'] = 'mem_sum'

        if 'dataset' in kwargs.keys():
            print('[WARNING] Loading an old version, check arguments below.')
            d = kwargs['dataset']
            del kwargs['dataset'] # Delete it from the stuff
            kwargs['dataset_dict'] = cls.__get_dict_old_way(cls, d)
            print(kwargs)

        if 'mask' in kwargs.keys():
            del kwargs['mask']

        snn = params['type']
        snn = snn(**kwargs)
        snn.set_network()
        snn.to(device)
        snn.load_state_dict(params['net'], strict= False) # be careful with this
        snn.epoch = params['epoch']
        snn.acc = params['acc_record']
        snn.train_loss = params['train_loss']
        snn.test_loss = params['test_loss']
        snn.test_spk_count = params['test_spk']

        # For backwards compatibility
        if 'model_name' not in params.keys():
            print('[WARNING] Loading and old version, model_name changed '
                  'to default.')
            snn.model_name = 'default'

        print('Instance of {} loaded successfully'.format(params['type']))

        return snn
    
    def __get_dict_old_way(cls, dataset_name):

        dict_path = os.path.join(DATASET_PATH, 'dataset_configs',
                                dataset_name + '.json')

        if os.path.isfile(dict_path):
            with open(dict_path, 'r') as f:
                data_dict = json.load(f)

        else:
            sys.exit('[ERROR] The dictionary of the dataset used does not '
                    'exit. create the dictionary in dataset_configs')
            
        data_dict['num_training_samples'] = data_dict['num_train_samples']
        data_dict['dataset_name'] = dataset_name
        del data_dict['num_train_samples']

        return data_dict

