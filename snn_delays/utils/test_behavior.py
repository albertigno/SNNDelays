import numpy as np
import torch
import os
#from snn_delays.utils.utils import calc_metrics

def tb_save_max_acc(snn, ckpt_dir, test_loader, dropout, test_every):
    '''
    test every "check_every" and save results only for the nets with best accuracy
    '''

    if (snn.epoch) % test_every == 0:
        snn.test(test_loader, dropout=dropout)
        last_acc = np.array(snn.acc)[-1,1]
        max_acc = np.max(np.array(snn.acc)[:,1])
        if last_acc == max_acc:
            print(f'saving max acc: {max_acc}')
            snn.save_model(
                snn.model_name+'_'+ ''.join(str(np.array(snn.acc)[-1,1]).split('.')) , 
                ckpt_dir)

def tb_save_max_last_acc(snn, ckpt_dir, test_loader, dropout, test_every):
    '''
    test every "check_every" and save results only for the nets with best accuracy.
    Remove old acc, only keep the 'max' and the last 'acc'
    '''

    if (snn.epoch) % test_every == 0:
        snn.test(test_loader, dropout=dropout)
        last_acc = np.array(snn.acc)[-1,1]
        max_acc = np.max(np.array(snn.acc)[:,1])

        # remove older acc:
        if snn.last_model_name is not None:
            snn.remove_model(snn.last_model_name, ckpt_dir)

        snn.last_model_name = snn.model_name+'_'+ ''.join(str(np.array(snn.acc)[-1,1]).split('.')) + f'_last_{snn.epoch}epoch'
        snn.save_model(snn.last_model_name, ckpt_dir)

        if last_acc == max_acc:
            if snn.last_max_model_name is not None:
                snn.remove_model(snn.last_max_model_name, ckpt_dir)
            print(f'saving max acc: {max_acc}')
            snn.last_max_model_name = snn.model_name+'_'+ ''.join(str(np.array(snn.acc)[-1,1]).split('.')) + f'_max_{snn.epoch}epoch'
            snn.save_model(snn.last_max_model_name, ckpt_dir)
    

