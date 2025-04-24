import numpy as np
import torch
import os
#from snn_delays.utils.utils import calc_metricsç
from snn_delays.utils.train_utils import get_gradients

def tb_minimal(snn, ckpt_dir, test_loader, test_every):
    '''
    test every "check_every" and save results only for the nets with best accuracy
    '''

    if (snn.epoch) % test_every == 0:
        snn.test(test_loader)
        max_acc = np.max(np.array(snn.acc)[:,1])
        print(f'max acc: {max_acc}')


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
            

def tb_save_max_acc_refac(snn, ckpt_dir, test_loader, test_every):
    '''
    test every "check_every" and save results only for the nets with best accuracy
    '''

    if (snn.epoch) % test_every == 0:
        snn.test(test_loader)
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
    Remove old acc, only keep the 'max' and the last 'acc'.
    Save and plot gradients
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

        # stored_grads = get_gradients(snn)

        # for name, grad in stored_grads.items():
        #     grad_norm = grad.norm().item()
        #     print(f"Gradient norm for '{name}': {grad_norm:.4f}")

        if last_acc == max_acc:
            if snn.last_max_model_name is not None:
                snn.remove_model(snn.last_max_model_name, ckpt_dir)
            print(f'saving max acc: {max_acc}')
            snn.last_max_model_name = snn.model_name+'_'+ ''.join(str(np.array(snn.acc)[-1,1]).split('.')) + f'_max_{snn.epoch}epoch'
            snn.save_model(snn.last_max_model_name, ckpt_dir)
    

def tb_addtask(snn, ckpt_dir, test_loader, dropout, test_every):
    if (snn.epoch) % test_every == 0:
        #if epoch>=50:
            #print('pooling delays')
            #snn.pool_delays('i', k= 10, freeze=False)
        for images, labels in test_loader:
            pred, ref = snn.propagate(images.to(snn.device), labels.to(snn.device))

        snn.save_model(snn.model_name, ckpt_dir)

        loss = snn.criterion(pred, ref)
        #tolerance = 0.04 # a             
        #correct = torch.sum(abs(pred-ref) < 0.04)
        #print(f"prediction {pred}, reference {ref}")
        print(f'Mean Error: {loss.item()/len(images)}% ')
        print('--------------------------')


def tb_addtask_refact(snn, ckpt_dir, test_loader, test_every):
    if (snn.epoch) % test_every == 0:
        #if epoch>=50:
            #print('pooling delays')
            #snn.pool_delays('i', k= 10, freeze=False)
        for images, labels in test_loader:
            pred, ref = snn.propagate(images.to(snn.device), labels.to(snn.device))

        snn.save_model(snn.model_name, ckpt_dir)

        loss = snn.criterion(pred, ref)
        #tolerance = 0.04 # a             
        #correct = torch.sum(abs(pred-ref) < 0.04)
        #print(f"prediction {pred}, reference {ref}")
        print(f'Mean Error: {loss.item()/len(images)}% ')
        print('--------------------------')


def tb_synthetic_refact(snn, ckpt_dir, test_loader, test_every):
    if (snn.epoch) % test_every == 0:

        for images, labels in test_loader:
            pred, ref = snn.propagate(images.to(snn.device), labels.to(snn.device))

        last_loss = np.array(snn.train_loss)[-1,1]
        min_loss = np.min(np.array(snn.train_loss)[:,1])

        if snn.last_model_name is not None:
            snn.remove_model(snn.last_model_name, ckpt_dir)

        snn.last_model_name = snn.model_name+f'_last_{snn.epoch}epoch'
        snn.save_model(snn.last_model_name, ckpt_dir)

        if last_loss == min_loss:
            if snn.last_min_model_name is not None:
                snn.remove_model(snn.last_min_model_name, ckpt_dir)

            print(f'saving min loss: {min_loss}')
            snn.last_min_model_name = snn.model_name+'_'+ f'_minloss_{snn.epoch}epoch'

        snn.save_model(snn.model_name, ckpt_dir)

        loss = snn.criterion(pred, ref)
        print(f'Mean Error: {loss.item()/len(images)}% ')
        print('--------------------------')