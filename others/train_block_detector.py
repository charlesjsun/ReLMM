import os 
import argparse
from collections import defaultdict
import numpy as np
import tensorflow as tf
from datetime import datetime

from discretizer import *
from envs import *
from losses import *
from policies import *
from replay_buffer import *
from train_functions import *

def train(train_buffer, validation_buffer, train_function, validation_function, 
          num_epochs=20, num_steps_per_epoch=100, savedir=None, model_savefunc=None):
    stats = defaultdict(list)
    num_train_steps = 0
    train_batch_size = 128
    validation_batch_size = 128
    
    for ep in range(num_epochs):
        total_training_losses = []
        total_acc = 0.
        for step in range(num_steps_per_epoch):
            data = train_buffer.sample_batch(train_batch_size)
            losses, acc = train_function(data)
            #import pdb; pdb.set_trace()
            if not isinstance(losses, (tuple, list)):
                losses = [losses]
            if len(total_training_losses) != len(losses):
                total_training_losses = [0.0 for _ in losses]
            for i in range(len(total_training_losses)):
                total_training_losses[i] += losses[i].numpy()
            num_train_steps += 1
            total_acc += acc.numpy()
        #import pdb; pdb.set_trace()
        stats['num_train_steps'].append(num_train_steps)
        stats['average_training_loss'].append([float(loss) / num_steps_per_epoch for loss in total_training_losses])
        stats['train_acc'].append(total_acc/num_steps_per_epoch)
        datas = validation_buffer.get_all_samples_in_batch(validation_batch_size)
        total_validation_loss = 0.0
        total_val_acc = 0.
        for data in datas:
            val_loss, val_acc = validation_function(data)
            total_validation_loss += val_loss.numpy()
            total_val_acc += val_acc.numpy()
        stats['validation_loss'].append(total_validation_loss / len(datas))
        stats['validation_acc'].append(total_val_acc/len(datas))
        
        print('-------------')
        for k,v in stats.items():
            print(k, v[-1])
        print('-------------')
        #print("Epoch", ep, "train loss", stats['average_training_loss'][-1], "val loss", stats['validation_loss'][-1], "train acc", acc)
        np.save(os.path.join(savedir, "diagnostics"), stats)
        if model_savefunc is not None:
            #print("saving model", savedir, "/model_"+str(ep))
            model_savefunc(ep)
            
            
def main():
    training_buffer_negative = np.load('others/logs/disc_dqn_empty_data_/train_buffer.npy', allow_pickle=True).item()
    validation_buffer_negative =  np.load('others/logs/disc_dqn_empty_data_/validation_buffer.npy', allow_pickle=True).item()
    training_buffer_positive = np.load('others/logs/disc_dqn_rectangles_data_/train_buffer.npy', allow_pickle=True).item()
    validation_buffer_positive =  np.load('others/logs/disc_dqn_rectangles_data_/validation_buffer.npy', allow_pickle=True).item()
    
    #import pdb; pdb.set_trace()
    train_buffer = ReplayBuffer(size=len(training_buffer_negative['rewards'])+len(training_buffer_positive['rewards']), 
                                observation_shape=training_buffer_negative['observations'].shape[1:], 
                                action_dim=1)
    validation_buffer = ReplayBuffer(size=len(validation_buffer_negative['rewards'])+ len(validation_buffer_positive['rewards']), 
                                observation_shape=validation_buffer_negative['observations'].shape[1:], 
                                action_dim=1)
    
    training_buffer_negative['rewards'] *= 0 
    validation_buffer_negative['rewards'] *= 0 
    training_buffer_positive['rewards'] *= 0 
    validation_buffer_positive['rewards'] *= 0 
    training_buffer_positive['rewards'] += 1
    validation_buffer_positive['rewards'] += 1 
    train_buffer.store_many_samples(training_buffer_negative)
    train_buffer.store_many_samples(training_buffer_positive)
    
    validation_buffer.store_many_samples(validation_buffer_negative)
    validation_buffer.store_many_samples(validation_buffer_positive)
    
    # Network will only have one output
    train_buffer._actions *= 0
    validation_buffer._actions *= 0
    discrete_dimension = 1
    
    logits_model = build_image_block_detector(image_size=60)
    optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    
    
    train_function = lambda data: train_discrete_sigmoid(logits_model, data, optimizer, discrete_dimension, return_acc=True)
    validation_function = lambda data: validation_discrete_sigmoid(logits_model, data, discrete_dimension, return_acc=True)

    name = 'block_detector'
    now = datetime.now()
    savedir='./others/logs/'+name+now.strftime("%m%d%Y-%H-%M-%S")
    os.makedirs(savedir)
    model_savefunc = lambda t: logits_model.save_weights(os.path.join(savedir, "model_"+str(t)))
    
    train(train_buffer, validation_buffer, train_function, validation_function, 
          num_epochs=20, num_steps_per_epoch=100, savedir=savedir, model_savefunc=model_savefunc)
    
    
main()