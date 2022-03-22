"""
Creates a fully connected feedforward neural network.

It includes methods to define, configure, train, and evaluate the model. 

Created 10 Mar 2022

@author: Brian Wade
@version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random
from contextlib import redirect_stdout
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, History
from tensorflow.keras.utils import plot_model


class NN_Model:
    def __init__(self, config, input_dim, output_dim):
        self.config = config
        #self.num_layers = len(config.hidden_nodes)
        self.layer_sizes = config.hidden_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.history = History()

        self.decay_step = round(self.config.max_epochs / self.config.num_decays_during_training)


    def make_model(self):
        "Creates the neural net model"
        # Make NN model
        self.model = Sequential()
        # Input and hidden layers
        for layer_num, layer_size in enumerate(self.layer_sizes):
            if layer_num == 0:
                self.model.add(Dense(layer_size, input_dim = self.input_dim, activation = 'relu')) 
            else:
                self.model.add(Dense(layer_size, activation = 'relu'))

        # binary classifier so use sigmoid at output layer
        self.model.add(Dense(self.output_dim, activation = 'sigmoid'))
        
        # Compile the network 
        #optimizer1 = keras.optimizers.Adam(lr = self.config.learning_rate)
        if self.config.train_fun == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate = self.config.learning_rate)
        else:
            optimizer = keras.optimizers.RMSprop(learning_rate = self.config.learning_rate)
        
        self.model.compile(optimizer = optimizer, loss = self.config.loss, metrics = self.config.metrics)

        return self.model.summary


    def train_model(self, x_data, y_data, model_name):

        es = EarlyStopping(monitor = 'val_loss', 
                            mode = 'min', 
                            verbose = self.config.verbose, 
                            patience = self.config.patience, 
                            min_delta = self.config.min_delta,
                            restore_best_weights = True)
                
        tnan = TerminateOnNaN()

        full_model_name = os.path.join(self.config.current_dir, 'Models', model_name)
        mc = ModelCheckpoint(full_model_name, 
                            monitor = 'val_loss', 
                            mode = 'min', 
                            verbose = self.config.verbose, 
                            save_best_only = True)

        def lr_step_power_scheduler(epoch, lr):
            if epoch % self.decay_step == 0 and epoch:
                return lr * pow(self.config.decay_rate, np.floor(epoch / self.decay_step))
            return lr

        def linear_lr_dec(epoch, lr):
            lr = self.config.learning_rate - ((self.config.learning_rate - self.config.learning_rate_min)/self.config.max_epochs)*epoch
            return lr

        def cosine_annealing(epoch, lr):
            current_max_lr = self.config.learning_rate - ((self.config.learning_rate - self.config.learning_rate_min) / self.config.max_epochs) * epoch 
                
            epochs_per_cycle = math.floor(self.config.max_epochs / self.config.num_decays_during_training)
            cos_inner = (math.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
            lr = current_max_lr/2 * (math.cos(cos_inner) + 1)
            return lr

        def cosine_annealing_linear(epoch, lr):
            if epoch <= self.config.max_epochs / 2:
                current_max_lr = self.config.learning_rate
            else: 
                current_max_lr = self.config.learning_rate_min + ((self.config.learning_rate - self.config.learning_rate_min)/(self.config.max_epochs - (self.config.max_epochs / 2)))*(self.config.max_epochs - epoch)

            epochs_per_cycle = math.floor(self.config.max_epochs / self.config.num_decays_during_training)
            cos_inner = (math.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
            lr = current_max_lr/2 * (math.cos(cos_inner) + 1)        
            return lr

        def no_lr_sched(epoch, lr):
            return lr

        if self.config.learning_rate_scheduler == 'lr_step_power_scheduler':
            lr_sched_method = lr_step_power_scheduler
        elif self.config.learning_rate_scheduler == 'linear_lr_dec':
            lr_sched_method = linear_lr_dec
        elif self.config.learning_rate_scheduler == 'cosine_annealing':
            lr_sched_method = cosine_annealing
        elif self.config.learning_rate_scheduler == 'cosine_annealing_linear':
            lr_sched_method = cosine_annealing_linear
        else:
            lr_sched_method = no_lr_sched
            
        lr_sched = LearningRateScheduler(lr_sched_method, verbose = self.config.verbose)

        self.history = self.model.fit(x_data['train'], y_data['train'],
                batch_size = self.config.batch_size, 
                epochs = self.config.max_epochs, 
                shuffle = self.config.shuffle,
                validation_data = (x_data['validation'], y_data['validation']),  
                callbacks = [es, tnan, mc, lr_sched], 
                verbose = self.config.verbose)

        # load the saved model
        #saved_model = load_model(full_model_name) 
        setattr(self, 'model', self.model)


    def evaluate_model(self, x_sets, y_sets):
        y_hat = dict()
        predProb = dict()
        accuracy= dict()
        auc = dict()
        for dataset, datavalues in x_sets.items():
            # predict output and predicted probability of positive class
            predProb[dataset] = self.model.predict(datavalues)
            y_hat[dataset] = np.array([int(x > 0.5) for x in predProb[dataset]])
        
            # log accuracy and auc in dict
            accuracy[dataset] = accuracy_score(y_sets[dataset], y_hat[dataset])
            auc[dataset] = roc_auc_score(y_sets[dataset], predProb[dataset])
        
        return accuracy, auc, y_hat, predProb


    def save_model(self, file_name):
        full_file_name = os.path.join(self.config.current_dir, 'Models', file_name)
        self.model.save(full_file_name)


    def load_model_sets(self, file_name):
        full_file_name = os.path.join(self.config.current_dir, 'Models', file_name)
        self.model = load_model(full_file_name)
    
    
    def plot_model(self):
        '''Save image of model architecture'''
        plot_model(self.model, to_file = os.path.join(self.config.current_dir, 'Images', 'NN_model.png'), show_shapes = True, show_layer_names = True)


    def save_model_summary(self):
        ''' Print model summary to file '''
        with open(os.path.join(self.config.current_dir, 'Results', 'modelsummary.txt'),'w+') as f:
            self.model.summary(print_fn = lambda x: f.write(x + '\n'))

