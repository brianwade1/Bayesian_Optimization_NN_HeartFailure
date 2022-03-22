"""
Performs Bayesian Regression to find the optimal hyperparameters for a
a neural network.

It includes methods to define, configure, train, and evaluate the model. 

Created 10 Mar 2022

@author: Brian Wade
@version: 1.0
"""
# Import libraries
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import skopt
from skopt import gbrt_minimize, gp_minimize, load
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer  
from skopt.plots import plot_convergence, plot_objective
from skopt.callbacks import CheckpointSaver

import os
import glob
import pickle
import joblib
import csv
import sys
import gc

# Functions in other scripts of this repo
from Tools.Get_and_prepare_data import get_and_prepare_data, createScaler_and_scale_data
from Tools.Evaluate_model import calculate_results, record_results, save_model
from Tools.Neural_Net_Model import NN_Model


#config_file = 'sat_maneuver_config.ini'
#model_name = 'best_model'

RESULTS_FILENAME = os.path.join('Results', 'BayesianOpt_Model_results.csv')


class light_weight_CheckpointSaver:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.x_iters = None
        self.func_vals = None
        self.num_in_set = None


    def __call__(self, gp_min_object):
        path = self.checkpoint_path
        self.x_iters = gp_min_object.x_iters # list of list of inputs for each iteration
        self.func_vals = gp_min_object.func_vals # list of output values for each iteration
        self.num_in_set = (len(self.func_vals))

        with open(self.checkpoint_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.num_in_set])
            writer.writerows(self.x_iters)
            writer.writerow(self.func_vals)

        print('check point write complete')


    def get_previous_runs(self):
        data = []
        with open(self.checkpoint_path, 'r') as f:
            for line in f:
                data.append(line)

        data = [line.rstrip('\n') for line in data]
        self.num_in_set = int(data[0])
        self.func_vals = [float(i) for i in data[-1].split(',')]

        self.x_iters = []
        for data_set in range(self.num_in_set):
            str_list = data[data_set+1].split(',')
            
            line = []
            for str_element in str_list:
                try:
                    element = int(str_element)
                except:
                    try:
                        element = float(str_element)
                    except:
                        element = str_element
                line.append(element)
            
            self.x_iters.append(line)

        return self
            

class BayesianTuning:
    def __init__(self, config, x_data, y_data):
        self.config = config
        self.x_data = x_data
        self.y_data = y_data
        self.iteration = 0
        self.interim_model_name = 'interim_model.h5' 
        self.best_model_name = 'best_model.save'
        self.best_metric = np.inf
        self.best_config = None
        self.best_model = None
        self.check_point_file = 'gp_min_checkpoint.txt'
        self.checkpoint_path = os.path.join(self.config.current_dir, 'Results', self.check_point_file)

        self.input_dim = self.x_data['train'].shape[1]
        self.output_dim = 1 #binary classifier

        # Check to see if a checkpoint already exists
        self.checkpoint_exists = os.path.exists(self.checkpoint_path)


    def setup_hyperparameter_ranges(self):
        #### Setup Optimization Ranges ####
        dim_learning_rate = Real(low = self.config.lr_low, high = self.config.lr_high, prior = 'log-uniform', name = 'learning_rate')
        dim_num_hidden_layers = Integer(low = self.config.num_hiddenlayers_low, high = self.config.num_hiddenlayers_high, name = 'num_hiddenlayers')
        dim_num_Layer1_nodes = Integer(low = self.config.num_Layer1_nodes_low, high = self.config.num_Layer1_nodes_high, name = 'num_Layer1_nodes')
        dim_num_Layer2_nodes = Integer(low = self.config.num_Layer2_nodes_low, high = self.config.num_Layer2_nodes_high, name = 'num_Layer2_nodes')
        dim_num_Layer3_nodes = Integer(low = self.config.num_Layer3_nodes_low, high = self.config.num_Layer3_nodes_high, name = 'num_Layer3_nodes')
        dim_batch_size = Integer(low = self.config.batch_size_low, high = self.config.batch_size_high, name = 'batch_size')
        dim_train_fun = Categorical(categories = self.config.train_fun_cats, name = 'training_function')
        dim_lr_decay = Real(low = self.config.lr_decay_low, high = self.config.lr_decay_high, name = "lr_decay")
        dim_decays_during_training = Categorical(categories = self.config.num_decays_during_training_cats, name = 'decays_during_training')
        dim_learning_rate_scheduler_cats = Categorical(categories = self.config.learning_rate_scheduler_cats, name = 'learning_rate_scheduler')

        #global dimensions
        self.dimensions = [dim_learning_rate,
                    dim_num_hidden_layers,
                    dim_num_Layer1_nodes,
                    dim_num_Layer2_nodes,
                    dim_num_Layer3_nodes,
                    dim_batch_size,
                    dim_train_fun,
                    dim_lr_decay,
                    dim_decays_during_training,
                    dim_learning_rate_scheduler_cats]

        # list of dimension labels
        self.dimension_labels = []
        for dimension in self.dimensions:
            self.dimension_labels.append(dimension.name)

        # default parameters (used on iteration 0)
        default_lr = self.config.learning_rate
        default_num_hidden_layers = len(self.config.hidden_nodes)
        # for layer_num in range(self.config.num_hiddenlayers_high):
        #     try:
        #         exec(f"default_Layer{layer_num+1}_nodes = {self.config.hidden_nodes[layer_num]}")
        #     except:
        #         exec(f"default_Layer{layer_num+1}_nodes = {self.config.hidden_nodes[0]}")
        default_Layer1_nodes = self.config.hidden_nodes[0]
        try:
            default_Layer2_nodes = self.config.hidden_nodes[1]
        except:
            default_Layer2_nodes = self.config.hidden_nodes[0]
        try:
            default_Layer3_nodes = self.config.hidden_nodes[2]
        except:
            default_Layer3_nodes = self.config.hidden_nodes[0]
        default_batch_size = self.config.batch_size
        default_train_fun = self.config.train_fun
        default_decay_rate = self.config.decay_rate
        default_num_decays_during_training = self.config.num_decays_during_training
        default_learning_rate_scheduler = self.config.learning_rate_scheduler

        self.default_parameters = [default_lr, 
                                default_num_hidden_layers, 
                                default_Layer1_nodes,
                                default_Layer2_nodes,
                                default_Layer3_nodes, 
                                default_batch_size, 
                                default_train_fun,
                                default_decay_rate,
                                default_num_decays_during_training,
                                default_learning_rate_scheduler]


    #@use_named_args(dimensions = dimensions)
    #def fitness(self, learning_rate, num_hiddenlayers, num_Layer1_nodes, num_Layer2_nodes, num_Layer3_nodes, batch_size, training_function, lr_decay):
    def fitness(self, parm_list):

        # Clear the Keras session
        keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        ops.reset_default_graph()
        gc.collect()

        model_name = 'model_' + str(self.iteration)

        inputs = {}
        for i, dimension in enumerate(self.dimension_labels):
            inputs[dimension] = parm_list[i]

        # Modify Internal Configs
        self.config.learning_rate = inputs['learning_rate']
        self.config.num_hidden_layers = inputs['num_hiddenlayers']
        hidden_node_size_list = []
        for n in range(self.config.num_hidden_layers):
            node_str = f"num_Layer{n + 1}_nodes"
            hidden_node_size_list.append(inputs[node_str])
        self.config.hidden_nodes = hidden_node_size_list
        self.config.batch_size = inputs['batch_size']
        self.config.train_fun = inputs['training_function']
        self.config.decay_rate = inputs['lr_decay']
        self.config.num_decays_during_training = inputs['decays_during_training']
        self.config.learning_rate_scheduler = inputs['learning_rate_scheduler']
        
        # Make and train model
        nn_model = NN_Model(self.config, self.input_dim, self.output_dim)
        nn_model.make_model()  
        nn_model.train_model(self.x_data, self.y_data, self.interim_model_name)
        accuracy, auc, y_hat, predProb = nn_model.evaluate_model(self.x_data, self.y_data)

        # Print the Accuracy of test.
        print()
        print(f'Accuracy train: {accuracy["train"]}, Accuracy val: {accuracy["validation"]}, Accuracy test: {accuracy["test"]}')
        print(f'AUC train: {auc["train"]}, AUC val: {auc["validation"]}, AUC test: {auc["test"]}')
        print()

        # If best model so far save it and make plots
        # Bayesian opt minimizes so we need 1 - accuracy
        if self.config.eval_metric.lower() in ['Accuracy', 'accuracy', 'acc']:
            eval_metric = - accuracy[self.config.eval_dataset]
        elif self.config.eval_metric.lower() in ['Area Under Curve', 'AUC', 'auc']:
            eval_metric = - auc[self.config.eval_dataset]
        else:
            eval_metric =  - accuracy[self.config.eval_dataset]
        
        if eval_metric < self.best_metric:
            print('saveing model and makeing plots!!!')
            print()
            
            self.best_metric = eval_metric
            
            nn_model.plot_model()
            nn_model.save_model_summary()   
 
            nn_model.save_model(self.best_model_name)
            self.best_model = nn_model
            self.best_config = self.config

            # save results
            record_results('BayesianOpt_NN', accuracy, auc, RESULTS_FILENAME)

        ## Clean Up
        # Delete the Keras model with these hyper-parameters from memory.
        del nn_model
        
        # # Clear the Keras session
        keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

        #del nn_model
    
        self.iteration += 1
        
        return eval_metric


    def do_optimization(self):
        
        #checkpoint_saver = CheckpointSaver(self.checkpoint_path, compress = True, store_objective = False) # keyword arguments will be passed to `skopt.dump`
        checkpoint_saver = light_weight_CheckpointSaver(self.checkpoint_path)

        if self.checkpoint_exists & self.config.continue_tng:
            res = checkpoint_saver.get_previous_runs()
            x0 = res.x_iters
            y0 = res.func_vals
            # graph = tf.Graph()
            # with tf.compat.v1.Session(graph=graph):
            #     #get previous results
            #     res = checkpoint_saver.get_previous_runs()
            #     #res = load(self.checkpoint_path)
            #     x0 = res.x_iters
            #     y0 = res.func_vals
        else:
            if self.checkpoint_exists:
                os.remove(self.checkpoint_path)
            x0 = self.default_parameters
            y0 = None

        self.gp_result = gp_minimize(func = self.fitness,
                            dimensions = self.dimensions,
                            n_calls = self.config.n_calls,
                            n_initial_points = self.config.n_initial_points,
                            noise = self.config.noise,
                            n_jobs = self.config.n_jobs,
                            kappa = self.config.kappa,
                            random_state = self.config.seed,
                            callback = [checkpoint_saver],
                            verbose = True,
                            x0 = x0,
                            y0 = y0)

        # Delete Intermediate Models
        os.remove(os.path.join(self.config.current_dir, 'Models', self.interim_model_name))
        
        # Remove gp_min checkpoint because we are complete.
        #os.remove(self.checkpoint_path)


    def plot_optimization_convergence(self):
        # Plot the progress of the optimization
        plot_convergence(self.gp_result)
        plt.savefig(os.path.join(self.config.current_dir, 'Images', "Bayesian_Optimization_Convergence_Plot.png"))
        plt.close()


    def plot_optimization_objective_space(self):
        plot_objective(self.gp_result, n_points = 100, dimensions = self.dimension_labels)
        plt.savefig(os.path.join(self.config.current_dir, 'Images', "Bayesian_Optimization_Objective_Plot.png"))
        plt.close()


    def save_best_parameters_to_csv(self):
        ''' Save the best NN parameters to csv '''
        self.best_set = {}
        for i, parameter in enumerate(self.dimension_labels):
            self.best_set[parameter] = self.gp_result.x[i]
            
        with open(os.path.join(self.config.current_dir, 'Results', 'Bayesian_Opt_Parameters.csv'), 'w') as NN_parameter_file:
            writer = csv.writer(NN_parameter_file, delimiter = '=')
            for name, value in self.best_set.items():
                writer.writerow([name, value])
                
