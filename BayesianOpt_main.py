"""
Performs Bayesian Regression to find the optimal hyperparameters for a
feed forward fully connected neural network.

It includes methods to define, configure, train, and evaluate the model. 

Created 10 Mar 2022

@author: Brian Wade
@version: 1.0
"""

import sys
import os
from datetime import datetime
import numpy as np
import random
import tensorflow as tf
import warnings

# Functions in other scripts of this repo
from Tools.get_configuration import ConfigurationParameters
from Tools.Get_and_prepare_data import get_and_prepare_data, createScaler_and_scale_data
from Tools.Evaluate_model import calculate_results, record_results, save_model, clear_previous_results
from Tools.BayesianOpt import BayesianTuning


config_file = 'NN_BayesianOpt_config.ini'


def calculate_time_to_complete(startTime):
    delta_time = {}
    time_delta = datetime.now() - startTime
    delta_hour = time_delta.seconds//3600
    delta_min = ((time_delta.seconds - (delta_hour * 3600))//60)
    delta_sec = (time_delta.seconds - delta_hour*3600 - delta_min * 60)%60
    print('########################################################')
    print('Computations Complete')
    print(f'Time to complete analysis: {delta_hour} hours, {delta_min} minutes, {delta_sec} seconds')
    print('########################################################')


def set_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def main():
    startTime = datetime.now()

    config = ConfigurationParameters(config_file)

    set_random_seeds(config.seed)
    
    x_sets, y_sets, col_names = get_and_prepare_data(data_file = config.data_file, train_size = config.training_size, val_size = config.val_size, random_state = config.seed)

    x_sets_scaled = createScaler_and_scale_data(x_sets['train'], x_sets['validation'], x_sets['test'], save_scaler = True)
    
    tuner = BayesianTuning(config, x_sets_scaled, y_sets)
    tuner.setup_hyperparameter_ranges()
    tuner.do_optimization()

    tuner.save_best_parameters_to_csv()
    tuner.plot_optimization_convergence()
    tuner.plot_optimization_objective_space()   

    calculate_time_to_complete(startTime)


if __name__ == "__main__":
    main()
