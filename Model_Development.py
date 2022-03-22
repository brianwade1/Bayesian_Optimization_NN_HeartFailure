# Import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import os
import glob
import pickle
import joblib

# Functions in other scripts of this repo
from Tools.Get_and_prepare_data import get_and_prepare_data, createScaler_and_scale_data
from Tools.Evaluate_model import calculate_results, record_results, save_model, clear_previous_results



def do_LogisticRegression(x, y, max_iter = 5000, random_state = 42):
    # make logistic regression model
    logReg_model = LogisticRegression(random_state = random_state, max_iter = max_iter)
    logReg_model.fit(x, y)
    return logReg_model
  

def do_RandomForest(x, y, n_estimators = 100, random_state = 42):
    # make random forest classification model
    rf_model = RandomForestClassifier(n_estimators = n_estimators, random_state = RANDOM_STATE)
    rf_model.fit(x, y)
    return rf_model


def make_NN_model(hidden_layer_sizes, input_dim, output_dim):
    ''' Make the Neural Net architecture'''
    # Make NN model
    model = Sequential()
    # Input and hidden layers
    for layer_num, layer_size in enumerate(hidden_layer_sizes):
        if layer_num == 0:
            model.add(Dense(layer_size, input_dim = input_dim, activation = 'relu')) 
        else:
            model.add(Dense(layer_size, activation = 'relu'))

    # binary classifier so use sigmoid at output layer
    model.add(Dense(output_dim, activation = 'sigmoid'))
    return model

    
def train_NN_model(model, x_train, y_train, x_val, y_val, learning_rate, epoch, batch_size, loss, metrics, patience):  
    #### Compile and train the network ####
    optimizer1 = keras.optimizers.Adam(lr = learning_rate)
    model.compile(optimizer = optimizer1, loss = loss, metrics = metrics)

    #keras.optimizers.SGD(lr = 0.99, momentum = 0.99,  nesterov = True) 
    #model.compile(loss = loss, optimizer = 'SGD', metrics = metrics)

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 0, patience = 5)
    model_name = os.path.join('Models', 'best_NN_model.save')
    mc = ModelCheckpoint(model_name, monitor = 'val_loss', mode = 'min', verbose = 0, save_best_only = True)
    model.fit(x_train, y_train, batch_size = batch_size, epochs = epoch, validation_data = (x_val, y_val), callbacks=[es, mc])

    # load the saved model
    saved_model = load_model(model_name) 
    return saved_model   

    
if __name__ == "__main__":
    # Set Constants
    RANDOM_STATE = 42
    TRAIN_SIZE = 0.8
    VAL_SIZE = 0.1
    RESULTS_FILENAME = os.path.join('Results', 'model_development_results.csv')
    DATA_FILE = 'heart_cleaned.csv'

    clear_previous_results()
    x_sets, y_sets, col_names = get_and_prepare_data(data_file = DATA_FILE, train_size = TRAIN_SIZE, val_size = VAL_SIZE, random_state = RANDOM_STATE)

    ######################
    # Logistic Regression
    ######################
    logReg_model = do_LogisticRegression(x_sets['train'], y_sets['train'], max_iter = 5000, random_state = RANDOM_STATE)
    # save logReg model results
    log_reg_name = 'Logistic_Regression'
    logReg_accuracy, logReg_auc, logReg_y_hat, logReg_predProb = calculate_results(logReg_model, x_sets, y_sets, NN_model = False)
    record_results(log_reg_name, logReg_accuracy, logReg_auc, RESULTS_FILENAME)
    #save_predicted_results(log_reg_name, logReg_y_hat, logReg_predProb)
    save_model(logReg_model, log_reg_name)

    ######################
    # Random Forest
    ######################
    rf_model = do_RandomForest(x_sets['train'], y_sets['train'], n_estimators = 100, random_state = RANDOM_STATE)
    # save RF model results
    rf_name = 'Random_Forest'
    rf_accuracy, rf_auc, rf_y_hat, rf_predProb = calculate_results(rf_model, x_sets, y_sets, NN_model = False)
    record_results(rf_name, rf_accuracy, rf_auc, RESULTS_FILENAME)
    #save_predicted_results(rf_name, rf_y_hat, rf_predProb)
    save_model(rf_model, rf_name)


    ######################
    # Neural Net - New framework
    ######################
    # Functions in other scripts of this repo
    from Tools.get_configuration import ConfigurationParameters
    from Tools.Get_and_prepare_data import get_and_prepare_data, createScaler_and_scale_data
    from Tools.Evaluate_model import calculate_results, record_results, save_model, clear_previous_results
    from Tools.Neural_Net_Model import NN_Model
    import random

    config_file = 'NN_BayesianOpt_config.ini'

    config = ConfigurationParameters(config_file)

    def set_random_seeds(seed):
        os.environ['PYTHONHASHSEED']=str(seed)
        np.random.seed(seed)
        random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

    set_random_seeds(config.seed)
    
    x_sets, y_sets, col_names = get_and_prepare_data(data_file = config.data_file, train_size = config.training_size, val_size = config.val_size, random_state = config.seed)

    x_sets_scaled = createScaler_and_scale_data(x_sets['train'], x_sets['validation'], x_sets['test'], save_scaler = True)

    # Make and train model
    input_dim = x_sets_scaled['train'].shape[1]
    output_dim = 1
    nn_model = NN_Model(config, input_dim, output_dim)
    nn_model.make_model()  
    nn_model.train_model(x_sets_scaled, y_sets, 'Neural_Net_newFW')
    accuracy, auc, y_hat, predProb = nn_model.evaluate_model(x_sets_scaled, y_sets)

    print(accuracy)

  