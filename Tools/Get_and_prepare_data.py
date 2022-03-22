# Import libraries
import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import os
import joblib


def get_and_prepare_data(data_file, train_size, val_size, random_state):
    # Read Data
    data_path = os.path.join('Data', data_file)
    df = pd.read_csv(data_path)

    # Separate data and target
    target = df['HeartDisease']
    data = df.copy().drop('HeartDisease', axis = 1)
    col_names = data.columns[1:]

    # Split data into train, validation, test
    testval_size = 1 - train_size
    test_size = val_size / testval_size
    x_train, x_val_test, y_train, y_val_test = train_test_split(data, target, test_size = testval_size, random_state = random_state)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size = test_size, random_state = random_state)

    x_sets = {'train': x_train, 'validation': x_val, 'test': x_test}
    y_sets = {'train': y_train, 'validation': y_val, 'test': y_test}
    return x_sets, y_sets, col_names


def createScaler_and_scale_data(x_train, x_val, x_test, save_scaler):
    # Neural nets need to scaled data - fit to train, apply to val and test.
    std_scaler = preprocessing.StandardScaler()
    x_train_scaled = std_scaler.fit_transform(x_train)
    x_val_scaled = std_scaler.transform(x_val)
    x_test_scaled = std_scaler.transform(x_test)

    if save_scaler:
        # Save scaler info for later deployment
        scaler_filename = os.path.join('Models', 'std_scaler.save')
        joblib.dump(std_scaler, scaler_filename) 
    
    x_sets_scaled = {'train': x_train_scaled, 'validation': x_val_scaled, 'test': x_test_scaled}
    return x_sets_scaled


def scale_data(scaler, x_data):
    ''' Apply an already created scaler '''
    x_sets_scaled = dict()
    for dataset, datavalues in x_sets.items():
        x_sets_scaled[dataset] = scaler.transform(datavalues)
    return x_sets_scaled