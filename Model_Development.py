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
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import os
import glob
import pickle
import joblib

# Set Constants
RANDOM_STATE = 42
RESULTS_FILENAME = os.path.join('Results', 'model_fit_results.csv')
SAVE_PREDICTED_RESULTS = False

# Remove previous results
files = glob.glob('Results/*')
for f in files:
    os.remove(f)

# Read Data
data_path = os.path.join('Data', 'heart_cleaned.csv')
df = pd.read_csv(data_path)

# Separate data and target
target = df['HeartDisease']
data = df.copy().drop('HeartDisease', axis = 1)
col_names = data.columns[1:]

# Split data into train, validation, test
x_train, x_val_test, y_train, y_val_test = train_test_split(data, target, test_size = 0.2, random_state = RANDOM_STATE)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size = 0.5, random_state = RANDOM_STATE)

x_sets = {'train': x_train, 'validation': x_val, 'test': x_test}
y_sets = {'train': y_train, 'validation': y_val, 'test': y_test}

# Record results function
def record_results(model, method_name, x_sets, y_sets, NN_model = False, save_predicted_results = False):
    y_hat = dict()
    predProb = dict()
    accuracy= dict()
    auc = dict()
    for dataset, datavalues in x_sets.items():
        if NN_model:
            # predict output and predicted probability of positive class
            predProb[dataset] = model.predict(datavalues)
            y_hat[dataset] = np.array([int(x > 0.5) for x in predProb[dataset]])
        else:
            # predict output and predicted probability of positive class
            y_hat[dataset] = model.predict(datavalues)
            predProb[dataset] = model.predict_proba(datavalues)[:, 1]

        # log accuracy and auc in dict
        accuracy[dataset] = accuracy_score(y_sets[dataset], y_hat[dataset])
        auc[dataset] = roc_auc_score(y_sets[dataset], predProb[dataset])

    # save results
    with open(RESULTS_FILENAME, 'a') as f:
        f.write(f'--{method_name}--' + '\n')
        f.write('Accuracy' + '\n')
        for key, value in accuracy.items():
            f.write(str(key) + ' = ' + str(value) + '\n')
        f.write('AUC' + '\n')
        for key, value in auc.items():
            f.write(str(key) + ' = ' + str(value) + '\n')
        f.write('\n')

    # save predicted outputs
    if save_predicted_results:
        for dataset in y_hat:
            yhat_header = f"{method_name}_{dataset}_yhat_values"
            predProb_header = f"{method_name}_{dataset}_predicted_probs"
            yhat_filename = os.path.join('Results', yhat_header + ".csv")
            predProb_filename = os.path.join('Results', predProb_header + ".csv")
            np.savetxt(yhat_filename, y_hat[dataset], delimiter = ",", header = yhat_header)
            np.savetxt(predProb_filename, predProb[dataset], delimiter = ",", header = predProb_header)

    # save model# save logistic regression model
    savemodel_filename = os.path.join('Models', f'{method_name}_model.save')
    joblib.dump(model, savemodel_filename) 


######################
## Logistic Regression
######################
# make logistic regression model
logReg_model = LogisticRegression(random_state = RANDOM_STATE, max_iter = 5000)
logReg_model.fit(x_train, y_train)

# save model results
record_results(logReg_model, 'Logistic_Regression', x_sets, y_sets, NN_model = False, save_predicted_results = SAVE_PREDICTED_RESULTS)


######################
##  Random Forest
######################
# make random forest classification model
rf_model = RandomForestClassifier(n_estimators = 100, random_state = RANDOM_STATE)
rf_model.fit(x_train, y_train)

# save model results
record_results(rf_model, 'Random_Forest', x_sets, y_sets, NN_model = False, save_predicted_results = SAVE_PREDICTED_RESULTS)


######################
##  Neural Net
######################
# make Neural Net classification model

# Training Hyperparameters
epoch = 200 #How many times to iterate over the training data.
batch_size = 8
learning_rate = 0.001
loss = 'binary_crossentropy'
metrics=['accuracy']

# Architecture Hyperparameters:
InputLayer = 25
HL1 = 15
HL2 = 5

# Neural nets need to scaled data - fit to train, apply to val and test.
std_scaler = preprocessing.StandardScaler()
x_train_scaled = std_scaler.fit_transform(x_train)
x_val_scaled = std_scaler.transform(x_val)
x_test_scaled = std_scaler.transform(x_test)

x_sets_scaled = {'train': x_train_scaled, 'validation': x_val_scaled, 'test': x_test_scaled}

# Save scaler info for later deployment
scaler_filename = os.path.join('Models', 'std_scaler.save')
joblib.dump(std_scaler, scaler_filename) 

# Make NN model
model = Sequential()
input_dim = x_train_scaled.shape[1]
out_dim = 1

# Input and hidden layers
model.add(Dense(InputLayer, input_dim = input_dim, activation = 'elu')) 
model.add(Dense(HL1, activation = 'relu'))
model.add(Dense(HL2, activation = 'relu'))
model.add(Dense(out_dim, activation = 'sigmoid'))
    
#### Compile and train the network ####
optimizer1 = keras.optimizers.Adam(lr = learning_rate)
model.compile(optimizer = optimizer1, loss = loss, metrics = metrics)

#keras.optimizers.SGD(lr = 0.99, momentum = 0.99,  nesterov = True) 
#model.compile(loss = loss, optimizer = 'SGD', metrics = metrics)

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 0, patience = 5)
model_name = os.path.join('Models', 'best_NN_model.save')
mc = ModelCheckpoint(model_name, monitor = 'val_loss', mode = 'min', verbose = 0, save_best_only = True)
model.fit(x_train_scaled, y_train, batch_size = batch_size, epochs = epoch, validation_data = (x_val_scaled, y_val), callbacks=[es, mc])

# load the saved model
saved_model = load_model(model_name)    

# save model results
record_results(saved_model, 'Neural_Net', x_sets_scaled, y_sets, NN_model = True, save_predicted_results = SAVE_PREDICTED_RESULTS)
