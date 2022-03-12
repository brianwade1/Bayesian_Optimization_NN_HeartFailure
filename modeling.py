# Import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

# Set Constants
RANDOM_STATE = 42

# Read Data
data_path = os.path.join('Data', 'heart_cleaned.csv')
df = pd.read_csv(data_path)

# Separate data and target
target = df['HeartDisease']
data = df.copy().drop('HeartDisease', axis = 1)
data.head()

# Split data into train, validation, test
x_train, x_val_test, y_train, y_val_test = train_test_split(data, target, test_size = 0.2, random_state = RANDOM_STATE)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size = 0.5, random_state = RANDOM_STATE)

x_sets = {'train': x_train, 'validation': x_val, 'test': x_test}
y_sets = {'train': y_train, 'validation': y_val, 'test': y_test}

######################
## Logistic Regression
######################
# make logistic regression model
logReg_model = LogisticRegression(random_state = RANDOM_STATE)
logReg_model.fit(x_train, y_train)

# predict and evaluate with logistic regression model
y_hat_logReg = dict()
accuracy_logReg = dict()
auc_logReg = dict()
for dataset, datavalues in x_sets.items():
    y_hat_logReg[dataset] = logReg_model.predict(datavalues)
    auc_logReg[dataset] = accuracy_score(y_sets[dataset], y_hat_logReg[dataset])

coef_values = logReg_model.coef_[0]