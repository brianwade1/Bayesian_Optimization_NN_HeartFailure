# Import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# display all of the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns',None)

%matplotlib inline

# Read Data
data_path = os.path.join('Data', 'heart.csv')
df = pd.read_csv(data_path)

# Separate data and target
target = df['HeartDisease']
data = df.copy().drop('HeartDisease', axis = 1)
data.head()

# Split data into train, validation, test
x_train, x_val_test, y_train, y_val_test = train_test_split(data, target, test_size = 0.2, random_state = RANDOM_STATE)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size = 0.5, random_state = RANDOM_STATE)

