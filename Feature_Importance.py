# Import libraries
import math
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
col_names = data.columns[1:]

# make logistic regression model
logReg_model = LogisticRegression(random_state = RANDOM_STATE)
logReg_model.fit(data, target)

y_hat = logReg_model.predict(data)

accuracy = accuracy_score(target, y_hat)

coef_values = logReg_model.coef_[0]
odds_ratio = [math.exp(x) for x in coef_values]
odds_change_heart_disease = odds_change_heart_disease = [x - 1 for x in odds_ratio]

# Plot coefficient values
fig = plt.figure(figsize=(15, 12))
plt.subplots_adjust(hspace = 0.5)
plt.suptitle("Logistic Regression Coefficients", fontsize=18, y=0.95)

# log-odds ratio (raw coefficients)
ax = plt.subplot(1, 3, 1)
plt.bar(col_names, coef_values[1:])
ax.set_xlabel('Feature')
ax.set_ylabel('Log Odds Ratio')
ax.set_title('Logistic Regression Coefficients')

ax = plt.subplot(1, 3, 2)
plt.bar(col_names, odds_ratio[1:])
ax.set_xlabel('Feature')
ax.set_ylabel('Odds Ratio')
ax.set_title('Logistic Regression - Odds Ratio')

ax = plt.subplot(1, 3, 3)
plt.bar(col_names, odds_change_heart_disease[1:])
ax.set_xlabel('Feature')
ax.set_ylabel('Change in Odds Ratio')
ax.set_title('Logistic Regression - Change in Odds Ratio')

plt.savefig(os.path.join('Images', 'Logistic_Regression_LogOddsRatio.png'))

# Plot change in odds by itself
fig, ax = plt.subplots(figsize=(12,6))
plt.bar(col_names, odds_change_heart_disease[1:])
ax.set_xlabel('Feature')
ax.set_ylabel('Change in Odds Ratio')
ax.set_title('Logistic Regression - Change in Odds Ratio')
fig.savefig(os.path.join('Images', 'Change_in_Odds_Ratio.png'))
