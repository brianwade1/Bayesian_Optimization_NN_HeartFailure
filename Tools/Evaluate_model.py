# Import libraries
import numpy as np 
from sklearn.metrics import accuracy_score, roc_auc_score

import os
import pickle
import glob
import joblib



def clear_previous_results():
    # Remove previous results
    files = glob.glob('Results/*')
    for f in files:
        os.remove(f)


# Record results function
def calculate_results(model, x_sets, y_sets, NN_model = False):
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
    return accuracy, auc, y_hat, predProb


def record_results(method_name, accuracy, auc, results_filename):
    # save results
    with open(results_filename, 'a') as f:
        f.write(f'--{method_name}--' + '\n')
        f.write('Accuracy' + '\n')
        for key, value in accuracy.items():
            f.write(str(key) + ' = ' + str(value) + '\n')
        f.write('AUC' + '\n')
        for key, value in auc.items():
            f.write(str(key) + ' = ' + str(value) + '\n')
        f.write('\n')


def save_predicted_results(method_name, y_hat, predProb):
    # save predicted outputs
    for dataset in y_hat:
        yhat_header = f"{method_name}_{dataset}_yhat_values"
        predProb_header = f"{method_name}_{dataset}_predicted_probs"
        yhat_filename = os.path.join('Results', yhat_header + ".csv")
        predProb_filename = os.path.join('Results', predProb_header + ".csv")
        np.savetxt(yhat_filename, y_hat[dataset], delimiter = ",", header = yhat_header)
        np.savetxt(predProb_filename, predProb[dataset], delimiter = ",", header = predProb_header)


def save_model(model, method_name):
    # save model# save logistic regression model
    savemodel_filename = os.path.join('Models', f'{method_name}_model.save')
    joblib.dump(model, savemodel_filename) 