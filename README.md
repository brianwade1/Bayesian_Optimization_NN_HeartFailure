# Bayesian Optimization of Neural Net Model for Predictive Modeling for Heart Failure

This project predicts the likelihood for heart failure. The project takes place in three parts: exploratory data analysis (EDA) and data preparation, the creation of three initial binary classification models including logistic regression, random forests, and a neural network. Then, the hyperparameters of the neural net were optimized using Bayesian Optimization. 

---

## Folders and Files

This repo contains the following folders and files:

Folders:

* [Config](Config) : configuration file (.ini file) to specify the hyperparameters for the neural net trained in [Model_Development.py](Model_Development.py) and the hyperparameters for the Bayesian Optimization [BayesianOpt_main](BayesianOpt_main.py)
 

* [Data](Data) : Raw data and description
  * heart_raw.csv - Raw data from [kaggle website](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?select=heart.csv)

  * heart_cleaned.csv - transformed data exported at the end of the [EDA.ipynb](EDA.ipynb) file.

* [Images](Images): Images produced from the [EDA.ipynb](EDA.ipynb) and [Feature_Importance.py](Feature_Importance.py) file. These imaged are used in this readme file.

* [Models](Models): Saved trained models created in [Model_Development.py](Model_Development.py) file

* [Results](Results): Results of the trained models created in [Model_Development.py](Model_Development.py) and [BayesianOpt_main](BayesianOpt_main.py) files.

* [Tools](Tools): Supporting Scripts for the [Model_Development.py](Model_Development.py) and[BayesianOpt_main](BayesianOpt_main.py) files.

* [BayesianOpt.py](BayesianOpt.py) - Bayesian Optimization class which is called by the [BayesianOpt_main](BayesianOpt_main.py)

* [Evaluate_model.py](Evaluate_model.py) - function script which calculates the accuracy and area under the ROC curve (AUC) for the train, validation, and test data sets. It also saves the evaluation results to a csv file. 

* [Get_and_prepare_data.py](Get_and_prepare_data.py) - function script which reads the data csv file, divides the data into a train, validation, and test sets, and scales the data for modeling in the neural nets. 

* [get_configuration.py](get_configuration.py) - function script which reads and stores the settings in the [NN_BayesianOpt_config.ini](NN_BayesianOpt_config.ini) file.

* [Neural_Net_Model.py](Neural_Net_Model.py) - fully connected feed forward neural network class which contains methods to make network architecture, train, save, load, and evaluate the model.
 

Main Files:
* [BayesianOpt_main.py](BayesianOpt_main.py) - Main script to conduct a Bayesian Optimization of the hyperparameters for the neural net classifier.

* [EDA.ipynb](EDA.ipynb) - Exploratory Data Analysis and data preparation.

* [Feature_Importance.py](Feature_Importance.py) - Logistic regression model to explore influence of each feature (main effects only) and how each predictor feature changes the odds ratio of heart failure.

* [Model_Development.py](Model_Development.py) - development and evaluation of three independent prediction models: logistic regression, random forest classifier, and neural net classifier.

* [environment.yml](environment.yml) and [requirements.txt](requirements.txt)- python dependencies to recreate the virtual environment from [conda](https://docs.conda.io/en/latest/) or [pip](https://pypi.org/project/pip/).

---

## Data Source and Data Description
Data is from the following kaggle competition: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?select=heart.csv). The data includes 918 observations with 11 predictor features and 1 binary target features. These are described below:

### Predictor Features
1. Age: age of the patient [years]
2. Sex: sex of the patient [M: Male, F: Female]
3. ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
4. RestingBP: resting blood pressure [mm Hg]
5. Cholesterol: serum cholesterol [mm/dl]
6. FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
7. RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
8. MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
9. ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
10. Oldpeak: oldpeak = ST [Numeric value measured in depression]
11. ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]

### Target Feature:
1. HeartDisease: output class [1: heart disease, 0: Normal]

---

## Data Exploration and Understanding

The first step in the analysis was to explore the data for any initial insights using the [EDA.ipynb](EDA.ipynb). 

### Data Exploration
This allowed the author to ensure that there were no missing values, duplicate rows, or obvious outliers in the data. 

Next, a scatterplot matrix showed that all features contained some degree of information to predict heart failure, but many were correlated.

![EDA_ScatterplotMatrix](/Images/EDA_ScatterplotMatrix.png)

The correlation is confirmed with a correlation matrix below. Here we see that all predictors are moderately correlated with the target (heart disease) except for RestingECG and RestingBP.

![Pearson_Correlation](/Images/Pearson_Correlation.png)

### Cardinality of Categorical Features
Another critical step in data preparation is to convert categorical data to numeric through either one-hot-encoding or ordinal encoding. Below is the cardinality of each categorical feature.

![EDA_Cardinality](/Images/EDA_Cardinality.png)

All features have less than 4 distinct groups, so we don't have to worry about high cardinality. We will one-hot-encode Sex and ExerciseAngina because there is not a nature order. The others will be ordinal encoded.

### Outlier Detection
Outliers in the samples are not necessarily bad. However, if they are false readings, they will skew the data. Below is a box plot of the features.

![OutlierDetection](/Images/OutlierDetection.png)

From the boxplot, we see that there are samples where the resting BP was 0 or the cholesterol was 0. These are likely erroneous readings, so they were dropped from the dataset.

### Feature Importance
In order to assess the relation between the predictor features and heart disease, we remove features that are strongly correlated with each other to minimize the effect of multicollinearity. To do this we look at the Variance Inflation Factor (VIF).

| Feature | VIF |
| ------ | --------- |
| Age | 32.473409 |
| Sex | 4.462562 |
| ChestPainType | 2.317030 |
| RestingBP | 54.723330 |
| Cholesterol | 17.722887 |
| FastingBS | 1.313163 |
| RestingECG | 3.196918 |
| MaxHR | 30.455911 |
| ExerciseAngina | 2.787416 |
| Oldpeak | 3.074552 |
| ST_Slope | 12.770715 |
| HeartDisease | 4.007046 |

Normally VIF values greater than 10 or 20 can be problematic. Here, we see RestingBP, Age, and MaxHR are the largest VIF values. From the correlation matrix, we know that MaxHR is not well correlated with the target. We can also see that age also has a lower correlation. Therefore, **for the feature importance calculations only ([Feature_Importance.py](Feature_Importance.py)), we will drop MaxHR and Age.** Note, we will keep these in the prediction scripts ([Model_Development.py](Model_Development.py)) since they still contain useful information for the prediction. Once MaxHR and Age are dropped from the dataframe, the resulting VIF values are all less than 20. MaxHR still has a high VIF value, but because it is strongly correlated with the target, we will keep it in the dataset.

| Feature | VIF |
| ------ | --------- |
| Sex | 4.378796 |
| ChestPainType | 2.269470 |
| Cholesterol | 14.933645 |
| FastingBS | 1.261079 |
| RestingECG | 3.071725 |
| MaxHR | 23.940939 |
| ExerciseAngina | 2.715001 |
| Oldpeak | 2.832941 |
| ST_Slope | 11.644549 |
| HeartDisease | 3.776834 |

Using only the reduced dataframe (without MaxHR and Age), we build a logistic regression model and plot the coefficients (log odds ratios) and the transformed odds ratios. See [Feature_Importance.py](Feature_Importance.py) script.

![Logistic_Regression_LogOddsRatio](/Images/Logistic_Regression_LogOddsRatio.png)

The first plot shows the logistic regression coefficient values which are the log of the odds ratios. The middle graph transforms the coefficients into the odds ratios. This shows how an increase of one unit of a feature changes the odds of heart failure. Finally, the right graph shows the change in the odds ratio by subtracting 1 (100%) from the middle graph. This is a more interpretable view of the odds. For example, the first feature in each graph is chest pain. 

### Balance of target class
An imbalance in the target class can cause issues with classifiers and the use of the accuracy metric. Below is the number of samples in each target class.  These appear to be fairly balanced.

![EDA_TargetCounts](/Images/EDA_TargetCounts.png)


---

## Initial Modeling

Three initial predictive models were tested: logistic regression, random forest, and neural networks. Before modeling, the data was split into a train, validation, and test set which was 80%, 10%, and 10% of the full dataset respectively. All models were trained on the training set and evaluated on the validation and test sets. The logistic regression and random forest were trained on unscaled data. For the neural network, the data was scaled with a standard scaler (scale each feature to a mean of 0 and 1 standard deviation) which was fit to the training set and applied to teh validation and test sets.

### Logistic Regression

The logistic regression was a main effects only model with 5,000 iterations to ensure convergence.

### Random Forest
The random forest included 100 tree estimators.

### Neural Network
The neural network was trained with a binary cross entropy loss function and included early stopping when the validation set's loss increased for five epochs.

Architecture Hyperparameters
* Number of hidden layers: 2
* Number of nodes in first hidden layer: 15
* Number of nodes in second hidden layer: 5
* Relu activation functions in the input and hidden layers
* Sigmoid action function in the output layer

Training Hyperparameters
* Max Epochs: 200
* Validation Patience: 5
* Batch Size: 8
* Learning Rate: 0.001
* Optimizer: Adam

---

## Initial Results

The results of the fitting process are shown below.

### **Logistic Regression**
| Metric | Train Set| Validation Set | Test Set |
| ------ | --------- | --------- | --------- |
| Accuracy | 0.8356 | 0.8133 | 0.8933| 
| AUC | 0.91890 | 0.9040 | 0.9613 |


### **Random Forest**
| Metric | Train Set| Validation Set | Test Set |
| ------ | --------- | --------- | --------- |
| Accuracy | 1.000  | 0.8933 | 0.9200 | 
| AUC | 1.000 | 0.9378 | 0.9878 |

### **Neural Net**
| Metric | Train Set| Validation Set | Test Set |
| ------ | --------- | --------- | --------- |
| Accuracy | 0.8859  | 0.8267 | 0.9067 | 
| AUC | 0.9534 | 0.9331 | 0.9806 |

In this example, the random forest was the best performing model followed very closely by the neural network. 

---
## Bayesian Optimization of Neural Network Hyperparameter Results

The results of the optimization of the neural network hyperparamters are show below for 150 optimization iterations.

![Bayesian_Optimization_Convergence_Plot](/Images/Bayesian_Optimization_Convergence_Plot.png)

The best parameters for the neural network were:

| Parameter | Value |
| ------ | --------- |
| learning rate | 0.0098 |
| Number of Hidden Layers | 2 |
| Layer1 Nodes | 18 |
| Layer2 Nodes | 18 |
| Batch Size | 31 |
| Training Function | adam |
| learnign rate decay rate| 0.5607 |
| learning rate scheduler | linear_lr_dec |

These resulted in the below neural net

![NN_model](/Images/NN_model.png)

The hyperparamters interaction chart shows how changes in the hyperparameters effect the accuracy.

![Bayesian_Optimization_Objective_Plot](/Images/Bayesian_Optimization_Objective_Plot.png)

After optimization, the best neural net model produced the following results. These were close to the random forest results but generally not much better.
### **Neural Net**
| Metric | Train Set| Validation Set | Test Set |
| ------ | --------- | --------- | --------- |
| Accuracy | 0.8860  | 0.9054 | 0.8816 | 
| AUC | 0.9641 | 0.9496 | 0.9783 |




---
## References

1. fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved [Date Retrieved] from https://www.kaggle.com/fedesoriano/heart-failure-prediction.