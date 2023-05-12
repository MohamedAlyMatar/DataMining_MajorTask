import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from My_Cross_Validation import *

 
# reading data from csv file
dataset = pd.read_csv('../dataset/cleaned_dataset.csv')

x_train = dataset.drop('CLASS', axis=1)
y_train = dataset['CLASS']

#---------------------------------- prepare your dataset -------------------------------------------#

# Define the model and hyperparameter grid

# model = SVC()
# param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}

model = DecisionTreeClassifier()
param_grid = {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]}

# define the folds/splits
outer_cv = KFold(n_splits=9, shuffle=True, random_state=42)
# inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

#-----------------------------------------------------------------------------------------------------#

# call our custom function
cv_scores, avg_score, best_train_x, best_train_y, best_test_x, best_test_y = nested_cross_validation(x_train, y_train, model, param_grid, outer_cv)
print(cv_scores, avg_score)
print("Best training data set:\n", best_train_x)
print("Best training target set:\n", best_train_y)
print("Best testing data set:\n", best_test_x)
print("Best testing target set:\n", best_test_y)

#-----------------------------------------------------------------------------------------------------#
