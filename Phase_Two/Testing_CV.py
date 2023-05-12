import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from My_Cross_Validation import *

 
# reading data from csv file
dataset = pd.read_csv('../dataset/cleaned_dataset.csv')

data = dataset.drop('CLASS', axis=1)
target = dataset['CLASS']

#---------------------------------- prepare your dataset -------------------------------------------#

# Define the model
model = DecisionTreeClassifier()

# define the folds/splits
folds = KFold(n_splits=9, shuffle=True, random_state=42)

#-----------------------------------------------------------------------------------------------------#

# call our custom function to get the best folds
best_train_x, best_train_y, best_test_x, best_test_y = nested_cross_validation(data, target, model, folds)
print("Best training data set:\n", best_train_x)
print("Best training target set:\n", best_train_y)
print("Best testing data set:\n", best_test_x)
print("Best testing target set:\n", best_test_y)

#-----------------------------------------------------------------------------------------------------#
