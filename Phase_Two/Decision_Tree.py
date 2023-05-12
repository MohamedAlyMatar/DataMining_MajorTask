import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import metrics
from Multivariate_Outliers import cleaned_dataset
from sklearn.preprocessing import MinMaxScaler
from My_Cross_Validation import *
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold



# reading data from csv file
dataset = pd.read_csv('../dataset/cleaned_dataset.csv')

data = dataset.drop('CLASS', axis=1)
target = dataset['CLASS']


age = cleaned_dataset[['AGE']]
cr = cleaned_dataset[['Cr']]
bmi = cleaned_dataset[['BMI']]

# Data Normalization
normalizer = MinMaxScaler(feature_range=(0, 10))
age = normalizer.fit_transform(age)
cr = normalizer.fit_transform(cr)
bmi = normalizer.fit_transform(bmi)
cleaned_dataset[['AGE']] = age
cleaned_dataset[['Cr']] = cr
cleaned_dataset[['BMI']] = bmi

X = cleaned_dataset.drop('CLASS',axis=1)
Y = cleaned_dataset[['CLASS']]

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy")

param_grid = {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]}
outer_cv = KFold(n_splits=9, shuffle=True, random_state=42)

# Cross Validation
cv_scores, avg_score,best_training_x, best_training_y, best_testing_x,best_testing_y = nested_cross_validation(X_train, y_train, clf, param_grid, outer_cv)


# Train Decision Tree Classifer with the best training data
clf = clf.fit(best_training_x,best_training_y)
print(plot_tree(clf))

#Predict the response for test dataset
y_pred = clf.predict(best_testing_x)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(best_testing_y, y_pred))