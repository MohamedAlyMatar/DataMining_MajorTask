import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVC

 
# reading data from csv file
dataset = pd.read_csv('../dataset/cleaned_dataset.csv')

data = dataset.drop('CLASS', axis=1)
target = dataset['CLASS']

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Define the model and hyperparameter grid
model = SVC()
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}

# Perform nested cross-validation
outer_cv = KFold(n_splits=9, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Outer loop: model evaluation
outer_scores = []
for train_index, test_index in outer_cv.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inner loop: hyperparameter tuning
    model = SVC()  # Reinitialize the model
    grid_search = GridSearchCV(model, param_grid, cv=inner_cv)
    grid_search.fit(X_train_fold, y_train_fold)

    # Evaluate the model with the best hyperparameters on the validation fold
    best_model = grid_search.best_estimator_
    score = best_model.score(X_val_fold, y_val_fold)
    outer_scores.append(score)

# Compute the average performance across all folds
avg_score = np.mean(outer_scores)
print("Average Score:", avg_score)

# Compute the cross validation score
scores = cross_val_score(model, data, target, cv=outer_cv, scoring='accuracy')
print("Cross-validated Scores:", scores)

# Evaluate the best model on the test set
best_model.fit(X_train, y_train)
test_score = best_model.score(X_test, y_test)
print("Test Score:", test_score)