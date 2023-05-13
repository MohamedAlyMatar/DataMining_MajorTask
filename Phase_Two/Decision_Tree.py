import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import metrics
from Multivariate_Outliers import cleaned_dataset
from My_Cross_Validation import *
from sklearn.model_selection import KFold
from sklearn import tree
from matplotlib import pyplot as plt


# reading data from csv file
dataset = pd.read_csv('../dataset/cleaned_dataset.csv')

data = dataset.drop('CLASS', axis=1)
target = dataset['CLASS']

feature_names= data.columns.values.tolist()
target_name='CLASS'


# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy")
folds = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross Validation
cv_scores, avg_score,best_testing_x,best_testing_y = nested_cross_validation(X_train, y_train, clf,folds)

#Predict the response for test dataset
y_pred = clf.predict(best_testing_x)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(best_testing_y, y_pred))

# Plotting the Decision Tree
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                   feature_names=feature_names,  
                   class_names=target_name,
                   filled=True)

fig.savefig("decistion_tree.png")