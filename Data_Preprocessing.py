import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# reading data from csv file
dataset = pd.read_csv('dataset/Dataset of Diabetes.csv')


# checking if any value is empty
for tuple in dataset.values.tolist():
    for val in tuple:
        if val == np.nan:
            print('empty value')
# no empty values


categorical = dataset.select_dtypes(include=['object']).columns.tolist()
encoder = LabelEncoder()
for i in categorical:
    # cleaning data and unifying their format
    for val in dataset[[i]].values.tolist():
        dataset[[i]] = dataset[[i]].replace([val[0]], val[0].replace(' ', '').upper())
    # label encoding for categorical data
    dataset[i] = encoder.fit(dataset[i]).transform(dataset[i])


# checking if any categorical value is not labelled in the correct range
for val in dataset[['Gender']].values.tolist():
    if val[0] > 1:
        print('Error in Gender column')
for val in dataset[['CLASS']].values.tolist():
    if val[0] > 2:
        print('Error in Class column')
# no value is labelled incorrectly


# remove unnecessary id and patien number columns
dataset = dataset.iloc[:,2:]


# splitting the data into training and tesing
x = dataset.drop('CLASS', axis=1)
y = dataset[['CLASS']]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)