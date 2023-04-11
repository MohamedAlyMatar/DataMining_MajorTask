import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('Dataset of Diabetes .csv')
# print(dataset)

for tuple in dataset.values.tolist():
    for val in tuple:
        if val == np.nan:
            print('empty value')

categorical = dataset.select_dtypes(include=['object']).columns.tolist()
# print(categorical)
encoder = LabelEncoder()

# print(dataset[['CLASS']].values.tolist())
for i in categorical:
    for val in dataset[[i]].values.tolist():
        dataset[[i]] = dataset[[i]].replace([val[0]], val[0].replace(' ', '').upper())
    # print(dataset[[i]].values.tolist())
    dataset[i] = encoder.fit(dataset[i]).transform(dataset[i])
    # print(dataset[[i]].values.tolist())

for val in dataset[['Gender']].values.tolist():
    if val[0] > 1:
        print('Error in Gender column')
for val in dataset[['CLASS']].values.tolist():
    if val[0] > 2:
        print('Error in Class column')

print(dataset)