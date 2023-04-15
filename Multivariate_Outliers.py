from Data_Preprocessing import dataset
from sklearn.ensemble import IsolationForest

dataset2 = dataset.copy()
dataset2 = dataset2.drop(['CLASS'], axis=1)
model = IsolationForest(n_estimators=150, max_samples='auto', contamination=float(0.1), max_features=1.0)
model.fit(dataset2)
scores = model.decision_function(dataset2)
anomaly = model.predict(dataset2)

dataset2['scores'] = scores
dataset2['anomaly'] = anomaly
print(dataset2)

anomaly = dataset2.loc[dataset2['anomaly'] == -1]
anomaly_index = list(anomaly.index)
print('Total number of outliers is:', len(anomaly))

dataset2['CLASS']=dataset['CLASS']
cleaned_dataset=dataset2.drop(anomaly_index, axis = 0).reset_index(drop=True)
cleaned_dataset=cleaned_dataset.drop(['scores','anomaly'],axis=1)

print(cleaned_dataset)

