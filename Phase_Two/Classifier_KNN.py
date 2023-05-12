from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score

# import dataset
cleaned_dataset = pd.read_csv('../dataset/cleaned_dataset.csv')

# initializations
K_N = 20
accuracies = []
mean_accuracies = []
K_arr = []

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

# KNN classification + cross validation
for i in range(1, K_N+1):
    K_arr.append(i)
    KNN = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(KNN, X, Y, cv=30)

    accuracies.append(scores)

# Display mean and standard devation of accuracies for each K
for i in range(len(accuracies)):
    s = f'Accuracy for K={i+1}:'
    print(s)
    print('-'*len(s))
    mean_accuracies.append(accuracies[i].mean()*100)
    print(f"""Mean Accuracy: {round(accuracies[i].mean()*100)} %
Standard Deviation: {accuracies[i].std()}\n""")
    
# Plot the relation between K and mean accuracies
plt.figure('KNN',figsize=(10, 6))
plt.plot(K_arr, mean_accuracies)
plt.title('K Nearest Neigbors Classifier')
plt.xlabel('K (Number of Nearest Neighbors)')
plt.ylabel('Mean Accuracy (%)')
plt.xticks(range(0, K_N+2))
plt.grid()
plt.show()