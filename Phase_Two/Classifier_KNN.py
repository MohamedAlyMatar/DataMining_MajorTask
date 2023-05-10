from Multivariate_Outliers import cleaned_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = cleaned_dataset.drop('CLASS',axis=1)
Y = cleaned_dataset[['CLASS']]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)

KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train,Y_train)

KNN_RES = KNN.predict(X_test)
print("KNN Results: \n", KNN_RES)

accuracy = accuracy_score(Y_test,KNN_RES)
print("Accuracy: ", accuracy)
