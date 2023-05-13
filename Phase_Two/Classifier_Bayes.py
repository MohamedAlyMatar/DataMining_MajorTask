import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from My_Cross_Validation import nested_cross_validation
import matplotlib.pyplot as plt

dataset = pd.read_csv('../dataset/cleaned_dataset.csv')
X = dataset.drop('CLASS',axis=1)
Y = dataset[['CLASS']]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

GNB = GaussianNB()

folds = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy = cross_val_score(GNB,X,Y,cv=30)

best_training_x, best_training_y, best_testing_x,best_testing_y = nested_cross_validation(X_train, Y_train, GNB, folds)

Y_pred = GNB.predict(best_testing_x)



print("GNB Results: \n", Y_pred)

print("Accuracy: ", accuracy)

cm = confusion_matrix(best_testing_y, Y_pred)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Diabetic","Predicted Diabetic","Diabetic"])

cm_display.plot()
plt.show()