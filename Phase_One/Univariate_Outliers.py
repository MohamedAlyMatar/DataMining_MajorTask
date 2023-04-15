from Data_Preprocessing import dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def detect_outliers(feature):
    name = feature.name
    feature = feature.to_numpy()

    # finding the 1st quartile
    q1 = np.quantile(feature, 0.25)

    # finding the 3rd quartile
    q3 = np.quantile(feature, 0.75)
    med = np.median(feature)

    # finding the iqr region
    iqr = q3 - q1

    # finding upper and lower whiskers
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    # print(iqr, upper_bound, lower_bound)

    outliers = feature[(feature <= lower_bound) | (feature >= upper_bound)]
    print(f'The following are the outliers of {name}')
    print(outliers)
    print("Length of outliers:", len(outliers))


def main():
    for col in list(dataset.drop(['Gender', 'CLASS'], axis=1).columns):
        detect_outliers(dataset[col])
        fig = plt.figure(figsize=(10, 7))
        plt.boxplot(dataset[col].to_numpy())
        plt.title(str(col) + "Box Plot")
        plt.show()


if __name__ == '__main__':
    main()
