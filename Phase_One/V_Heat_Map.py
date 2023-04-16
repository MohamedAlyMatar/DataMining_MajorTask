from Data_Preprocessing import dataset
import matplotlib.pyplot as plt
import seaborn as sns



def main():
    corr = dataset.corr()
    sns.heatmap(corr, annot = True)
    plt.show()


if __name__ == '__main__':
    main()


