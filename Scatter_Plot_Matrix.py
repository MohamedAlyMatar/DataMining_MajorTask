from Data_Preprocessing import dataset
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    sns.set_theme(style="ticks")
    sns.pairplot(dataset, hue='CLASS')
    plt.show()


if __name__ == '__main__':
    main()
