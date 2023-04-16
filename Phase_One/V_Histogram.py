from Data_Preprocessing import dataset
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    plt.rcParams['font.size'] = 11
    plt.rcParams['figure.dpi'] = 75
    dataset.hist(figsize=(20,10))
    plt.show()

if __name__ == '__main__':
    main()
