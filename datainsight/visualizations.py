import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig(save_path)
    plt.close()


def plot_missing_values(df, save_path):
    plt.figure(figsize=(8, 6))
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values()
    if not missing.empty:
        sns.barplot(x=missing.values, y=missing.index, color="red")
        plt.title('Missing Values per Column')
        plt.xlabel('Count')
        plt.ylabel('Column')
        plt.savefig(save_path)
    plt.close()