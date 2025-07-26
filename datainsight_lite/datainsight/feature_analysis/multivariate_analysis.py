import matplotlib.pyplot as plt
import seaborn as sns
from .feature_analysis import FeatureAnalysis

class MultiVariateAnalysis(FeatureAnalysis):
    def plot_pairplot(self,hue=None):
        sns.pairplot(self.data[self._numerical_columns + ([hue] if hue else [])], hue=hue)
        plt.suptitle("Pairplot", y=1.02)
        plt.show()

    def plot_heatmap(self):
        corr = self.data[self._numerical_columns].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()
