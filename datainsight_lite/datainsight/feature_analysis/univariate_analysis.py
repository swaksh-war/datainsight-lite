import matplotlib.pyplot as plt
import seaborn as sns
import math
from .feature_analysis import FeatureAnalysis

class UniVariateAnalysis(FeatureAnalysis):
    #Histogram – distribution of continuous data
    def plot_histogram(self,bins=30):
        n_cols = 2  
        n_rows = math.ceil(len(self._numerical_columns) / n_cols)
        plt.figure(figsize=(n_cols * 6, n_rows * 4))

        for i, col in enumerate(self._numerical_columns, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(self.data[col], bins=bins, kde=True, color='skyblue')
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()
    
    #Boxplot – spread and outliers
    def plot_boxplot(self):
        n_cols = 2
        n_rows = math.ceil(len(self._numerical_columns) / n_cols)
        plt.figure(figsize=(n_cols * 6, n_rows * 4))

        for i, col in enumerate(self._numerical_columns, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.boxplot(x=self.data[col], color='orange')
            plt.title(f'Boxplot of {col}')
            plt.xlabel(col)

        plt.tight_layout()
        plt.show()

    #KDE plot – smoothed density of distribution
    def plot_kdeplot(self):
        n_cols = 2
        n_rows = math.ceil(len(self._numerical_columns) / n_cols)
        plt.figure(figsize=(n_cols * 6, n_rows * 4))

        for i, col in enumerate(self._numerical_columns, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.kdeplot(self.data[col], shade=True, color='purple')
            plt.title(f'KDE Plot of {col}')
            plt.xlabel(col)
            plt.ylabel("Density")

        plt.tight_layout()
        plt.show()

    #Bar chart – for categorical data
    def plot_barchart(self):
        n_cols = 2
        n_rows = math.ceil(len(self._categorical_columns) / n_cols)
        plt.figure(figsize=(n_cols * 6, n_rows * 4))

        for i, col in enumerate(self._categorical_columns, 1):
            plt.subplot(n_rows, n_cols, i)
            self.data[col].value_counts().plot(kind='bar', color='blue')
            plt.title(f'Bar Chart of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    #Pie chart – proportion of categories (not recommended for many categories)
    def plot_piechart(self):
        n_cols = 2
        n_rows = math.ceil(len(self._categorical_columns) / n_cols)
        plt.figure(figsize=(n_cols * 6, n_rows * 5))

        for i, col in enumerate(self._categorical_columns, 1):
            plt.subplot(n_rows, n_cols, i)
            self.data[col].value_counts().plot(
                kind='pie', 
                autopct='%1.1f%%', 
                startangle=90,
                textprops={'fontsize': 10},
                colormap='tab20'
            )
            plt.ylabel("")
            plt.title(f'Pie Chart of {col}')

        plt.tight_layout()
        plt.show()
