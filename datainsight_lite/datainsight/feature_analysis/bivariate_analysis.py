import matplotlib.pyplot as plt
import seaborn as sns
from .feature_analysis import FeatureAnalysis

class BiVariateAnalysis(FeatureAnalysis):
    #Scatter plot – two continuous variables
    def plot_scatterplot(self, x, y):
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=self.data[x], y=self.data[y])
        plt.title(f"Scatter Plot: {x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        plt.show()
    
    #Boxplot– categorical vs continuous
    def plot_boxplot(self, cat_col, num_col):
        plt.figure(figsize=(6,4))
        sns.boxplot(x=self.data[cat_col],y=self.data[num_col])
        plt.title(f"Boxplot: {num_col} by {cat_col}")
        plt.xlabel(cat_col)
        plt.ylabel(num_col)
        plt.tight_layout()
        plt.show()

    #Violin plot– categorical vs continuous
    def plot_violinplot(self,cat_col,num_col):
        plt.figure(figsize=(6,4))
        sns.violinplot(x=self.data[cat_col],y=self.data[num_col])
        plt.title(f"Violin Plot: {num_col} by {cat_col}")
        plt.xlabel(cat_col)
        plt.ylabel(num_col)
        plt.tight_layout()
        plt.show()

    #Grouped bar chart – two categorical variables
    def plot_grouped_barchat(self,cat1,cat2):
        plt.figure(figsize=(7, 5))
        crosstab = self.data.groupby([cat1, cat2]).size().unstack().fillna(0)
        crosstab.plot(kind='bar', stacked=False)
        plt.title(f"Grouped Bar Chart: {cat1} vs {cat2}")
        plt.xlabel(cat1)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    #Line plot – time series or ordered data
    def plot_lineplot(self, x, y):
        plt.figure(figsize=(7, 5))
        sns.lineplot(x=self.data[x], y=self.data[y])
        plt.title(f"Line Plot: {y} over {x}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        plt.show()