import pandas as pd
from utils import datatypes
import os
import matplotlib.pyplot as plt
import seaborn as sns
class Reporter:
    def __init__(self, file_path: datatypes.Str, sheet_name : datatypes.Str | None = None):
        self.data = self.__validate_file(file_path, sheet_name=sheet_name)
    

    def __dataset_summary(self, data: datatypes.Dataframe) -> datatypes.Dict:
        return {
            "shape": data.shape,
            "columns": data.columns,
            "dtypes": data.dtypes.apply(str).to_dict(),
            "missing_values": data.isnull().sum().to_dict()
        }
    
    
    def __describe_numerical(self, data: datatypes.Dataframe) -> datatypes.Dataframe:
        return data.describe().transpose()
    
    
    def __detect_outliers(self, data: datatypes.Dataframe) -> datatypes.Dict:
        outliers_info = {}

        for col in data.select_dtypes(include="number").columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR , Q3 + 1.5 * IQR

            outliers = data[(data[col] < lower)|(data[col] > upper)]

            outliers_info[col] = len(outliers)
        
        return outliers_info


    def __plot_corr_heatmap(self, data: datatypes.Dataframe) -> datatypes.Str:
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.savefig("heatmap.png")
        plt.close()

        return "heatmap.png"
    

    def __plot_missing_values(self, data: datatypes.Dataframe) -> None:
        plt.figure(figsize=(8, 6))
        missing = data.isnull().sum()
        missing = missing[missing > 0].sort_values()
        if not missing.empty:
            sns.barplot(x=missing.values, y=missing.index, color='red')
            plt.title('Missing Values per Column')
            plt.xlabel('Count')
            plt.ylabel('Column')
            plt.savefig("missing.png")
        else:
            print("No Missing Value Found. Skipped Creating PNG")


    def __validate_file(self, file_path: datatypes.Str, **kwargs) -> datatypes.Dataframe:
        sheet_name = kwargs.get('sheet_name', None)

        extension = os.path.basename(file_path).split('.')[-1]
        if extension == 'csv':
            df = pd.read_csv(file_path)
        elif extension == 'xlsx':
            if sheet_name is None:
                raise KeyError("sheet_name cant be none in case of excel files. pass proper sheet_name")
            df = pd.read_excel(file_path, sheet_name)
        
        elif extension == 'json':
            df = pd.read_json(file_path)
        
        else:
            raise TypeError("Only CSV/Excel/Json type file are accepted for now.")
        
        return df
    

    def report(self):
        summary_dict = self.__dataset_summary(self.data)
        description = self.__describe_numerical(self.data)
        outliers_info = self.__detect_outliers(self.data)
        heatmap_path = self.__plot_corr_heatmap(self.data)
        missing_value_path = self.__plot_missing_values(self.data)
        
        return {
            "summary": summary_dict,
            "description": description,
            "outliers": outliers_info,
            "heatmap_path": heatmap_path,
            "missing_path": missing_value_path
        }
    

    