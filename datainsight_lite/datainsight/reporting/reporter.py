import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import pandas as pd
from ...utils import datatypes
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ...utils.categorical.label_encoder import LabelEncoder

class Reporter:
    def __init__(self, file_path: datatypes.Str, sheet_name : datatypes.Str | None = None, **kwargs):
        self.data = self.__validate_file(file_path, sheet_name=sheet_name)
        self._numerical_columns, self._categorical_columns = self.__get_num_cat_cols(**kwargs)    

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

        for col in self._numerical_columns:
            print(col)
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR , Q3 + 1.5 * IQR

            outliers = data[(data[col] < lower)|(data[col] > upper)]

            outliers_info[col] = len(outliers)
        
        return outliers_info


    def __plot_corr_heatmap(self) -> datatypes.Str:
        plt.figure(figsize=(10, 8))
        encoded_df = self.__handle_categorical_columns()
        sns.heatmap(encoded_df.corr(), annot=True, cmap="coolwarm", fmt='.2f')
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
        heatmap_path = self.__plot_corr_heatmap()
        missing_value_path = self.__plot_missing_values(self.data)
        
        return {
            "summary": summary_dict,
            "description": description,
            "outliers": outliers_info,
            "heatmap_path": heatmap_path,
            "missing_path": missing_value_path
        }
    
    def __handle_categorical_columns(self, inplace=False) -> None|datatypes.Dataframe:
        encoder=LabelEncoder()
        if inplace:
            for col in self._categorical_columns:
                self.data[col]=encoder.fit_transform(self.data[col])
            return self.data
        else:
            new_df = self.data.copy()
            for col in new_df.columns:
                if col in self._categorical_columns:
                    new_df[col]=encoder.fit_transform(new_df[col])
            return new_df      

    def __get_num_cat_cols(self, **kwargs):
        """
        kwargs: specials_cols_to_treat_as_numerical, special_cols_to_treat_as_categorical
        """
        cat_cols = kwargs.get("cols_to_treat_as_cat", [])
        num_cols = kwargs.get("cols_to_treat_as_num", [])
            
        for col in self.data.columns:
            if self.data[col].dtype == object and (col not in num_cols):
                cat_cols.append(col)
            if self.data[col].dtype != object and (col not in cat_cols):
                num_cols.append(col)
        return num_cols, cat_cols
    
    