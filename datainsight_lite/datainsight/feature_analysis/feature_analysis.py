import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

from reporting.reporter import Reporter

class FeatureAnalysis:
    def __init__(self):
        rep=Reporter()
        self.data = rep.__handle_categorical_columns()
        self._numerical_columns, self._categorical_columns = rep.__get_num_cat_cols()

    def display_unique(self):
        unique_dict= {}
        for col in self._categorical_columns:
            unique_column_data = self.data[col].unique().tolist()
            unique_dict[col] = unique_column_data
        return unique_dict
