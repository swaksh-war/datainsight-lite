import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

from utils.datatypes import Dataframe

class OneHotEncoder:
    def __init__(self, drop="none",  handle_unknown = "ignore"):
        self.classes_ = None
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.auxiliary_array_size = None
        self.drop = drop


    def __fit(self, data: Dataframe):
        for col in data.columns:
            unique_col_items = data[col].unique().tolist()
            self.classes_ = {}
            self.classes_[col] = unique_col_items
            self.label_to_idx[col] = {label:idx for idx, label in enumerate(unique_col_items)}
            self.idx_to_label[col] = {idx:label for idx, label in enumerate(unique_col_items)}
        
        return self
    

    def __transform(self, data: Dataframe):
        pass


    def fit(self, data: Dataframe):
        if not isinstance(data, Dataframe):
            raise TypeError("Only DataFramme supported")
        else:
            return self.__fit(data)
        
    def transform(self, data: Dataframe):
        if not isinstance(data, Dataframe):
            raise TypeError("Only Dataframe is Supported")
        
        else:
            return self.__trasform(data)
