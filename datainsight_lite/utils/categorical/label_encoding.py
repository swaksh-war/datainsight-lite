import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

from .base_encoder import BaseEncoder
from datatypes import Series, List, Dataframe
import pandas as pd
import numpy as np
import json

class LabelEncoder(BaseEncoder):
    def __init__(self, handle_missing="error", missing_value=-1, handle_unknown="error", unknown_value=-1):
        self.classes_ :None|list|dict = None
        self._label2id = None
        self._id2label = None
        self._handle_missing = handle_missing
        self._missing_value = missing_value
        self._handle_unknown = handle_unknown
        self._unknown_value = unknown_value


    def _fit(self, data : Series|List|Dataframe):
        if isinstance(data, Series):
            unique_elems = data.unique().tolist()
            self.classes_ = unique_elems
            self._label2id = {label: idx for idx, label in enumerate(unique_elems)}
            self._id2label = {idx: label for label, idx in self._label2id.items()}
            print("Fitted mappings:", self._label2id)
            return self

        if isinstance(data, Dataframe):
            self.classes_ = {}
            self._label2id = {}
            self._id2label = {}
            for col in data.columns:
                columnar_data = data[col]
                unique_column_data = columnar_data.unique().tolist()
                self.classes_[col] = unique_column_data
                self._label2id[col] = {label: idx for idx, label in enumerate(unique_column_data)}
                self._id2label[col] = {idx: label for label, idx in self._label2id[col].items()}
            return self

        if isinstance(data, List):
            unique_elems = list(set(data))
            self.classes_ = unique_elems
            self._label2id = {label: idx for idx, label in enumerate(self.classes_)}
            self._id2label = {idx: label for label, idx in self._label2id.items()}
            return self


    def fit(self, data: List|Dataframe|Series):
        if isinstance(data, (Series, Dataframe, List)):
            data = self.__check_missing(data, if_true=self._handle_missing, value=self._missing_value)
            return self._fit(data)
        else:
            raise TypeError("Only Pandas Series, Dataframe and normal list is supported")


    def __transform_single_feature(self, data: Series, col_name: str| None , if_unknown: str, unknown_value: str|int|float) -> Series:
        if self._label2id is  None:
            raise RuntimeError("Encoder isnt fitted. run fit() first or fit_transform() to fit and transform simultaniously")

        if col_name is None:
            mapping = self._label2id
        else:
            if col_name not in self._label2id:
                raise RuntimeError(f"Column '{col_name}' was not fitted. run fit() first or fit_transform() to fit and transform simultaniously")
            mapping = self._label2id[col_name]

        def map_label(label):
            if label in mapping:
                return mapping[label]
            elif pd.isna(label):
                return np.nan
            else:
                if if_unknown == "error":
                    raise ValueError(f"Unseen Label '{label}' in column '{col_name}'")
                elif if_unknown == "replace":
                    return unknown_value
                elif if_unknown == "skip":
                    return label
                else:
                    raise ValueError(f"Invalid handle_unknown: {if_unknown}")

        return data.map(map_label)


    def __transform(self, data, if_unknown, unknown_value) -> Series | Dataframe:
        if self._id2label is None or self._label2id is None:
            raise RuntimeError("Looks Like Encoder isn't fitted run fit_transform or fit function first before transforming.")

        else:
            if isinstance(data, Series):
                return self.__transform_single_feature(data, None, if_unknown, unknown_value)
            if isinstance(data, List):
                data = pd.Series(np.array(data))
                return self.__transform_single_feature(data, None, if_unknown, unknown_value).to_list()
            if isinstance(data, Dataframe):
                for col in data.columns:
                    col_data = data[col]
                    transformed_data = self.__transform_single_feature(col_data, col, if_unknown, unknown_value)
                    data[col] = transformed_data
                return data


    def transform(self, data) -> Series|Dataframe:
        if isinstance(data, (Series, Dataframe, List)):
            return self.__transform(data, self._handle_unknown, self._unknown_value)
        else:
            raise TypeError("Only Pandas Series, Dataframe and normal List is supported")
        

    def fit_transform(self, data: Series|Dataframe|List):
        self.fit(data)
        return self.transform(data)


    def __inverse_transform(self, data: List|Series|Dataframe, col_name: str| None) -> Series:
        if self._id2label is None:
            raise RuntimeError("Encoder isnt fitted. run fit() first or fit_transform() to fit and transform simultaniously")
        
        if col_name is None:
            mapping = self._id2label
        else:
            if col_name not in self._id2label:
                raise RuntimeError(f"Column '{col_name}' was not fitted. run fit() first or fit_transform() to fit and transform simultaniously")
            mapping = self._id2label[col_name]
        
        def map_id(id):
            if id in mapping:
                return mapping[id]
            elif pd.isna(id):
                return np.nan
            else:
                raise ValueError(f"Unseen ID {id} in column '{col_name}'")
        
        return data.map(map_id)
    

    def inverse_transform(self, data: Series| Dataframe| List, return_type: Series|List = "as"):
        if not isinstance(data, (Series, Dataframe, List)):
            raise TypeError("Only Pandas Series, Dataframe and normal List is supported")

        if isinstance(data, List):
            data = pd.Series(np.array(data))
            data = self.__inverse_transform(data, None)
            return data.to_list()
        if isinstance(data, Series):
            data = self.__inverse_transform(data, None)
            if return_type == "as" or return_type == "series":
                return data
            elif return_type == "list":
                return data.to_list()

        if isinstance(data, Dataframe):
            for col in data.columns:
                col_data = data[col]
                transformed_data = self.__inverse_transform(col_data, col)
                data[col] = transformed_data
            return data



    def __check_missing(self, data: Series| Dataframe| List, if_true: str, value: str|int|float):
        if isinstance(data, List):
            data = pd.Series(np.array(data))
            obj = self.__check_missing_single_feature(data, if_true, value)
            return obj.to_list()

        elif isinstance(data, Series):
            return self.__check_missing_single_feature(data, if_true, value)

        elif isinstance(data, Dataframe):
            for col in data.columns:
                col_data = data[col]
                obj = self.__check_missing_single_feature(col_data, if_true, value)
                data[col] = col_data
            return data
        else:
            raise TypeError("Only List, Series, Dataframe are accepted")        


    def __check_missing_single_feature(self, data: Series, if_true, value) -> Series:
        total_missing = data.isna().sum()
        print(f"Total Missing: {total_missing}")
        if total_missing > 0 and if_true == "error":
            raise ValueError("There is Null or NaN inside data and missing_handling mode is 'error' fix the data or change the mode to replace and pass missing_value with it")

        if total_missing > 0 and if_true == "replace":
            return data.fillna(value)
        
        return data


    def to_dict(self, exclude=None):
        exclude = exclude or []

        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("__") and k not in exclude
        }
    

    def save(self, path: str, exclude=None):
        state = self.to_dict(exclude)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
    

    @classmethod
    def load(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"file {path} does not exist")
        
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        instance = cls.__new__(cls)
        instance.__dict__.update(state)

        if hasattr(instance, "_post_load"):
            instance._post_load()
        
        return instance