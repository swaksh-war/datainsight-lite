import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

from .base_encoder import BaseEncoder
from datatypes import Series, Dataframe, List, Str
import pandas as pd
import numpy as np

class LabelBinarizer(BaseEncoder):
    def __init__(self, handle_missing:Str = "error", missing_value:Str|int|float=-1, handle_unknown:Str = "error", unknown_value: Str|int|float=-1):
        self.classes_ = []
        self._label2id = None
        self._id2label = None
        self._handle_mmissing = handle_missing
        self._missing_value = missing_value
        self._handle_unknown = handle_unknown
        self._unknown_value = unknown_value


    def _fit(self, data: Series):
        unique_labels = data.unique().tolist()
        self.classes_ = unique_labels
        self._label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self._id2label = {idx: label for label, idx in self._label2id.items()}
        
        return self


    def fit(self, data) -> Series|List:
        if not isinstance(data, (Series, List)):
            raise TypeError("Only Series and List are supported for Label Binarization")
        
        data: Series = self.__check_missing(data)
        return self._fit(data)

    def transform(self, data):
        return super().transform(data)
    
    def _transform(self, data):
        return super()._transform(data)
    

    def fit_transform(self, data):
        return super().fit_transform(data)


    def inverse_transform(self, data):
        return super().inverse_transform(data)
    
    def _inverse_transform(self, data):
        return super()._inverse_transform(data)
    
    def to_dict(self):
        return super().to_dict()
    
    @classmethod
    def from_dict(cls, state_dict):
        return super().from_dict(state_dict)

    def __check_missing(self, data: Series| List):
        if isinstance(data, List):
            data = pd.Series(np.array(data))
            return self.__check_missing_single_feature(data, None)
        if isinstance(data, Series):
            return self.__check_missing_single_feature(data, None)


    def __check_missing_single_feature(self, data: Series, col_name: str|None):
        total_missing = data.isna().sum()
        if total_missing == 0:
            return data
        else:
            if self._handle_mmissing == "error":
                if col_name is not None:
                    raise ValueError(f"data in column '{col_name}' contains missing value. fix data or dont pass 'error' as handle_missing")
                else:
                    raise ValueError("data contains missing value. fix data or dont pass 'error' as handle_missing")
            if self._handle_mmissing == "replace":
                return data.fillna(self._missing_value)
            else:
                raise RuntimeError("pass 'error' or 'replace' as argument value for handle missing. if replace then pass the value aswell default is -1")


