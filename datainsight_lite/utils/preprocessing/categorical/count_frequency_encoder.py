import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

from .label_encoder import LabelEncoder
from datatypes import Series, List, Dataframe
from collections import Counter


class CountFrequencyEncoder(LabelEncoder):
    def __init__(self, handle_missing="error", missing_value=-1, handle_unknown="error", unknown_value=-1, mode="count"):
        
        super().__init__(handle_missing, missing_value, handle_unknown, unknown_value)
        
        self.mode = mode


    def _fit(self, data: Series|List|Dataframe):
        if isinstance(data, Series):
            unique_elems = data.unique().tolist()
            elems_counter = Counter(data.tolist())
            
            if self.mode == "frequency":
                total = elems_counter.total()
                elems_counter = dict(elems_counter)
                for elem in unique_elems:
                    elems_counter[elem] = elems_counter[elem]/total

            self.classes_ = unique_elems
            self._label2id = dict(elems_counter) if isinstance(elems_counter, dict) else elems_counter
            self._id2label = {idx: label for label, idx in self._label2id.items()}
            return self
        
        if isinstance(data, Dataframe):
            self.classes_ = {}
            self._label2id = {}
            self._id2label = {}

            for col in data.columns:
                columnar_data = data[col]
                unique_column_data = columnar_data.unique().tolist()
                self.classes_[col] = unique_column_data
                elems_counter = Counter(columnar_data.tolist())

                if self.mode == "frequency":
                    total = elems_counter.total()
                    elems_counter = dict(elems_counter)
                    for elem in unique_column_data:
                        elems_counter[elem] = elems_counter/total

                self._label2id[col] = dict(elems_counter) if isinstance(elems_counter, dict) else elems_counter
                self._id2label[col] = {idx: label for label, idx in self._label2id[col].items()}
            
            return self
        
        if isinstance(data, List):
            unique_elems = list(set(data))
            self.classes_ = unique_elems
            elems_counter = Counter(data)

            if self.mode == "frequency":
                total = elems_counter.total()
                elems_counter = dict(elems_counter)
                for elem in unique_elems:
                    elems_counter[elem] = elems_counter/total

            self._label2id = dict(elems_counter) if isinstance(elems_counter, dict) else elems_counter
            self._id2label = {idx: label for label, idx in self._label2id.items()}
        
        return self