import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

from datatypes import Series, List, Str
from .label_encoder import LabelEncoder

class OrdinalEncoder(LabelEncoder):
    def __init__(self, handle_missing="error", missing_value=-1, handle_unknown="error", unknown_value=-1, order: List|Str = "auto"):
        
        """
        Options for Order:
        1. Insert a List for unique_elements in order i.e if [a, b, a, c, a, b] then you need to pass the order somewhat like this :- [a, b, c] or [b, a, c] as per user required
        2. auto
        """

        super().__init__(handle_missing, missing_value, handle_unknown, unknown_value)
        
        self.order = order

    def _fit(self, data: List | Series):
        if isinstance(data, List):
            unique_labels = list(set(data))
            if isinstance(self.order, Str):
                if self.order == "auto":
                    ordered_class = sorted(unique_labels)
                    self.classes_ = ordered_class
                    self._label2id = {label: idx for idx, label in enumerate(self.classes_)}
                    self._id2label = {idx: label for label, idx in self._label2id.items()}
                    return self

                else:
                    raise ValueError("order should be either a list of objects or auto defining --> 'lexicographic' or 'numerical'")
            if isinstance(self.order, List):
                valid = self.__check_order(unique_labels)
                if valid:
                    self.classes_ = self.order
                    self._label2id = {label: idx for idx, label in enumerate(self.classes_)}
                    self._id2label = {idx: label for label, idx in self._label2id.items()}
                    return self

        elif isinstance(data, Series):
            unique_labels = data.unique().tolist()
            if isinstance(self.order, Str):
                if self.order == "auto":
                    ordered_class = sorted(unique_labels)
                    self.classes_ = ordered_class
                    self._label2id = {label: idx for idx, label in enumerate(self.classes_)}
                    self._id2label = {idx: label for label, idx in self._label2id.items()}
                    return self

                else:
                    raise ValueError("order should be either a list of objects or auto defining --> 'lexicographic' or 'numerical'")
            
            if isinstance(self.order, List):
                valid = self.__check_order(unique_labels)
                if valid:
                    self.classes_ = self.order
                    self._label2id = {label: idx for idx, label in enumerate(self.classes_)}
                    self._id2label = {idx: label for label, idx in self._label2id.items()}
                    return self
    
    def __check_order(self, labels):
        
        for elem in self.order:
            if elem not in labels:
                raise ValueError("order missing items for labels. Please check the order list passed properly")
        
        return True