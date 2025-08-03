import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

from abc import ABC, abstractmethod
import json
from datatypes import Series, Dataframe, List

DataTypes = (Series | Dataframe | List)

class BaseEncoder(ABC):
    
    @abstractmethod
    def fit(self, data: DataTypes): ...


    @abstractmethod
    def transform(self, data: DataTypes): ...


    @abstractmethod
    def fit_transform(self, data: DataTypes): ...
    
    
    @abstractmethod
    def inverse_transform(self, data: DataTypes): ...


    @abstractmethod
    def _fit(self, data): ...


    @abstractmethod
    def _transform(self, data): ...


    @abstractmethod
    def _inverse_transform(self, data): ...


    @abstractmethod
    def to_dict(self):
        pass
    

    @classmethod
    @abstractmethod
    def from_dict(cls, state_dict):
        pass


    def save(self, filename: str):
        state_dict = self.to_dict()
        with open(filename, "w") as f:
            json.dump(state_dict, f, indent=4)
        print(f"Saved encoder to {filename}")
    

    @classmethod
    def load(cls, filename: str):
        with open(filename, "r") as f:
            state_dict = json.load(f)
        return cls.from_dict(state_dict)