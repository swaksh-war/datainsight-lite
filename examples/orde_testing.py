import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import pandas as pd

from datainsight_lite.utils.preprocessing.categorical.ordinal_encoder import OrdinalEncoder

ord_encoder = OrdinalEncoder(order = "auto")

df = pd.read_csv("test_with_categorical.csv")

data = df['Gender']

data = ord_encoder.fit_transform(data)

print(data)

print(ord_encoder.__dict__)