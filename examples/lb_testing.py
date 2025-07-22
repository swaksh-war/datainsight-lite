import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import pandas as pd
from datainsight_lite.utils.categorical.label_binarizer import LabelBinarizer

lb = LabelBinarizer()

df = pd.read_csv("test_with_categorical.csv")

# data = ["male", "female", "male", "female"]
data = df["Gender"]
print(lb.fit_transform(data))
print(type(lb.fit_transform(data)))
print(type(lb.fit_transform(data, True)))