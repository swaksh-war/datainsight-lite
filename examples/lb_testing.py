import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import pandas as pd
import numpy as np
from datainsight_lite.utils.categorical.label_binarizer import LabelBinarizer

lb = LabelBinarizer()

df = pd.read_csv("test_with_categorical.csv")

data = ["male", "female", "male", "female", "other", "other"]
# data = df["Gender"]
d = lb.fit_transform(data)
print(d)
print(lb.inverse_transform([[0,1,0], [0,0,1]]))

print(lb.inverse_transform(np.array([[0,1,0], [0,0,1]])))