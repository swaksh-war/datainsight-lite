import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
import pandas as pd
from datainsight_lite.utils.categorical.label_encoder import LabelEncoder
from datainsight_lite.utils.datatypes import Series, Dataframe, List

le = LabelEncoder()

df = pd.read_csv("test_with_categorical.csv")
cols = ["Gender","Region","Membership","IsActive"]
print(df[cols])
data = df[cols]
print(type(data))
print(isinstance(data, Series))
print(isinstance(data, Dataframe))
print(isinstance(data, List))

data = le.fit_transform(data)

print(data)

# print(le.inverse_transform([0,1,2,1,2]))

le.save("test.json")

new_le = LabelEncoder.load("test.json")
print(new_le.__dict__)