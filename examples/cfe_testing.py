import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

from datainsight_lite.utils.categorical.count_frequency_encoder import CountFrequencyEncoder
import pandas as pd


df = pd.read_csv("test_with_categorical.csv")
cols = ["Gender","Region","Membership","IsActive"]


cfe = CountFrequencyEncoder(mode="frequency")

data = df['Gender']

data = cfe.fit_transform(data)

print(data)
print("\n")

print("Details of CountFrequencyEncoder: ")
print(cfe.__dict__)

cfe.save("count_freq_config.json")

print("\n")

print(cfe.inverse_transform([0.4, 0.37]))

cfe2 = cfe.load("count_freq_config.json")

print("\n")

print("Details of CountFrequencyEncoder: ")
print(cfe2.__dict__)