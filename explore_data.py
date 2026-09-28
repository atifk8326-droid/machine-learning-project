import pandas as pd

df = pd.read_csv("data/iris.csv")

print("DATASET SIZE")
print(df.shape)

print("\nMISSING VALUES")
print(df.isna().sum())

print("\nFLOWERS PER SPECIES")
print(df["species"].value_counts())

print("\nMEASUREMENT SUMMARY")
print(df.describe().round(2))
