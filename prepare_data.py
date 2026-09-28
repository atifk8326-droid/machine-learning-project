from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

# Load the flower measurements and species labels.
iris = load_iris()

# Arrange the measurements into a table.
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target_names[iris.target]

# Save the table in our data folder.
Path("data").mkdir(exist_ok=True)
df.to_csv("data/iris.csv", index=False)

# Display the first five rows and the table size.
print(df.head())
print("\nRows and columns:", df.shape)
print("\nDataset saved to data/iris.csv")