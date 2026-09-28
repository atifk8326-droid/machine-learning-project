from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Read the dataset.
df = pd.read_csv("data/iris.csv")

# 2. Separate measurements (X) from species labels (y).
X = df.drop(columns=["species"])
y = df["species"]

# 3. Keep 20% aside for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Scale the measurements and create the classifier.
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# 5. Learn using only the training data.
model.fit(X_train, y_train)

# 6. Predict species for the unseen test data.
predictions = model.predict(X_test)

# 7. Measure how well the model performed.
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print("Training flowers:", len(X_train))
print("Testing flowers:", len(X_test))
print(f"\nTest accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(report)

# 8. Save the trained model and evaluation results.
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

joblib.dump(model, "models/iris_model.joblib")

with open("reports/evaluation.txt", "w", encoding="utf-8") as file:
    file.write(f"Test accuracy: {accuracy:.2%}\n\n")
    file.write(report)

print("Model saved to models/iris_model.joblib")
print("Results saved to reports/evaluation.txt")