# Machine Learning Project

This project will demonstrate a complete machine learning workflow:
data exploration, preprocessing, model training, evaluation, and
an interactive Streamlit application.

## Project status
Git and GitHub repository setup completed.
# Iris Flower Predictor

A machine learning project that predicts an Iris flower's species from four measurements. A Streamlit application lets users enter measurements and receive a prediction from a saved model.

## Dataset

The Iris dataset is loaded using scikit-learn and saved as `data/iris.csv`.

- 150 samples, with 50 samples per species
- Species: setosa, versicolor, and virginica
- Features: sepal length, sepal width, petal length, and petal width
- All measurements are in centimetres
- No missing values were found during exploration

Dataset source:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html

## Machine Learning Workflow

1. Load the dataset and save it as a CSV file.
2. Explore its size, missing values, class counts, and summary statistics.
3. Separate the measurement features from the species labels.
4. Split the data into 120 training samples and 30 test samples, using a stratified split and random_state=42.
5. Use a pipeline containing StandardScaler and LogisticRegression.
6. Fit the scaler and classifier using only the training data.
7. Evaluate predictions on the held-out test data.
8. Save the complete pipeline using joblib.
9. Load the saved pipeline in Streamlit without retraining.

## Results

- Test accuracy: 93.33% (28 of 30 predictions correct)
- Macro-average F1 score: 0.93
- Full classification report: `reports/evaluation.txt`

These results describe performance on one small test split and do not guarantee accuracy for every new flower.

## Project Files

- `prepare_data.py`: creates the dataset CSV
- `explore_data.py`: explores the dataset
- `train_model.py`: trains, evaluates, and saves the model
- `app.py`: Streamlit prediction interface
- `data/iris.csv`: dataset
- `models/iris_model.joblib`: saved scaler and classifier pipeline
- `reports/evaluation.txt`: evaluation results
- `requirements.txt`: Python dependencies
- `.gitignore`: excludes the virtual environment and temporary Python files

## Run Locally on Windows

Use Python 3.12. Open Command Prompt in the project folder.

Create and activate a virtual environment:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bat
python -m pip install -r requirements.txt
```

Run the application using the included saved model:

```bat
python -m streamlit run app.py
```

To reproduce the dataset exploration and model training:

```bat
python prepare_data.py
python explore_data.py
python train_model.py
```

## Using the Application

Enter the four measurements and click Predict. The application displays the predicted species. Input limits reflect the measurement ranges observed in the dataset.

## Live Application

The application is deployed on Streamlit Community Cloud:

https://atifk8326-droid-machine-learning-project-app-dz4puz.streamlit.app/