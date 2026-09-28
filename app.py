from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸")

# Load the model we already trained.
model_path = Path(__file__).parent / "models" / "iris_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

st.title("🌸 Iris Flower Predictor")
st.write("Enter the flower measurements in centimetres, then click Predict.")

# Collect all four measurements before predicting.
with st.form("measurements"):
    sepal_length = st.number_input(
        "Sepal length (cm)", min_value=4.3, max_value=7.9,
        value=5.1, step=0.1
    )
    sepal_width = st.number_input(
        "Sepal width (cm)", min_value=2.0, max_value=4.4,
        value=3.5, step=0.1
    )
    petal_length = st.number_input(
        "Petal length (cm)", min_value=1.0, max_value=6.9,
        value=1.4, step=0.1
    )
    petal_width = st.number_input(
        "Petal width (cm)", min_value=0.1, max_value=2.5,
        value=0.2, step=0.1
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    # Use the same column names and order as during training.
    measurements = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    prediction = model.predict(measurements)[0]
    st.success(f"Predicted species: {prediction.capitalize()}")

st.caption(
    "Learning demo using the Iris dataset. "
    "Input limits reflect the dataset's observed measurement ranges."
)