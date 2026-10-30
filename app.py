
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# page configuration

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "wine_model.joblib"

st.set_page_config(
    page_title="Wine Quality Predictor",
    layout="centered"
)

IMAGE_PATH = PROJECT_DIR / "assets" / "whiteandred.jpg"


# loading in the trained model

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# website introduction

st.title("Wine Quality Predictor")

st.image(
    IMAGE_PATH,
    use_container_width=True
)

st.caption(
    "Image credit: Getty Images/iStockphoto"
)

st.write(
    "This website uses a Random Forest Model to let you explore how the chemical properties of wine " 
    "relate to its predicted quality. "
    "It predicts wine quality on a scale from 0 to 10 where higher values indicate a higher sensory quality."
)



st.info(
    "Enter the physicochemical properties of a wine to receive an estimated quality score."
)

# user input (wine type and characteristics)

st.header("Wine characteristics")

wine_type = st.selectbox(
    "Wine type",
    ["white", "red"]
)

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input(
        "Fixed acidity",
        value=7.0,
        min_value=0.0
    )

    volatile_acidity = st.number_input(
        "Volatile acidity",
        value=0.30,
        min_value=0.0
    )

    citric_acid = st.number_input(
        "Citric acid",
        value=0.30,
        min_value=0.0
    )

    residual_sugar = st.number_input(
        "Residual sugar",
        value=5.0,
        min_value=0.0
    )

    chlorides = st.number_input(
        "Chlorides",
        value=0.05,
        min_value=0.0
    )

    free_sulfur_dioxide = st.number_input(
        "Free sulfur dioxide",
        value=30.0,
        min_value=0.0
    )

with col2:
    total_sulfur_dioxide = st.number_input(
        "Total sulfur dioxide",
        value=120.0,
        min_value=0.0
    )

    density = st.number_input(
        "Density",
        value=0.995,
        min_value=0.0
    )

    ph = st.number_input(
        "pH",
        value=3.2,
        min_value=0.0
    )

    sulphates = st.number_input(
        "Sulphates",
        value=0.50,
        min_value=0.0
    )

    alcohol = st.number_input(
        "Alcohol (%)",
        value=10.5,
        min_value=0.0
    )


# quality prediction (triggered by button click)

st.divider()

if st.button("Predict wine quality", type="primary",
             use_container_width=True):

    input_data = pd.DataFrame([{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": ph,
        "sulphates": sulphates,
        "alcohol": alcohol,
        "wine_type": wine_type
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction result")

    st.metric(
        label="Predicted wine quality",
        value=f"{prediction:.2f} / 10"
    )

    st.write(
        "The result is an estimated quality score based on the trained Random Forest model. "
        "It is only an estimate, and not a guarantee of actual sensory quality"
    )