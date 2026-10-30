
# Wine Quality Predictor

A small machine learning project that predicts wine quality based on physicochemical properties.
Link to the website: https://wine-quality-predictor-tneawdwyfkyrivmrcf6xt9.streamlit.app/

## About

The project is trained on the Wine Quality dataset from the UCI Machine Learning Repository. The model used is a Random Forest regression model that is trained to predict wine quality for red and white wines.

The web application is built with Streamlit and allows users to enter wine properties (the physiochemical properties) and receive a predicted quality score.

## How to run

1. Clone the repository.
2. Install the required packages:

   `pip install -r requirements.txt`

3. Start the application:

   `streamlit run app.py`

## Project files

- `app.py` – The web application.
- `exploratory_data.py` – Exploratory data analysis and visualizations.
- `training_model.py` – Model training, cross-validation and evaluation.
- `data/` – The original red and white wine datasets.
- `models/` – The trained model.
- `outputs/` – Figures generated during data exploration and model evaluation.

## Dataset

Cortez et al. (2009), Wine Quality Dataset, UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/186/wine+quality
