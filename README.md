# -ML-housing-price-predictor
# 🏡 California Housing Price Predictor 2025

This project uses machine learning to predict median house values across California based on housing and demographic features. It includes data preprocessing, model training, evaluation, and deployment using `joblib`.

---
## [Live Demo](https://youtu.be/pExhfoI6PPM)
## 📁 Project Structure

- `housing_price_predictor_2025.ipynb` — Main notebook for data analysis and model training.
- `california_housing_model.pkl` — Saved machine learning model.
- `preprocessing_pipeline.pkl` — Saved preprocessing pipeline for numeric features.
- `README.md` — Project overview and usage instructions.

---

## 📊 Dataset

The dataset includes features such as:
- `longitude`, `latitude`
- `housing_median_age`
- `total_rooms`, `total_bedrooms`
- `population`, `households`
- `median_income`
- `median_house_value` (target)

---

## 🧠 Model Overview

- **Model Type**: Regression (e.g., Random Forest, Gradient Boosting)
- **Pipeline**: Numeric preprocessing using scaling/imputation
- **Evaluation Metrics**: RMSE, R² Score

---

## 🔧 How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/housing_price_predictor_2025.git
cd housing_price_predictor_2025
---

### 2. Install Dependencies

pip install pandas scikit-learn joblib matplotlib seaborn
---

### 3. Run the Notebook
Open [housing_price_predictor_2025.ipynb](https://colab.research.google.com/drive/1xROSFCmZp6w2CM9pf-ylUcrCGOxRQ7EQ?usp=sharing) in Jupyter or Google Colab.
---

🚀 Predicting New Data

import joblib
import pandas as pd

# Load model and pipeline
model = joblib.load('california_housing_model.pkl')
pipeline = joblib.load('preprocessing_pipeline.pkl')

# Example new data
new_data = pd.DataFrame({
    'longitude': [-122.23],
    'latitude': [37.88],
    'housing_median_age': [41],
    'total_rooms': [880],
    'total_bedrooms': [129],
    'population': [322],
    'households': [126],
    'median_income': [8.3252]
})

# Preprocess and predict
prepared_data = pipeline.transform(new_data)
prediction = model.predict(prepared_data)
print(f"Predicted Median House Value: ${prediction[0]:,.2f}")

----

📌 Highlights
- 🔥 Correlation heatmap for feature selection
- 🗺️ Geospatial scatter plot for housing trends
- 💾 Model and pipeline saved with joblib
- 🎯 Ready for deployment or integration into web apps
---

👩‍💻 Author
Zahabia Ahmed
Technologist | Educator | Motivator
Passionate about AI, data science, and creative storytelling.

Perfect, Zahabia! Here's a well-structured README.md draft tailored for your housing_price_predictor_2025.ipynb project. It’s designed to be clear, professional, and beginner-friendly—ideal for GitHub, portfolio sharing, or even as a script base for your YouTube content.

# 🏡 California Housing Price Predictor 2025

This project uses machine learning to predict median house values across California based on housing and demographic features. It includes data preprocessing, model training, evaluation, and deployment using `joblib`.

---

## 📁 Project Structure

- `housing_price_predictor_2025.ipynb` — Main notebook for data analysis and model training.
- `california_housing_model.pkl` — Saved machine learning model.
- `preprocessing_pipeline.pkl` — Saved preprocessing pipeline for numeric features.
- `README.md` — Project overview and usage instructions.

---

## 📊 Dataset

The dataset includes features such as:
- `longitude`, `latitude`
- `housing_median_age`
- `total_rooms`, `total_bedrooms`
- `population`, `households`
- `median_income`
- `median_house_value` (target)

---

## 🧠 Model Overview

- **Model Type**: Regression (e.g., Random Forest, Gradient Boosting)
- **Pipeline**: Numeric preprocessing using scaling/imputation
- **Evaluation Metrics**: RMSE, R² Score

---

## 🔧 How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/housing_price_predictor_2025.git
cd housing_price_predictor_2025


2. Install Dependencies
pip install pandas scikit-learn joblib matplotlib seaborn


3. Run the Notebook
Open housing_price_predictor_2025.ipynb in Jupyter or Google Colab.

🚀 Predicting New Data
import joblib
import pandas as pd

# Load model and pipeline
model = joblib.load('california_housing_model.pkl')
pipeline = joblib.load('preprocessing_pipeline.pkl')

# Example new data
new_data = pd.DataFrame({
    'longitude': [-122.23],
    'latitude': [37.88],
    'housing_median_age': [41],
    'total_rooms': [880],
    'total_bedrooms': [129],
    'population': [322],
    'households': [126],
    'median_income': [8.3252]
})

# Preprocess and predict
prepared_data = pipeline.transform(new_data)
prediction = model.predict(prepared_data)
print(f"Predicted Median House Value: ${prediction[0]:,.2f}")



📌 Highlights
- 🔥 Correlation heatmap for feature selection
- 🗺️ Geospatial scatter plot for housing trends
- 💾 Model and pipeline saved with joblib
- 🎯 Ready for deployment or integration into web apps

👩‍💻 Author
Zahabia Ahmed
Data scientist | Technologist | Educator | Motivator 
Passionate about AI, data science, and creative storytelling.

📬 Contact
For collaboration, feedback, or content ideas:
📧 zahabia0ahmed@gmail.com
📺 YouTube: [Zahabia Ahmed](http://www.youtube.com/@ZahabiaAhmed)

---














```bash
git clone https://github.com/your-username/housing_price_predictor_2025.git
cd housing_price_predictor_2025
