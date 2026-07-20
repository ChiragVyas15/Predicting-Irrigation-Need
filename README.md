# Predicting-Irrigation-Need
ntelligent Irrigation Need Prediction using CatBoost. Performs feature engineering, multiclass classification, and real-time predictions through a Streamlit application to assist farmers in efficient water management.


# 🌱 Irrigation Need Prediction using Machine Learning

A machine learning project that predicts the irrigation requirement (**Low, Medium, High**) using soil, weather, crop, and environmental parameters. The project uses **CatBoost Classifier**, feature engineering, and a **Streamlit** web application for real-time predictions. :contentReference[oaicite:0]{index=0}

---

## 📌 Features

- Predicts irrigation need (Low, Medium, High)
- CatBoost multiclass classification model
- Advanced feature engineering
- Interactive Streamlit web application
- Batch prediction using CSV files
- Generates submission files for predictions

---

## 📂 Dataset Features

- Soil Type
- Soil pH
- Soil Moisture
- Organic Carbon
- Electrical Conductivity
- Temperature
- Humidity
- Rainfall
- Sunlight Hours
- Wind Speed
- Crop Type
- Crop Growth Stage
- Season
- Irrigation Type
- Water Source
- Field Area
- Mulching Used
- Previous Irrigation
- Region

---

## 🛠 Feature Engineering

The following features were created:

- Moisture Deficit
- Water Availability
- Evaporation Index
- Heat Stress
- Water per Area
- Moisture × Rainfall
- Soil pH Category

These engineered features improve the model's predictive performance. :contentReference[oaicite:1]{index=1}

---

## 🤖 Machine Learning Model

- Algorithm: CatBoost Classifier
- Multiclass Classification
- Loss Function: MultiClass
- Evaluation Metric: Accuracy

Training configuration includes 1000 iterations, learning rate 0.05, and depth 6. :contentReference[oaicite:2]{index=2}

---

## 📊 Technologies Used

- Python
- Pandas
- NumPy
- CatBoost
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn

---

## 📁 Project Structure

```
├── train.csv
├── test.csv
├── final_model.cbm
├── App.py
├── generate_submission.py
├── submission.csv
├── sample_submission.csv
└── README.md
```

---

## 🚀 Run the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Streamlit App

```bash
streamlit run App.py
```

### Generate Predictions

```bash
python generate_submission.py
```

---

## 📈 Workflow

1. Load dataset
2. Perform feature engineering
3. Train CatBoost model
4. Evaluate performance
5. Predict irrigation need
6. Generate submission file
7. Deploy with Streamlit

---

## 🎯 Prediction Classes

- Low
- Medium
- High

---

## 🌾 Applications

- Smart Farming
- Precision Agriculture
- Water Resource Management
- Crop Monitoring
- Decision Support for Farmers

---

## 📄 License

This project is intended for educational and research purposes.
