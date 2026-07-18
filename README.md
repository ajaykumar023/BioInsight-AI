# 🧬 BioInsight AI

BioInsight AI is an Explainable AI-powered web application for heart disease risk classification using machine learning and clinical patient data.

The project builds an end-to-end machine learning workflow including data exploration, preprocessing, model comparison, cross-validation, held-out test evaluation, error analysis, model persistence, explainability, and an interactive Streamlit web application.

> **Disclaimer:** BioInsight AI is an educational machine learning project. It is not a medical device and must not be used for diagnosis, treatment, or clinical decision-making.

---

## 🎯 Project Objective

The objective of BioInsight AI is to build a reproducible machine learning system that:

- Processes clinical heart disease data.
- Explores patient and feature distributions.
- Prevents preprocessing data leakage using scikit-learn Pipelines.
- Compares multiple classification algorithms.
- Selects a model using cross-validation on training data.
- Evaluates the selected model on a held-out test set.
- Performs prediction error analysis.
- Explains model predictions using Explainable AI techniques.
- Provides predictions and explanations through a Streamlit web application.

---

## 📊 Dataset

The project uses the Cleveland Heart Disease dataset.

Dataset characteristics:

- 920 patient records
- 13 input features
- 1 target variable
- Binary classification task

The original target values are converted into:

- `0` — No Heart Disease
- `1` — Heart Disease Present

Example features include:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression
- ST Slope
- Number of Major Vessels
- Thalassemia

---

## 🏗️ Project Structure

```text
BioInsight-AI/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── heart_disease.csv
│   │
│   └── processed/
│       └── heart_disease_clean.csv
│
├── models/
│   └── best_model.joblib
│
├── notebooks/
│   ├── 01_data_exploration.py
│   └── 02_eda.py
│
├── reports/
│   ├── figures/
│   ├── cv_results.csv
│   ├── final_test_results.csv
│   ├── model_results.csv
│   └── misclassified_patients.csv
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt