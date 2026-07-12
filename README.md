# 🧬 BioInsight AI

BioInsight AI is an Explainable AI-based disease risk prediction system designed to predict heart disease risk using clinical patient data.

The project combines Machine Learning, Explainable AI, and an interactive web application to provide disease risk predictions along with understandable explanations of the factors influencing the model's decision.

## 🎯 Project Objective

The objective of BioInsight AI is to build an end-to-end Machine Learning system capable of:

- Processing clinical patient data
- Predicting the presence of heart disease
- Comparing multiple Machine Learning models
- Evaluating model performance using appropriate metrics
- Explaining predictions using SHAP
- Providing predictions through an interactive Streamlit application

## 📊 Dataset

The project uses the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.

The dataset contains 303 patient records and 14 attributes, including:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate
- Exercise-induced angina
- ST depression
- ST slope
- Number of major vessels
- Thalassemia
- Heart disease diagnosis

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- SHAP
- Streamlit
- Git
- GitHub

## 📁 Project Structure

BioInsight-AI/
- app/ — Streamlit application
- data/raw/ — Original dataset
- data/processed/ — Cleaned datasets
- models/ — Trained ML models
- notebooks/ — Data exploration and experimentation
- reports/figures/ — Generated visualizations
- src/ — Reusable source code

## 🚀 Project Status

### Day 1 — Project Setup and Dataset Exploration

- [x] Created project repository
- [x] Created Python virtual environment
- [x] Installed project dependencies
- [x] Created modular project structure
- [x] Downloaded the UCI Cleveland Heart Disease dataset
- [x] Loaded dataset using Pandas
- [x] Inspected dataset dimensions and features
- [x] Checked missing values
- [x] Analyzed target distribution
- [x] Added basic dataset validation

### Upcoming

- [ ] Data cleaning and preprocessing
- [ ] Exploratory Data Analysis
- [ ] Feature preparation
- [ ] Baseline Machine Learning models
- [ ] Model comparison and evaluation
- [ ] SHAP explainability
- [ ] Streamlit web application

## ⚠️ Disclaimer

BioInsight AI is an educational Machine Learning project and is not intended for medical diagnosis or clinical use.