# Ad Click Prediction Model 📊

This project is an end-to-end machine learning application that predicts whether a user will click on an online advertisement. It uses supervised learning to understand user behavior and estimate click-through probability based on demographic and usage data.

The goal of this project is not just model accuracy, but also building a **complete, reproducible data science workflow** — from analysis to deployment.

---

## 📌 Project Overview

The model is trained on the **Advertisement – Click on Ad** dataset, which contains user-level information such as:

- Age  
- Area Income  
- Daily Time Spent on Site  
- Daily Internet Usage  
- Gender  
- Timestamp  
- Ad Topic Line  
- Clicked on Ad (target variable)

The output is a prediction of **Click / No Click**, along with the probability of a click.

---

## 🔍 What’s Included

### 📈 Exploratory Data Analysis (EDA)
- 10–15 meaningful analyses  
- Feature distributions and summary statistics  
- Correlation analysis  
- Click behavior by age, gender, and income  
- Time-of-day effects on ad clicks  
- Basic analysis of ad topic categories  

### 🛠️ Data Preprocessing
- Dropping irrelevant columns (e.g. City)  
- Feature extraction from timestamps (hour of day)  
- Simple categorization of ad topic text  
- Encoding categorical features  
- Train-test split  

### 🤖 Machine Learning
- Supervised classification model  
- Models used  **Logistic Regression**  
- Evaluation using accuracy and confusion matrix  
- Click-through probability prediction  

### 🌐 Streamlit Web App
- Interactive EDA visualizations  
- Model explanation  
- User input form for real-time predictions  
- Displays predicted class and probability  

---

## 🧪 Tech Stack

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Streamlit  

---

## 🚀 How to Run

1. Clone the repository  
2. Install dependencies  
   ```bash
   pip install -r requirements.txt
