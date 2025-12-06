# Ad-Click-Prediction-Model
Ad Click Prediction Model using Python – predicting user engagement and click-through rates on advertisements.
This repository contains an end‑to‑end data science project for predicting whether a user will click on an online advertisement using supervised machine learning. The project follows the full workflow: Exploratory Data Analysis (EDA), data preprocessing, model training and evaluation, and deployment as an interactive Streamlit web application.

The dataset (Advertisement – Click on Ad) includes user‑level information such as age, area income, daily time spent on site, daily internet usage, gender, timestamp, and the ad’s topic line, along with a binary label indicating whether the user clicked the ad. The goal is to learn patterns in user behaviour and ad characteristics in order to estimate the probability of a click.

Key features of this project:

Exploratory Data Analysis (EDA): 10–15 analyses including summary statistics, feature distributions, correlation analysis, click‑through behaviour by demographic segments, time‑of‑day effects, and simple analysis of ad topics.

Data Preprocessing: Handling missing values (if any), dropping irrelevant columns (e.g. City), extracting features from timestamps (such as hour), converting ad topic text into simple categorical groups, encoding categorical variables, and splitting the data into training and test sets.

Machine Learning Model: A supervised classification model (e.g. Logistic Regression / Random Forest) trained to predict whether a user will click on an ad, evaluated using metrics such as accuracy and confusion matrix, and capable of producing click‑probability scores.

Streamlit Web App: An interactive interface that presents key EDA visualizations, explains the model, and allows users to input feature values (age, income, time on site, internet usage, gender, hour, ad topic category) to get real‑time predictions of click vs. no‑click with associated probability.

Reproducible Workflow: All steps (EDA, preprocessing, modeling, and app code) are organized so that the project can be run locally and tracked using Git and GitHub as the version‑control backbone.
