# Heart Disease Prediction using Machine Learning Models

This project aims to predict the likelihood of heart disease in individuals based on various health-related parameters. The predictions are made using machine learning models trained on a dataset of heart disease data. The user inputs their health information into a web application, and the system provides predictions using multiple algorithms.

## Features
- Predicts whether a person has heart disease or not (Heart Disease / No Heart Disease).
- Uses multiple machine learning algorithms for predictions: 
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Decision Tree
  - Logistic Regression
- Displays the predictions of all models on a single page for comparison.
- Provides visualizations comparing the prediction performance of different models.
- Web interface built using Django for user-friendly interaction.

## Dataset
The dataset used for training the models consists of 1000 records with the following input parameters:

- **Age:** Integer value between 20 and 100.
- **Gender:** 0 for Female, 1 for Male.
- **Cholesterol:** Integer value representing Cholesterol in mg/dL.
- **Blood Pressure:** Integer value representing Blood Pressure in mmHg.
- **Heart Rate:** Integer value representing Heart Rate in beats per minute.
- **Smoking:** 0 for No, 1 for Yes.
- **Alcohol Intake:** 0 for No, 1 for Yes.
- **Family History:** 0 for No, 1 for Yes (whether the individual has a family history of heart disease).
- **Diabetes:** 0 for No, 1 for Yes.
- **Obesity:** 0 for No, 1 for Yes.
- **Exercise Induced Angina:** 0 for No, 1 for Yes.
- **Chest Pain Type:** Categorical value with the following options:
  - 0: Typical Angina
  - 1: Atypical Angina
  - 2: Non-Anginal Pain
  - 3: Asymptomatic


## How It Works
1. **Input:** Users input their health details via a form on the web application.
2. **Prediction:** The system uses pre-trained machine learning models (KNN, Random Forest, Decision Tree, Logistic Regression) to predict whether the user is likely to have heart disease.
3. **Output:** The predictions from all models are displayed on a results page, along with any visualizations comparing the results.
