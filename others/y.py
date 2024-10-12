import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('heart_disease_dataset.csv')

# Preprocess the data
# Initialize the label encoder
label_encoder = LabelEncoder()

# Encoding the categorical columns
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Smoking'] = label_encoder.fit_transform(df['Smoking'])
df['Alcohol Intake'] = label_encoder.fit_transform(df['Alcohol Intake'])
df['Family History'] = label_encoder.fit_transform(df['Family History'])
df['Diabetes'] = label_encoder.fit_transform(df['Diabetes'])
df['Obesity'] = label_encoder.fit_transform(df['Obesity'])
df['Exercise Induced Angina'] = label_encoder.fit_transform(df['Exercise Induced Angina'])
df['Chest Pain Type'] = label_encoder.fit_transform(df['Chest Pain Type'])

# Feature and target variables
X = df.drop(columns=['Heart Disease'])  # All 15 features (including both categorical and numeric)
y = df['Heart Disease']  # Target

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (scale the features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize a dictionary to store model names and their accuracies
model_accuracies = {}

# Function to evaluate a model and store accuracy
def evaluate_model(model, model_name):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[model_name] = accuracy

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
evaluate_model(knn, "K-Nearest Neighbors")

# Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
evaluate_model(decision_tree, "Decision Tree")

# Random Forest
random_forest = RandomForestClassifier(random_state=42)
evaluate_model(random_forest, "Random Forest")

# Logistic Regression
log_reg = LogisticRegression(random_state=42)
evaluate_model(log_reg, "Logistic Regression")

# To make predictions for a new user input using all models
def predict_heart_disease(model, user_data):
    user_data_df = pd.DataFrame([user_data], columns=X.columns)  # Create DataFrame with feature names
    user_data_scaled = scaler.transform(user_data_df)  # Scale the new input using the same scaler
    prediction = model.predict(user_data_scaled)
    return 'Heart Disease' if prediction == 1 else 'No Heart Disease'

# Example of a new input (replace with actual input values)
new_input = [60, 1, 280, 150, 95, 1, 3, 1, 1, 1, 1, 8, 200, 1, 0]  # 15 input features

# Predicting heart disease using all models
print("\nPrediction for new input using different models:")
for model_name, model in zip(model_accuracies.keys(), [knn, decision_tree, random_forest, log_reg]):
    prediction = predict_heart_disease(model, new_input)
    print(f"{model_name}: {prediction}")

# Plotting the accuracies of all models
plt.figure(figsize=(10, 5))
plt.bar(model_accuracies.keys(), model_accuracies.values(), color=['skyblue', 'lightgreen', 'orange', 'pink'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.show()
