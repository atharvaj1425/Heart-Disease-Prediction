from django.shortcuts import render
from django import forms
import joblib
import numpy as np

# Load all models and scaler
knn_model = joblib.load('new models/knn_model01.pkl')
decision_tree_model = joblib.load('new models/decision_tree_model01.pkl')
random_forest_model = joblib.load('new models/random_forest_model01.pkl')
log_reg_model = joblib.load('new models/log_reg_model01.pkl')
scaler = joblib.load('new models/scaler.pkl')

# Example accuracies (to be replaced by actual values if available)
model_accuracies = {
    'KNN': 0.85,
    'Decision Tree': 0.80,
    'Random Forest': 0.88,
    'Logistic Regression': 0.82,
}

# Define the form within the view
class HeartDiseaseForm(forms.Form):
    age = forms.IntegerField(label='Age', min_value=20, max_value=100)
    gender = forms.ChoiceField(choices=[(0, 'Female'), (1, 'Male')], label='Gender')
    cholesterol = forms.IntegerField(label='Cholesterol(mg/dL)')
    blood_pressure = forms.IntegerField(label='Blood Pressure(in mmHg)')
    heart_rate = forms.IntegerField(label='Heart Rate(beats per min)')
    smoking = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label='Smoking')
    alcohol_intake = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label='Alcohol Intake')
    family_history = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label='Family History')
    diabetes = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label='Diabetes')
    obesity = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label='Obesity')
    exercise_induced_angina = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label='Exercise Induced Angina')
    chest_pain_type = forms.ChoiceField(
        choices=[(0, 'Typical Angina'), (1, 'Atypical Angina'), (2, 'Non-Anginal Pain'), (3, 'Asymptomatic')],
        label='Chest Pain Type'
    )
    exercise_hours = forms.FloatField(label='Exercise Hours per Week')
    stress_level = forms.FloatField(label='Stress Level (1-10)')
    blood_sugar = forms.FloatField(label='Blood Sugar Level(in mg/dL)')

# View to display form
# Define the view to handle form input and make predictions
def predict_heart_disease(request):
    if request.method == 'POST':
        form = HeartDiseaseForm(request.POST)
        if form.is_valid():
            # Extract form data and ensure the correct format
            data = [
                form.cleaned_data['age'],
                int(form.cleaned_data['gender']),  # Encoded 0 or 1 for gender
                form.cleaned_data['cholesterol'],
                form.cleaned_data['blood_pressure'],
                form.cleaned_data['heart_rate'],
                int(form.cleaned_data['smoking']),  # Encoded 0 or 1 for smoking
                int(form.cleaned_data['alcohol_intake']),  # Encoded 0 or 1 for alcohol intake
                form.cleaned_data['exercise_hours'],
                int(form.cleaned_data['family_history']),  # Encoded 0 or 1 for family history
                int(form.cleaned_data['diabetes']),  # Encoded 0 or 1 for diabetes
                int(form.cleaned_data['obesity']),  # Encoded 0 or 1 for obesity
                form.cleaned_data['stress_level'],
                form.cleaned_data['blood_sugar'],
                int(form.cleaned_data['exercise_induced_angina']),  # Encoded 0 or 1 for exercise-induced angina
                int(form.cleaned_data['chest_pain_type']),  # Encoded as 0, 1, 2, 3 for chest pain types
            ]

            # Convert to numpy array and scale the data
            input_data = np.array([data])
            input_data_scaled = scaler.transform(input_data)

            # Get predictions from all models
            knn_prediction = knn_model.predict(input_data_scaled)[0]
            decision_tree_prediction = decision_tree_model.predict(input_data_scaled)[0]
            random_forest_prediction = random_forest_model.predict(input_data_scaled)[0]
            log_reg_prediction = log_reg_model.predict(input_data_scaled)[0]

            # Aggregate the predictions in a dictionary
            predictions = {
                'KNN': 'Heart Disease' if knn_prediction == 1 else 'No Heart Disease',
                'Decision Tree': 'Heart Disease' if decision_tree_prediction == 1 else 'No Heart Disease',
                'Random Forest': 'Heart Disease' if random_forest_prediction == 1 else 'No Heart Disease',
                'Logistic Regression': 'Heart Disease' if log_reg_prediction == 1 else 'No Heart Disease',
            }

            # Prepare data for the chart
            chart_labels = list(predictions.keys())
            chart_data = [1 if value == 'Heart Disease' else 0 for value in predictions.values()]

            # Render the results page with predictions and chart data
            return render(request, 'predict/results.html', {
                'predictions': predictions,
                'chart_labels': chart_labels,
                'chart_data': chart_data
            })

    else:
        form = HeartDiseaseForm()

    return render(request, 'predict/home.html', {'form': form})


# View for displaying results
# New view for displaying results
def results_view(request):
    predictions = request.session.get('predictions', {})
    return render(request, 'predict/results.html', {'predictions': predictions})
