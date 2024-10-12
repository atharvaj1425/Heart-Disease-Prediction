# predictor/forms.py
from django import forms

class PredictionForm(forms.Form):
    age = forms.IntegerField(label='Age')
    sex = forms.ChoiceField(label='Sex', choices=[(0, 'Female'), (1, 'Male')])
    cp = forms.ChoiceField(label='Chest Pain Type', choices=[(0, 'Typical Angina'), (1, 'Atypical Angina'), (2, 'Non-anginal Pain'), (3, 'Asymptomatic')])
    trestbps = forms.IntegerField(label='Resting Blood Pressure')
    chol = forms.IntegerField(label='Serum Cholesterol')
    fbs = forms.ChoiceField(label='Fasting Blood Sugar', choices=[(0, 'Less than 120 mg/dl'), (1, 'Greater than or equal to 120 mg/dl')])
    restecg = forms.ChoiceField(label='Resting Electrocardiographic Results', choices=[(0, 'Normal'), (1, 'Having ST-T wave abnormality'), (2, 'Showing probable or definite left ventricular hypertrophy')])
    thalach = forms.IntegerField(label='Maximum Heart Rate Achieved')
    exang = forms.ChoiceField(label='Exercise Induced Angina', choices=[(0, 'No'), (1, 'Yes')])
    oldpeak = forms.FloatField(label='Oldpeak (depression induced by exercise)')
    slope = forms.ChoiceField(label='Slope of Peak Exercise ST Segment', choices=[(0, 'Upsloping'), (1, 'Flat'), (2, 'Downsloping')])
    ca = forms.ChoiceField(label='Number of Major Vessels', choices=[(0, '0'), (1, '1'), (2, '2'), (3, '3')])
    thal = forms.ChoiceField(label='Thalassemia', choices=[(0, 'Normal'), (1, 'Fixed Defect'), (2, 'Reversible Defect')])
