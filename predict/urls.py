from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_heart_disease, name='home'),  # Home page with form
    path('results/', views.results_view, name='results'),  # Results page
]
