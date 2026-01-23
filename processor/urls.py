from django.urls import path
from . import views

urlpatterns = [
    path('new_transaction/', views.new_transaction, name='new_transaction'),
    path('retrain_all/', views.retrain_all, name='retrain_all'),
    path('retrain_classifier/', views.retrain_classifier, name='retrain_classifier'),
]