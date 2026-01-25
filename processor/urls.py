from django.urls import path
from . import views

urlpatterns = [
    path('label_customer/', views.new_transaction, name='label_customer'),
    path('retrain_all/', views.retrain_all, name='retrain_all'),
    path('relabel_all/', views.relabel_all, name='relabel_all'),
]