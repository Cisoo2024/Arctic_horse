from django.urls import path
from . import views

urlpatterns = [
    path('route/', views.route_form, name='route_form'),
    path('route/<int:pk>/', views.route_result, name='route_result'),
]