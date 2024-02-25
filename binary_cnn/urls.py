from django.urls import path
from . import views


app_name = 'binary_cnn'
urlpatterns = [
    # Home page
    path('', views.home, name='home'),
    path('ml_models/', views.ml_models, name="ml_models"),
    path('upload/', views.upload, name='upload'),

    path('classify_form/', views.classify_form, name='classify_form'),
    path('classify_result.html/<int:image_id>/', views.classify_result, name='classify_result.html'),
]
