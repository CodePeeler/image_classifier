from django.urls import path
from . import views


app_name = 'binary_cnn'
urlpatterns = [
    # Home page
    path('', views.home, name='home'),
    path('ml_models/', views.ml_models, name="ml_models"),
    path('upload/', views.upload, name='upload'),
    path('classify_summary/<int:img_id>', views.classify_summary, name='classify_summary'),
    path('classify_summary/summary_update/<int:img_id>', views.summary_update, name='summary_update'),
]
