from django.urls import path
from . import views


app_name = 'binary_cnn'
urlpatterns = [
    # Home page
    path('', views.home, name='home'),
    path('models/', views.models, name="models"),
    path('model/<int:model_id>', views.model, name="model"),
    path('model/delete/<int:model_id>', views.delete_model, name="delete_model"),
    path('create/', views.create, name="create"),
    path('train/<int:model_id>', views.train, name="train"),
    path('classify_form/', views.classify_form, name='classify_form'),
    path('classify_result/<int:image_id>/', views.classify_result, name='classify_result.html'),
]
