from django.urls import path
from . import views


app_name = 'binary_cnn'
urlpatterns = [
    # Home page
    path('', views.home, name='home'),
    path('models/', views.models, name="models"),
    path('create/', views.create, name="create"),
    path('train/<int:model_id>', views.train, name="train"),
    path('update_status/', views.update_status, name='update_status'),
    path('classify_form/', views.classify_form, name='classify_form'),
    path('classify_result/<int:image_id>/', views.classify_result, name='classify_result.html'),
]
