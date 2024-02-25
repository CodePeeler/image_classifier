from django import forms
from .models import BinaryModel, TrainingConfig, Image


class BinaryModelForm(forms.ModelForm):
    class Meta:
        model = BinaryModel
        fields = ['name', 'description']
        labels = {'name': 'Name', 'description': 'Description'}


class TrainingConfigForm(forms.ModelForm):
    class Meta:
        model = TrainingConfig
        fields = ['training_ds_dir', 'validation_ds_dir', 'classification_classes', 'learning_rate', 'accuracy']
        labels = {'training_ds_dir': 'Training Data Directory', 'validation_ds_dir': 'Validation Data Directory',
                  'classification_classes': 'Classification Classes', 'learning_rate': 'Learning Rate', 'accuracy': 'Model Accuracy'}


class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['img_file']
        labels = {'img_file': ''}
