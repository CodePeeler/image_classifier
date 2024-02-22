from django import forms
from .models import BinaryModel, Image


class BinaryModelForm(forms.ModelForm):
    class Meta:
        model = BinaryModel
        fields = ['name', 'description']
        labels = {'name': 'Name', 'description': 'Description'}


class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['img_file']
        labels = {'img_file': ''}
