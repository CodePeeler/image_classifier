from .models import Dataset, DSCategory, BinaryModel, TrainConfig, Image
from django import forms
from django.utils.translation import gettext_lazy as _


class DatasetForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['save_dir', 'class_labels', 'type', 'input_shape', 'category']
        labels = {'save_dir': '', 'class_labels': 'labels', 'type': 'Type', 'input_shape': 'Shape',
                  'category': 'Category'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Exclude labels from all fields
        for field_name in self.fields:
            self.fields[field_name].label = False

        self.fields['class_labels'].widget.attrs['placeholder'] = _('labels')
        self.fields['type'].widget.attrs['placeholder'] = _('type')
        self.fields['input_shape'].widget.attrs['placeholder'] = _('shape:  e.g  300, 300, 3')
        self.fields['category'].widget.attrs['placeholder'] = _('category')


class BinaryModelForm(forms.ModelForm):
    class Meta:
        model = BinaryModel
        fields = ['name', 'description']
        labels = {'name': 'Name', 'description': 'Description'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Exclude labels from all fields
        for field_name in self.fields:
            self.fields[field_name].label = False

        # Optionally, you can set placeholders for the fields instead of labels
        self.fields['name'].widget.attrs['placeholder'] = _('Name')
        self.fields['description'].widget.attrs['placeholder'] = _('Description')


class TrainConfigForm(forms.ModelForm):

    class Meta:
        model = TrainConfig
        fields = ['training_ds', 'validation_ds', 'learning_rate', 'minimum_accuracy', 'num_of_batches',
                  'num_of_epochs', 'training_ds', 'validation_ds']
        labels = {'training_ds': 'Training', 'validation_ds': 'Validation'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields['training_ds'].queryset = Dataset.objects.filter(category=DSCategory.TRAINING)
        self.fields['validation_ds'].queryset = Dataset.objects.filter(category=DSCategory.VALIDATION)
        self.fields['learning_rate'].widget.attrs['placeholder'] = _('learning rate: e.g 0.001')
        self.fields['minimum_accuracy'].widget.attrs['placeholder'] = _('min accuracy: e.g 0.99')
        self.fields['num_of_batches'].widget.attrs['placeholder'] = _('number of batches: e.g 8')
        self.fields['num_of_epochs'].widget.attrs['placeholder'] = _('number of epochs:  e.g  15')


class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['img_file']
        labels = {'img_file': ''}
