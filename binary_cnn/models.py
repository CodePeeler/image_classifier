import os
import shutil

from django.db import models
from django.db.models.signals import pre_delete
from django.dispatch import receiver


class DSType(models.TextChoices):
    IMAGE = 'Image'
    TEXT = 'Text'
    AUDIO = 'Audio'
    VIDEO = 'Video'


class DSCategory(models.TextChoices):
    TRAINING = 'Training'
    VALIDATION = 'Validation'
    CLASSIFY = 'Classify'
    OTHER = 'Other'


class Dataset(models.Model):
    name = models.CharField(max_length=200, unique=True)
    save_dir = models.FileField(max_length=100, upload_to='binary_cnn/datasets/tmp')
    type = models.CharField(max_length=10, choices=DSType.choices)
    category = models.CharField(max_length=10, choices=DSCategory.choices)
    input_shape = models.CharField(max_length=255, blank=True)
    date_added = models.DateTimeField(auto_now=True)

    # Necessary for Supervised Learning.
    class_labels = models.CharField(max_length=2000, blank=True)

    def __str__(self):
        """Return a string representation of the model. """
        return self.name


@receiver(pre_delete, sender=Dataset)
def delete_dataset_file(sender, instance, **kwargs):
    # Check if the instances save_dir exits.
    if instance.save_dir:
        # Get the path to zip file
        ds_zip_path = instance.save_dir.path
        # Check if the zip file exists and delete it
        if os.path.exists(ds_zip_path):
            os.remove(ds_zip_path)


class Status(models.TextChoices):
    UNTRAINED = 'Untrained'
    TRAINING = 'Training'
    TRAINED = 'Trained'
    ERROR = 'Error'


class BinaryModel(models.Model):
    #bm_owner = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    description = models.TextField()
    # Updates date anytime bm is saved.
    date_added = models.DateTimeField(auto_now=True)

    save_dir = models.CharField(max_length=100, default='binary_cnn/saved_models')

    path_model_summary = models.CharField(max_length=100)
    status = models.CharField(max_length=9, choices=Status.choices, default=Status.UNTRAINED)

    is_active = models.BooleanField(default=False)

    def __str__(self):
        """Return a string representation of the model. """
        return self.name


#  Delete Binary Model on file system on delete of instances.
@receiver(pre_delete, sender=BinaryModel)
def delete_binary_model_file(sender, instance, **kwargs):
    # Delete the all cnn code files and directories.
    if instance.save_dir and instance.name:
        path = instance.save_dir + '/' + instance.name
        if os.path.exists(path):
            shutil.rmtree(path)
    # Delete the model summary text file.
    if os.path.exists(instance.path_model_summary):
        os.remove(instance.path_model_summary)


# BinaryModels are trained upon creation
class TrainingConfig(models.Model):
    # name = models.CharField(max_length=100)
    training_ds_dir = models.CharField(null=True, max_length=200)
    validation_ds_dir = models.CharField(null=True, max_length=200)

    #TODO: Delete this field.
    classification_classes = models.CharField(max_length=1000)

    learning_rate = models.DecimalField(max_digits=10, decimal_places=8)
    accuracy = models.DecimalField(max_digits=10, decimal_places=8)

    binary_model = models.OneToOneField(BinaryModel, on_delete=models.CASCADE)

    def __str__(self):
        """Return a string representation of the model. """
        return self.__class__.__name__


class Image(models.Model):
    img_file = models.ImageField(upload_to='binary_cnn/static/')
    is_classified = models.BooleanField(default=False)
    date_classified = models.DateTimeField(auto_now=True)
    img_classification = models.CharField(blank=True, max_length=100)
    prob_classification = models.DecimalField(null=True, max_digits=10, decimal_places=8)
    model_id = models.IntegerField(null=True)

    def __str__(self):
        """Return a string representation the type of model. """
        return self.img_file.name.split('/')[-1]


#  Delete Image on file system on delete of instances.
@receiver(pre_delete, sender=Image)
def delete_image_file(sender, instance, **kwargs):
    # Check if the image field has a value
    if instance.img_file:
        # Get the path of the image file
        image_path = instance.img_file.path
        # Check if the image file exists and delete it
        if os.path.exists(image_path):
            os.remove(image_path)



