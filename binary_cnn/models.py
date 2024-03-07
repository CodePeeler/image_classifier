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
    is_active = models.BooleanField(default=False)

    # Necessary for Supervised Learning.
    class_labels = models.CharField(max_length=2000, blank=True)

    def __str__(self):
        """Return a string representation of the model. """
        return self.name


@receiver(pre_delete, sender=Dataset)
def delete_dataset_dir(sender, instance, **kwargs):
    # Check if the instances save_dir exits.
    if instance.save_dir:
        # Get path to datasets root dir
        path = instance.save_dir.name
        # Ensure path still exists
        if os.path.exists(path):
            shutil.rmtree(path)


class Status(models.TextChoices):
    UNTRAINED = 'Untrained'
    TRAINING = 'Training'
    TRAINED = 'Trained'
    ERROR = 'Error'


class BinaryModel(models.Model):
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


class TrainConfig(models.Model):
    binary_model = models.OneToOneField(BinaryModel, on_delete=models.CASCADE)

    training_ds = models.ForeignKey(Dataset, on_delete=models.PROTECT, related_name='config_training_ds')
    validation_ds = models.ForeignKey(Dataset, on_delete=models.PROTECT, related_name='config_validation_ds')

    learning_rate = models.DecimalField(default=0.001, max_digits=10, decimal_places=8)
    minimum_accuracy = models.DecimalField(default=0.99, max_digits=10, decimal_places=8)
    num_of_batches = models.IntegerField(default=8)
    num_of_epochs = models.IntegerField(default=15)

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
