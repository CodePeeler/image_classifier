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


class TemplateType(models.TextChoices):
    UNKNOWN = "Unknown"
    BINARY_MODEL = 'Binary Model'


class Status(models.TextChoices):
    UNTRAINED = 'Untrained'
    TRAINING = 'Training'
    TRAINED = 'Trained'
    ERROR = 'Error'


class BinaryModel(models.Model):
    type = TemplateType.BINARY_MODEL
    name = models.CharField(max_length=100)
    description = models.TextField()

    # Updates date anytime bm is saved.
    date_added = models.DateTimeField(auto_now=True)

    save_dir = models.CharField(max_length=100, default='binary_cnn/saved_models')

    static_dir_path = models.CharField(max_length=100)
    status = models.CharField(max_length=9, choices=Status.choices, default=Status.UNTRAINED)

    is_active = models.BooleanField(default=False)

    def __str__(self):
        """Return a string representation of the model. """
        return self.name


#  Delete Binary Model on file system on delete of instances.
@receiver(pre_delete, sender=BinaryModel)
def delete_binary_model_file(sender, instance, **kwargs):
    # Delete the saved model.
    if instance.save_dir and instance.name:
        path = instance.save_dir + '/' + instance.name
        if os.path.exists(path):
            shutil.rmtree(path)
    # Delete the model's static content.
    if instance.static_dir_path:
        path = instance.static_dir_path
        if os.path.exists(path):
            shutil.rmtree(path)


class TrainConfig(models.Model):
    name = models.CharField(max_length=100)
    binary_model = models.OneToOneField(BinaryModel, on_delete=models.CASCADE)

    training_ds = models.ForeignKey(Dataset, on_delete=models.PROTECT, related_name='config_training_ds')
    validation_ds = models.ForeignKey(Dataset, on_delete=models.PROTECT, related_name='config_validation_ds')

    learning_rate = models.DecimalField(default=0.001, max_digits=10, decimal_places=8)
    minimum_accuracy = models.DecimalField(default=0.99, max_digits=10, decimal_places=8)
    num_of_batches = models.IntegerField(default=8)
    num_of_epochs = models.IntegerField(default=15)

    def __str__(self):
        """Return a string representation of the model. """
        return self.name


class TrainingKpi(models.Model):
    name = models.CharField(max_length=100)
    config = models.OneToOneField(TrainConfig, on_delete=models.CASCADE)

    actual_accuracy = models.DecimalField(default=0, max_digits=10, decimal_places=8)
    actual_val_accuracy = models.DecimalField(default=0, max_digits=10, decimal_places=8)
    actual_epochs = models.IntegerField(default=0)

    # The proportion of true positives predicted to total number of positives predicted.
    # Indicates how many of the positive predictions are correct.
    precision = models.DecimalField(default=0, max_digits=10, decimal_places=8)

    # The proportion of correctly predicted true samples to the actual number of true samples.
    # Sensitivity - what percentage of pop did model catch.
    recall = models.DecimalField(default=0, max_digits=10, decimal_places=8)

    # f1_score = 2*(precision * recall)/(precision + recall)
    f1_score = models.DecimalField(default=0, max_digits=10, decimal_places=8)

    training_time_dhms = models.CharField(blank=True, max_length=100)

    def __str__(self):
        """Return a string representation of the model. """
        return self.name

    def set_training_time_dhms(self, train_time):
        """The train_time is in seconds. """
        days = int(train_time / (24 * 3600))
        r = int(train_time % (24 * 3600))
        hours = r // 3600
        r = int(r % 3600)
        minutes = r // 60
        seconds = int(r % 60)
        self.training_time_dhms = f"{days}:d {hours}:h {minutes}:m {seconds}:s"


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
