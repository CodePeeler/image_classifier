import os

from django.db import models
from django.db.models.signals import pre_delete
from django.dispatch import receiver


# BinaryModels are trained upon creation
class TrainingConfig(models.Model):
    learning_rate = models.DecimalField(max_digits=10, decimal_places=8)
    accuracy = models.DecimalField(max_digits=10, decimal_places=8)

    def __str__(self):
        """Return a string representation of the model. """
        return self.__class__.__name__


class BinaryModel(models.Model):
    #owner = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    description = models.TextField()
    date_added = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False)
    parameters = models.OneToOneField(TrainingConfig,
                                      on_delete=models.CASCADE, related_name='binary_model')

    def __str__(self):
        """Return a string representation of the model. """
        return self.name


class Image(models.Model):
    img_file = models.ImageField(upload_to='binary_cnn/static/')
    classified = models.BooleanField(default=False)
    date_classified = models.DateTimeField(auto_now=True)
    img_classification = models.CharField(blank=True, max_length=100)
    prob_classification = models.DecimalField(null=True, max_digits=10, decimal_places=8)

    def __str__(self):
        """Return a string representation the type of model. """
        return self.img_file.name.split('/')[-1]


#  Clean-up, delete image file on delete of Image instances.
@receiver(pre_delete, sender=Image)
def delete_image_file(sender, instance, **kwargs):
    # Check if the image field has a value
    if instance.img_file:
        # Get the path of the image file
        image_path = instance.img_file.path
        # Check if the image file exists and delete it
        if os.path.exists(image_path):
            os.remove(image_path)



