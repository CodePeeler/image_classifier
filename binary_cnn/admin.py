from django.contrib import admin

# Register your models here.
from .models import BinaryModel, TrainingConfig, Image
admin.site.register(BinaryModel)
admin.site.register(TrainingConfig)
admin.site.register(Image)
