from django.contrib import admin

# Register your models here.
from .models import Dataset, BinaryModel, TrainConfig, TrainingKpi, Image
admin.site.register(BinaryModel)
admin.site.register(Image)
admin.site.register(TrainConfig)
admin.site.register(TrainingKpi)
admin.site.register(Dataset)
