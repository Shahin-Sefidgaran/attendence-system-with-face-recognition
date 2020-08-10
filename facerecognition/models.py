from django.db import models

# Create your models here.

class employes(models.Model):
    employe_photo = models.FileField(null=False)
    employe_name = models.CharField(max_length=255)
    employe_id = models.CharField(max_length=10)
