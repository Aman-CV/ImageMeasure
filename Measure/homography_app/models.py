# models.py
from django.db import models

class PetVideos(models.Model):
    name = models.CharField(max_length=255)
    participant_name = models.CharField(max_length=255, default="NoName")
    file = models.FileField(upload_to='videos/')
    distance = models.FloatField(default=0)
    pet_type = models.CharField(max_length=32,  default="STANDING_JUMP")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
