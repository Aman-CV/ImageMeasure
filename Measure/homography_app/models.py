# models.py

from django.db.models import CheckConstraint, Q
import os
from django.db import models
from django.db.models.signals import post_delete
from django.dispatch import receiver

class PetVideos(models.Model):
    name = models.CharField(max_length=255)
    is_video_processed = models.BooleanField(default=False)
    participant_name = models.CharField(max_length=255, default="NoName")
    participant_id = models.CharField(max_length=64, default="dummy")
    file = models.FileField(upload_to='videos/')
    distance = models.FloatField(default=0)
    duration = models.FloatField(default=0)
    pet_type = models.CharField(max_length=32, default="STANDING_JUMP")
    processed_file = models.FileField(upload_to='post_processed_video/', blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    progress = models.PositiveSmallIntegerField(default=0)
    to_be_processed = models.BooleanField(default=False)
    assessment_id = models.CharField(max_length=64, default="dummy")
    test_id = models.CharField(max_length=64, default="jump")

    def __str__(self):
        return self.name

@receiver(post_delete, sender=PetVideos)
def delete_files_on_model_delete(sender, instance, **kwargs):
    for field in ['file', 'processed_file']:
        file_field = getattr(instance, field)
        if file_field and file_field.name:
            if os.path.isfile(file_field.path):
                os.remove(file_field.path)



class CalibrationDataModel(models.Model):
    assessment_id = models.CharField(max_length=64, default="dummy")
    test_id = models.CharField(max_length=64, default="jump")
    start_pixel = models.IntegerField(default=0)
    end_pixel = models.IntegerField(default=1)
    unit_distance = models.FloatField(default=2.5908)


class SingletonHomographicMatrixModel(models.Model):
    # homograph is in media/homograph
    matrix = models.FileField(upload_to='homograph/')
    file = models.FileField(upload_to="calibrated_images/")
    mask = models.FileField(upload_to="masks/", blank=True, null=True)
    unit_distance = models.FloatField(default=60)
    hsv_value = models.JSONField(default=dict)
    tracker_hsv_value = models.JSONField(default=dict)
    start_pixel = models.IntegerField(default=0)
    end_pixel = models.IntegerField(default=1)
    start_pixel_broad_jump = models.IntegerField(default = 1)
    class Meta:
        constraints = [
            CheckConstraint(
                check=Q(id=1),
                name='only_one_instance'
            )
        ]

    def save(self, *args, **kwargs):
        if not self.pk and SingletonHomographicMatrixModel.objects.exists():
            raise Exception("Only one instance of SingletonModel is allowed.")
        self.pk = 1
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        raise Exception("Deletion of SingletonModel instances is not allowed.")

    @classmethod
    def load(cls):
        obj, created = cls.objects.get_or_create(pk=1)
        return obj
