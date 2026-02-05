# models.py

from django.db.models import CheckConstraint, Q
import os
import uuid
from django.db import models
from django.db.models.signals import post_delete
from django.dispatch import receiver



def processed_file_name(instance, old_filename):
    extension = os.path.splitext(old_filename)[1]
    filename = str(uuid.uuid4()) + extension
    return 'post_processed_video/' + filename

def file_name(instance, old_filename):
    extension = os.path.splitext(old_filename)[1]
    filename = str(uuid.uuid4()) + extension
    return 'videos/' + filename

class PetVideos(models.Model):
    name = models.CharField(max_length=255)
    is_video_processed = models.BooleanField(default=False)
    participant_name = models.CharField(max_length=255, default="NoName")
    participant_id = models.CharField(max_length=64, default="dummy")
    file = models.FileField(upload_to=file_name)
    distance = models.FloatField(default=0)
    duration = models.FloatField(default=0)
    pet_type = models.CharField(max_length=32, default="STANDING_JUMP")
    processed_file = models.FileField(upload_to=processed_file_name, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    progress = models.PositiveSmallIntegerField(default=0)
    to_be_processed = models.BooleanField(default=False)
    assessment_id = models.CharField(max_length=64, default="dummy")
    test_id = models.CharField(max_length=64, default="jump")

    def __str__(self):
        return self.name

    def run_processing(self):
        """
        Decides which processing function to call
        based on test_id.
        """

        from .task import (
            process_sit_and_reach,
            process_sit_and_throw,
            process_video_task,
            process_15m_dash,
        process_plank
        )

        if self.test_id in ("vPbXoPK4", "reach"):
            return process_sit_and_reach(
                self.id,
                test_id=self.test_id,
                assessment_id=self.assessment_id,
            )

        elif self.test_id in ("BwbJyXKl", "throw"):
            return process_sit_and_throw(
                self.id,
                test_id=self.test_id,
                assessment_id=self.assessment_id,
            )
        elif self.test_id in ("lzb1PEKm", "15run"):
            return process_15m_dash(self.id, test_id=self.test_id,assessment_id=self.assessment_id)

        elif self.test_id in ("Vnb7E6L6", "6x10run"):
            return process_plank(self.id, test_id=self.test_id, assessment_id=self.assessment_id)

        return process_video_task(
            self.id,
            test_id=self.test_id,
            assessment_id=self.assessment_id,
        )


@receiver(post_delete, sender=PetVideos)
def delete_files_on_model_delete(sender, instance, **kwargs):
    for field in ['file', 'processed_file']:
        file_field = getattr(instance, field)
        if file_field:
            file_field.delete(save=False)

# @receiver(post_delete, sender=PetVideos)
# def delete_files_on_model_delete(sender, instance, **kwargs):
#     for field in ['file', 'processed_file']:
#         file_field = getattr(instance, field)
#         if file_field and file_field.name:
#             if os.path.isfile(file_field.path):
#                 os.remove(file_field.path)



class CalibrationDataModel(models.Model):
    assessment_id = models.CharField(max_length=64, default="dummy")
    test_id = models.CharField(max_length=64, default="jump")
    start_pixel = models.IntegerField(default=0)
    end_pixel = models.IntegerField(default=1)
    unit_distance = models.FloatField(default=2.5908)
    use_homograph = models.BooleanField(default=False)
    homography_points = models.JSONField(default=dict)
    origin_x = models.IntegerField(default=0)
    origin_y = models.IntegerField(default=0)

    def __str__(self):
        return self.test_id


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
