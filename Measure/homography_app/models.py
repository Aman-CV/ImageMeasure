# models.py
from django.db import models
from django.db.models import CheckConstraint, Q

class PetVideos(models.Model):
    name = models.CharField(max_length=255)
    participant_name = models.CharField(max_length=255, default="NoName")
    file = models.FileField(upload_to='videos/')
    distance = models.FloatField(default=0)
    pet_type = models.CharField(max_length=32,  default="STANDING_JUMP")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class SingletonHomographicMatrixModel(models.Model):
    # homograph is in media/homograph
    matrix = models.FileField(upload_to='homograph/')
    file = models.FileField(upload_to="calibrated_images/")
    unit_distance = models.IntegerField(default=60)
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
