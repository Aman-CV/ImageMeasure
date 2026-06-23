# models.py

import os
import uuid
import logging

from django.db import models
from django.db.models.signals import post_delete
from django.dispatch import receiver

logger = logging.getLogger(__name__)


def skill_video_name(instance, old_filename):
    extension = os.path.splitext(old_filename)[1]
    filename = str(uuid.uuid4()) + extension
    return 'skill_assessment_videos/' + filename


class SkillAssessment(models.Model):
    video = models.FileField(upload_to=skill_video_name)
    assessment_name = models.CharField(max_length=255)
    participant_name = models.CharField(max_length=255)
    test_name = models.CharField(max_length=255)
    is_registered = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.assessment_name} - {self.participant_name} ({self.test_name})"

    def register(self):
        """Register the (S3-hosted) video with the external assessment API."""
        # Imported lazily so this app does not pull in homography_app's heavy
        # module-level dependencies (torch, mobile_sam, cv2, ...) on import.
        from homography_app.helper import test_video_url

        test_video_url(
            assessment_id=self.assessment_name,
            test_id=self.test_name,
            participant_id=self.participant_name,
            vurl=self.video.url,
        )
        self.is_registered = True

    def save(self, *args, **kwargs):
        # First save uploads the file to S3 (default S3Boto3Storage backend).
        super().save(*args, **kwargs)

        # Register once, after the file is available at its S3 URL.
        if self.video and not self.is_registered:
            self.register()
            super().save(update_fields=["is_registered"])


@receiver(post_delete, sender=SkillAssessment)
def delete_files_on_model_delete(sender, instance, **kwargs):
    if instance.video:
        instance.video.delete(save=False)
