from homography_app.models import PetVideos
from homography_app.task import process_video_task
from django.conf import settings
from django.core.files.storage import default_storage
from django.apps import apps
import os

def schedule_all_videos():
    for video in PetVideos.objects.all():
        process_video_task(video.id)

def delete_orphan_petvideos():
    UPLOAD_DIR = "/Users/notcamelcase/PycharmProjects/ImageMeasure/Measure/media/videos/"

    used_files = set(
        PetVideos.objects.exclude(file='').values_list('file', flat=True)
    )

    all_files = set()
    root = os.path.join(settings.MEDIA_ROOT, UPLOAD_DIR)

    for dp, dn, fn in os.walk(root):
        for f in fn:
            rel_path = os.path.relpath(os.path.join(dp, f), settings.MEDIA_ROOT)
            all_files.add(rel_path)

    stray_files = all_files - used_files
    for f in stray_files:
        print(f)
        # default_storage.delete(f)

    return stray_files
