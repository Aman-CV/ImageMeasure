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
    root = os.path.join(settings.TEMP_STORAGE, UPLOAD_DIR)

    for dp, dn, fn in os.walk(root):
        for f in fn:
            rel_path = os.path.relpath(os.path.join(dp, f), settings.TEMP_STORAGE)
            all_files.add(rel_path)

    stray_files = all_files - used_files
    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv'}

    for f in stray_files:
        if not any(f.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
            continue  # skip non-video files

        print(f)
        default_storage.delete(f)

    return stray_files


def pf():
    for video in PetVideos.objects.all():
        if video.file:
            print("Original file URL:", video.file.url)
        if video.processed_file:
            print("Processed file URL:", video.processed_file.url)
