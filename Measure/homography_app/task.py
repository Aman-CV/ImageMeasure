import os
import logging
from background_task import background
from django.conf import settings
from django.core.files import File
import cv2
import numpy as np
from .models import PetVideos
from .helper import merge_close_points, DEFAULT_HSV, TOL_S, TOL_H, TOL_V

logger = logging.getLogger('homography_app')

@background(schedule=0, remove_existing_tasks=True)
def process_video_task(petvideo_id):
    logger.info(f"[process_video_task] Starting processing for PetVideo ID: {petvideo_id}")

    try:
        video_obj = PetVideos.objects.get(id=petvideo_id)
    except PetVideos.DoesNotExist:
        logger.error(f"[process_video_task] PetVideo ID {petvideo_id} does not exist")
        return

    video_path = video_obj.file.path
    original_name = os.path.basename(video_obj.file.name)

    output_dir = os.path.join(settings.MEDIA_ROOT, 'post_processed_video')
    os.makedirs(output_dir, exist_ok=True)

    temp_output_path = os.path.join(output_dir, f"temp_{original_name}")
    final_output_path = os.path.join(output_dir, f"processed_{original_name}")

    try:
        # --- OpenCV write ---
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # HSV detection
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = DEFAULT_HSV
            lower = np.array([max(h - TOL_H, 0), max(s - TOL_S, 0), max(v - TOL_V, 0)])
            upper = np.array([min(h + TOL_H, 179), min(s + TOL_S, 255), min(v + TOL_V, 255)])
            mask = cv2.inRange(hsv_frame, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detected_points = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detected_points.append((cx, cy))
            detected_points = merge_close_points(detected_points, threshold=40)
            for pt in detected_points[:4 if len(detected_points) > 4 else len(detected_points)]:
                cv2.circle(frame, pt, 15, (0, 0, 255), -1)

            out.write(frame)

        cap.release()
        out.release()

        # --- ffmpeg encode ---
        import subprocess
        subprocess.run([
            'ffmpeg', '-i', temp_output_path,
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-c:a', 'aac', '-movflags', '+faststart', '-y',
            final_output_path
        ], check=True)

        # Save ffmpeg output to model
        with open(final_output_path, 'rb') as f:
            video_obj.processed_file.save(f"processed_{original_name}", File(f), save=False)

        video_obj.distance = 10.9
        video_obj.is_video_processed = True
        video_obj.save()

        # Cleanup temp files
        for path in [temp_output_path, final_output_path]:
            if os.path.exists(path):
                os.remove(path)

        logger.info(f"[process_video_task] Finished processing PetVideo ID: {petvideo_id}")

    except Exception as e:
        logger.error(f"[process_video_task] Error processing PetVideo ID {petvideo_id}: {e}")
