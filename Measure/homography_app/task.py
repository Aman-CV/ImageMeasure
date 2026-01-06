import os
import logging
from background_task import background
from django.conf import settings
from django.core.files import File
import cv2
import json
import numpy as np

from .models import PetVideos, SingletonHomographicMatrixModel
from .helper import filter_and_smooth, detect_biggest_jump, \
    distance_from_homography, get_flat_start, \
    ankle_crop_color_detection, correct_white_balance, highest_peak_by_adjacent_minima
import glob
from scipy.signal import savgol_filter
from ultralytics import YOLO

from .sit_and_reach_helper_ import detect_yellow_strip_positions_mask, find_three_centers_from_mask, \
    estimate_distance_between_points, middle_finger_movement_distance
from .sit_and_throw_helper import get_first_bounce_frame, get_first_bounce_frame_MOG

logger = logging.getLogger('homography_app')


@background(schedule=0, remove_existing_tasks=True)
def process_sit_and_throw(petvideo_id, test_id=""):
    if test_id == "":
        logger.info(f"[process_video_task] INVALID TEST ID: {petvideo_id}")
        return
    logger.info(f"[process_video_task] Starting processing for PetVideo ID: {petvideo_id}")
    try:
        video_obj = PetVideos.objects.get(id=petvideo_id)
    except PetVideos.DoesNotExist:
        logger.error(f"[process_video_task] PetVideo ID {petvideo_id} does not exist")
        return
    # if not video_obj.to_be_processed:
    #     with open(video_obj.file.path, 'rb') as f:
    #         video_obj.is_video_processed = True
    #         video_obj.progress = 100
    #         video_obj.processed_file.save(os.path.basename(video_obj.file.name), File(f), save=True)
    #     logger.info(f"[process_video_task] Video requires no processing time recorded: {petvideo_id}")
    #     return
    video_path = video_obj.file.path
    if video_obj.processed_file:
        video_obj.processed_file.delete(save=False)
        video_obj.processed_file = None
    output_dir = os.path.join(settings.MEDIA_ROOT, 'post_processed_video')
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(video_obj.file.path, 'rb') as f:
            video_obj.processed_file.save(os.path.basename(video_obj.file.name), File(f), save=True)
            video_obj.is_video_processed = False
            video_obj.progress = 0


        cx, cy, cf = get_first_bounce_frame_MOG(video_path)
        if cx is None or cy is None:
            logger.error(f"[process_sit_and_throw] Error in detecting ball: {petvideo_id}")
            video_obj.distance = 0
            video_obj.is_video_processed = True
            video_obj.progress = 100
            video_obj.save()
            return
        print("s", cx, cy, cf)
        homograph_obj = SingletonHomographicMatrixModel.load()
        distance = round(homograph_obj.unit_distance *  abs(cx - homograph_obj.start_pixel) / abs(homograph_obj.start_pixel - homograph_obj.end_pixel),2)
        video_obj.distance = distance
        video_obj.is_video_processed = True
        video_obj.progress = 100
        original_name = os.path.basename(video_obj.file.name)

        final_output_path = os.path.join(output_dir, f"processed_{original_name}")

        import subprocess
        subprocess.run([
            'ffmpeg', '-i', "motion_output.mp4",
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-c:a', 'aac', '-movflags', '+faststart', '-y',
            final_output_path
        ], check=True)

        with open(final_output_path, 'rb') as f:
            video_obj.processed_file.save(f"processed_{original_name}", File(f), save=False)
        for path in [final_output_path]:
            if os.path.exists(path):
                os.remove(path)

        video_obj.save()


        logger.info(f"[process_video_task] Finished processing PetVideo ID: {petvideo_id}")

    except Exception as e:
        logger.error(f"[process_video_task] Error processing PetVideo ID {petvideo_id}: {e}", exc_info=True)


@background(schedule=0, remove_existing_tasks=True)
def process_sit_and_reach(petvideo_id, test_id=""):
    if len(test_id) == 0:
        logger.info(f"[process_video_task] INVALID TEST ID: {petvideo_id}")
        return
    if test_id == "vPbXoPK4":
        logger.info(f"[process_video_task] Starting processing for PetVideo ID (Sit and reach variant): {petvideo_id}")
        try:
            video_obj = PetVideos.objects.get(id=petvideo_id)
        except PetVideos.DoesNotExist:
            logger.error(f"[process_video_task] PetVideo ID {petvideo_id} does not exist")
            return
        video_path = video_obj.file.path
        if video_obj.processed_file:
            video_obj.processed_file.delete(save=False)
            video_obj.processed_file = None


        try:
            video_obj.is_video_processed = False
            video_obj.progress = 0
            homograph_obj = SingletonHomographicMatrixModel.load()

            distance = middle_finger_movement_distance(video_obj.file.path)

            with open(video_obj.file.path, 'rb') as f:
                video_obj.processed_file.save(os.path.basename(video_obj.file.name), File(f), save=False)

            if not distance:
                logger.info(f"[process_video_task] Finger tip detection failed: {petvideo_id}")
            else:
                print(distance)
            video_obj.distance = round(distance if distance else 0, 2)
            video_obj.is_video_processed = True
            video_obj.progress = 100
            video_obj.save()



            logger.info(f"[process_video_task] Finished processing PetVideo ID: {petvideo_id}")

        except Exception as e:
            logger.error(f"[process_video_task] Error processing PetVideo ID {petvideo_id}: {e}", exc_info=True)
        return


@background(schedule=0, remove_existing_tasks=True)
def process_video_task(petvideo_id, enable_color_marker_tracking=True, enable_start_end_detector=True, test_id=""):
    if test_id == "":
        logger.info(f"[process_video_task] INVALID TEST ID: {petvideo_id}")
        return
    if test_id == "vPbXoPK4":
        logger.info(f"[process_video_task] Starting processing for PetVideo ID (Sit and reach variant): {petvideo_id}")
        try:
            video_obj = PetVideos.objects.get(id=petvideo_id)
        except PetVideos.DoesNotExist:
            logger.error(f"[process_video_task] PetVideo ID {petvideo_id} does not exist")
            return
        video_path = video_obj.file.path
        original_name = os.path.basename(video_obj.file.name)
        if video_obj.processed_file:
            video_obj.processed_file.delete(save=False)
            video_obj.processed_file = None
        output_dir = os.path.join(settings.MEDIA_ROOT, 'post_processed_video')
        os.makedirs(output_dir, exist_ok=True)

        temp_output_path = os.path.join(output_dir, f"temp_{original_name}")
        final_output_path = os.path.join(output_dir, f"processed_{original_name}")
        try:
            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
            video_obj.is_video_processed = False
            video_obj.progress = 0
            cap = cv2.VideoCapture(video_path)
            homograph_obj = SingletonHomographicMatrixModel.load()
            mask_path = homograph_obj.mask.path
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, (1280, 720))
            distance = 1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (1280, 720))
                f1 = detect_yellow_strip_positions_mask(frame, mask_img, int(720 * 0.6))
                x = find_three_centers_from_mask(f1)

                centers_sorted = sorted(x, key=lambda c: c[0], reverse=True)
                distance = estimate_distance_between_points(centers_sorted)

                y = int(0.6 * height)
                dot_length = 10
                gap = 5

                for x in range(0, width, dot_length + gap):
                    cv2.line(frame, (x, y), (x + dot_length, y), (0, 255, 0), 2)
                frame = cv2.resize(frame, (width, height))
                out.write(frame)
            cap.release()
            out.release()
            import subprocess
            subprocess.run([
                'ffmpeg', '-i', temp_output_path,
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                '-c:a', 'aac', '-movflags', '+faststart', '-y',
                final_output_path
            ], check=True)

            with open(final_output_path, 'rb') as f:
                video_obj.processed_file.save(f"processed_{original_name}", File(f), save=False)

            if not distance:
                logger.info(f"[process_video_task] All markers not detected: {petvideo_id}")
            else:
                print(centers_sorted)
            video_obj.distance = round(distance if distance else 0, 2)
            video_obj.is_video_processed = True
            video_obj.progress = 100
            video_obj.save()

            for path in [temp_output_path, final_output_path]:
                if os.path.exists(path):
                    os.remove(path)

            logger.info(f"[process_video_task] Finished processing PetVideo ID: {petvideo_id}")

        except Exception as e:
            logger.error(f"[process_video_task] Error processing PetVideo ID {petvideo_id}: {e}", exc_info=True)
        return
    logger.info(f"[process_video_task] Starting processing for PetVideo ID: {petvideo_id}")
    logger.info(f"[process_video_task] color_marker_tracking is {enable_color_marker_tracking}")
    logger.info(f"[process_video_task] jump detection is {enable_start_end_detector}")
    try:
        video_obj = PetVideos.objects.get(id=petvideo_id)
    except PetVideos.DoesNotExist:
        logger.error(f"[process_video_task] PetVideo ID {petvideo_id} does not exist")
        return
    if not video_obj.to_be_processed:
        with open(video_obj.file.path, 'rb') as f:
            video_obj.is_video_processed = True
            video_obj.progress = 100
            video_obj.processed_file.save(os.path.basename(video_obj.file.name), File(f), save=True)
        logger.info(f"[process_video_task] Video requires no processing time recorded: {petvideo_id}")
        return
    video_path = video_obj.file.path
    original_name = os.path.basename(video_obj.file.name)
    if video_obj.processed_file:
        video_obj.processed_file.delete(save=False)
        video_obj.processed_file = None
    output_dir = os.path.join(settings.MEDIA_ROOT, 'post_processed_video')
    os.makedirs(output_dir, exist_ok=True)

    temp_output_path = os.path.join(output_dir, f"temp_{original_name}")
    final_output_path = os.path.join(output_dir, f"processed_{original_name}")
    try:
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        trajectory = []
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        model = YOLO("yolov8m-pose.pt")
        current_frame = 0
        last_logged_progress = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = correct_white_balance(frame)
            frame = cv2.resize(frame, (1280, 720))
            mask, ankle_points = ankle_crop_color_detection(frame, CLAHE=clahe, model=model)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_points = [1280, 720] if len(ankle_points) < 2 else list(ankle_points[-1])
            offset = 5 #pixel
            detected_points[-1] = offset + detected_points[-1]
            for cnt in contours:
                if cnt is None or len(cnt) == 0:
                    continue

                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    if detected_points[1] < cy or cy < 720 // 2:
                        if enable_color_marker_tracking:
                            detected_points = [cx, cy]

            # cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

            trajectory.append(detected_points)
            current_frame += 1
            if total_frames > 0:
                progress = int((current_frame / total_frames) * 100)
                if progress >= last_logged_progress + 10:
                    video_obj.progress = progress
                    video_obj.save(update_fields=["progress"])
                    last_logged_progress = progress

        cap.release()

        cap = cv2.VideoCapture(video_path)
        traj_cnt = 0
        trajectory = filter_and_smooth(trajectory, threshold=10)
        np.save("trajectory.npy", trajectory)
        y_smooth = savgol_filter(trajectory[:, 1], window_length=11, polyorder=2)
        y_ns = max(y_smooth) - y_smooth
        p, st, end1, x1, x2 = highest_peak_by_adjacent_minima(y_ns, trajectory[:, 0])
        print(p, st, end1, x1, st)
        dy = np.gradient(y_smooth)
        limit_cut = max(dy) / 10
        dy[np.logical_and(dy > -limit_cut, dy < limit_cut)] = 0
        success, [f1, f2] = get_flat_start(dy, window=len(dy) // 10)
        print(f1, f2)
        if not success:
            logger.info(f"[process_video_task] Info processing PetVideo ID {petvideo_id}: flats not deteced")
        start, end = detect_biggest_jump(dy[f1[1]: f2[1]] if success else dy)

        if success and start and end and enable_start_end_detector:
            if st is None or end1 is None:
                end, start = end + f1[1], start + f1[1]
            else:
                start, end = st, end1
                print(start, end)
        else:
            start, end = 0, len(trajectory) - 1
        pt1 = trajectory[start if start else 0, :]
        pt2 = trajectory[end if end else len(trajectory) - 1, :]
        print(pt1, pt2)
        pt1[0] -= 5
        pt2[0] -= 5
        y_offset = 10
        pt1[-1] += y_offset
        pt2[-1] += y_offset #offset correction to get marked point closer to ground and heels
        print(pt1, pt2)
        trajectory = [tuple(map(int, point)) for point in trajectory]
        #sorted_points = sorted(trajectory, key=lambda p: p[1], reverse=True)


        # folder_path = "../media/homograph"  # <-- change this
        # files = glob.glob(os.path.join(folder_path, "homography_*.json"))
        #
        # if not files:
        #     raise FileNotFoundError("No file found matching homography_*.json")
        #
        # latest_file = max(files, key=os.path.getmtime)

        homograph_obj = SingletonHomographicMatrixModel.load()


        # if homograph_obj.start_pixel_broad_jump != 1:
        #     pt1[0] = homograph_obj.start_pixel_broad_jump
        # distance_ft = round(distance_from_homography(pt1, pt2, H), 2)
        distance_ft = round(homograph_obj.unit_distance * abs(pt2[0] - homograph_obj.start_pixel) / abs(
            homograph_obj.start_pixel - homograph_obj.end_pixel), 2)
        pt2[1] = pt1[1]
        # img_line = np.array([[trajectory[start], trajectory[end]]], dtype=np.float32)
        # world_line = cv2.perspectiveTransform(img_line, H)[0]
        # p1, p2 = world_line
        # vec = p2 - p1

        # length = np.linalg.norm(vec)
        # unit_vec = vec / length if length != 0 else 1
        # num_marks = int(length)  # one marker per foot
        # scale_world = np.array([p1 + i * unit_vec for i in range(num_marks + 1)], dtype=np.float32).reshape(-1, 1, 2)
        # H_inv = np.linalg.inv(H)
        # scale_img = cv2.perspectiveTransform(scale_world, H_inv)
        # scale_img = scale_img[:, 0, :].astype(int)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # frame = correct_white_balance(frame)
            frame = cv2.resize(frame, (1280, 720))
            if start <= traj_cnt <= end:
                overlay = frame.copy()

                overlay[:] = (0, 0, 160)  # BGR red
                alpha = 0.3  # transparency factor
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.circle(frame, trajectory[traj_cnt], 20, (0, 255, 0), 2)
            traj_cnt += 1
            cv2.line(frame, trajectory[start], trajectory[end], (0, 0, 0), 2)

            # for i, (x, y) in enumerate(scale_img):
            #     if 0 <= x < width and 0 <= y < height:
            #         cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            #         cv2.putText(frame, f"{i}ft", (x + 6, y - 6),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            frame = cv2.resize(frame, (width, height))
            out.write(frame )
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

        with open(final_output_path, 'rb') as f:
            video_obj.processed_file.save(f"processed_{original_name}", File(f), save=False)

        video_obj.distance = distance_ft / 3.281
        video_obj.is_video_processed = True
        video_obj.progress = 100
        video_obj.save()

        for path in [temp_output_path, final_output_path]:
            if os.path.exists(path):
                os.remove(path)

        logger.info(f"[process_video_task] Finished processing PetVideo ID: {petvideo_id}")

    except Exception as e:
        logger.error(f"[process_video_task] Error processing PetVideo ID {petvideo_id}: {e}", exc_info=True)


@background(schedule=0, remove_existing_tasks=True)
def process_broad_jump(petvideo_id):
    pass