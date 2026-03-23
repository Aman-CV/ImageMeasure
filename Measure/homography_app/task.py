import os
import logging
import subprocess
from background_task import background
from django.conf import settings
from django.core.files import File
import cv2
import numpy as np

from .checkpoint_crossing import detect_crossing_rightmost_ankle, detect_crossing_person_box, \
    detect_crossing_person_box_reverse_nobuffer, write_video_until_frame
from .models import PetVideos, SingletonHomographicMatrixModel, CalibrationDataModel
from .helper import filter_and_smooth, detect_biggest_jump, \
    get_flat_start, \
    ankle_crop_color_detection, correct_white_balance, highest_peak_by_adjacent_minima, test_video_url, \
    image_point_to_real_point
from scipy.signal import savgol_filter
from ultralytics import YOLO

from .plank_helper import mark_right_side_pose
from .sit_and_reach_helper_ import middle_finger_movement_distance
from .sit_and_throw_helper import get_first_bounce_frame_MOG

logger = logging.getLogger('homography_app')

def _legacy_video_path(video_obj):
    ext = os.path.splitext(os.path.basename(video_obj.file.name))[1]
    return os.path.join(settings.TEMP_VIDEO_STORAGE, f"videot_{video_obj.id}{ext}")


def _canonical_video_path(video_obj):
    return os.path.join(settings.TEMP_VIDEO_STORAGE, f"videot_{video_obj.id}.mp4")


def _remove_files(*paths):
    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)


def _cleanup_local_video(video_obj):
    _remove_files(_legacy_video_path(video_obj), _canonical_video_path(video_obj))


def _encode_to_h264(input_path, output_path):
    subprocess.run([
        'ffmpeg', '-i', input_path,  "-loglevel", "error",
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
        '-c:a', 'aac', '-movflags', '+faststart', '-y',
        output_path
    ],     check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL)


def _ensure_local_video(video_obj):
    """Stage 1: make sure a local processing video is available and return its path."""
    mp4_path = _canonical_video_path(video_obj)
    legacy_path = _legacy_video_path(video_obj)
    if os.path.exists(mp4_path):
        return mp4_path
    if os.path.exists(legacy_path):
        return legacy_path

    downloaded_path = download_and_save_video(video_obj)
    if os.path.exists(downloaded_path):
        return downloaded_path

    # Fallback for older workflows.
    if os.path.exists(legacy_path):
        return legacy_path
    raise FileNotFoundError(f"Local video was not created for id={video_obj.id}")


def _save_processed_file(video_obj, output_path, original_name):
    if video_obj.processed_file:
        video_obj.processed_file.delete(save=False)
        video_obj.processed_file = None
    with open(output_path, 'rb') as f:
        video_obj.processed_file.save(original_name, File(f), save=False)


def download_and_save_video(obj):
    video = obj.file

    os.makedirs(settings.TEMP_VIDEO_STORAGE, exist_ok=True)

    ext = os.path.splitext(video.name)[1] or ".mp4"

    raw_path = os.path.join(
        settings.TEMP_VIDEO_STORAGE,
        f"raw_{obj.id}{ext}"
    )

    with video.open("rb") as src, open(raw_path, "wb") as destination:
        for chunk in src.chunks():
            destination.write(chunk)
        destination.flush()
        os.fsync(destination.fileno())

    print("Raw video size:", os.path.getsize(raw_path))

    final_path = os.path.join(
        settings.TEMP_VIDEO_STORAGE,
        f"videot_{obj.id}.mp4"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-loglevel", "error",
            "-i", raw_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-y",
            final_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    _remove_files(raw_path)
    print("Deleted raw upload:", raw_path)
    return final_path



@background(schedule=0, remove_existing_tasks=True)
def process_sit_and_throw(petvideo_id, test_id="", assessment_id=""):
    if test_id == "" or assessment_id == "":
        logger.info(f"[process_video_task] INVALID TEST/ASSESSMENT ID: {petvideo_id}")
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
    video_path = _ensure_local_video(video_obj)

    output_dir = os.path.join(settings.TEMP_STORAGE, 'post_processed_video')
    os.makedirs(output_dir, exist_ok=True)
    try:
        video_obj.is_video_processed = False
        video_obj.progress = 0
        homograph_obj = CalibrationDataModel.objects.filter(
            assessment_id=assessment_id,
            test_id=test_id
        ).first()
        use_homograph = homograph_obj.use_homograph if homograph_obj else False
        if not homograph_obj:
            homograph_obj = SingletonHomographicMatrixModel.load()
        opth = f"motion_output_{petvideo_id}.mp4"
        cx, cy, cf = get_first_bounce_frame_MOG(video_path,
                    start_cutoff=(homograph_obj.origin_x + 10.0)/1280.0 if homograph_obj.start_pixel !=0 else 0.25, video_obj=video_obj, output_pth=opth)
        if cx is None or cy is None:
            logger.error(f"[process_sit_and_throw] Error in detecting ball: {petvideo_id}")
            # video_obj.distance = 0
            video_obj.is_video_processed = True
            video_obj.progress = 100
            video_obj.save()
            return
        print("s", cx, cy, cf)

        distance = homograph_obj.unit_distance *  abs(cx - homograph_obj.start_pixel) / abs(homograph_obj.start_pixel - homograph_obj.end_pixel)
        rp1 = None
        pt1 = [cx, cy + 5]
        rp2 = None
        if use_homograph:
            rp1 = image_point_to_real_point(homograph_obj.homography_points, homograph_obj.unit_distance, pt1)
            if homograph_obj.origin_y != 0 and homograph_obj.origin_x != 0:
                origin = np.array([homograph_obj.origin_x, homograph_obj.origin_y])
                rp2 = image_point_to_real_point(homograph_obj.homography_points, homograph_obj.unit_distance, origin)

        if rp1 and rp2 and use_homograph:
            distance = np.sqrt((rp1[0] - rp2[0]) ** 2 + (rp1[1] - rp2[1]) ** 2)

        # video_obj.distance = distance
        is_changed = video_obj.update_metrix(0, distance)
        video_obj.is_video_processed = True
        video_obj.progress = 100
        original_name = os.path.basename(video_obj.file.name)

        final_output_path = f"temp_media_store/processed_{original_name}"

        _encode_to_h264(opth, final_output_path)
        if is_changed:
            _save_processed_file(video_obj, final_output_path, original_name)
        _remove_files(final_output_path, opth)
        _cleanup_local_video(video_obj)
        video_obj.save()

        test_video_url(assessment_id, test_id, video_obj.participant_id, video_obj.processed_file.url)
        logger.info(f"[process_video_task] Finished processing PetVideo ID: {petvideo_id}")

    except Exception as e:
        logger.error(f"[process_video_task] Error processing PetVideo ID {petvideo_id}: {e}", exc_info=True)


@background(schedule=0, remove_existing_tasks=True)
def process_sit_and_reach(petvideo_id, test_id="", assessment_id=""):
    if len(test_id) == 0 or len(assessment_id) == 0:
        logger.info(f"[process_video_task] INVALID TEST/ ASSESSMENT ID: {petvideo_id}")
        return
    if test_id == "vPbXoPK4" or test_id == "reach":
        logger.info(f"[process_video_task] Starting processing for PetVideo ID (Sit and reach variant): {petvideo_id}")
        try:
            video_obj = PetVideos.objects.get(id=petvideo_id)
        except PetVideos.DoesNotExist:
            logger.error(f"[process_video_task] PetVideo ID {petvideo_id} does not exist")
            return
        video_path = _ensure_local_video(video_obj)

        try:
            video_obj.is_video_processed = False
            video_obj.progress = 0
            homograph_obj = CalibrationDataModel.objects.filter(
                assessment_id=assessment_id,
                test_id=test_id
            ).first()
            use_homograph = homograph_obj.use_homograph if homograph_obj else False
            if not homograph_obj:
                homograph_obj = SingletonHomographicMatrixModel.load()
            opth = f"motion_output_{petvideo_id}.mp4"
            distance, pt1, pt2 = middle_finger_movement_distance(video_path, video_obj=video_obj, output_pth=opth, target_y=[homograph_obj.origin_x, homograph_obj.origin_y] if homograph_obj.origin_y != 0 else None)

            if not distance:
                distance = 0
            print(distance)
            distance = None
            rp1 = None
            rp2 = None
            if use_homograph and pt1 and pt2:
                rp2 = image_point_to_real_point(homograph_obj.homography_points, homograph_obj.unit_distance, pt2)
                if use_homograph and homograph_obj.origin_x != 0 and homograph_obj.origin_y != 0:
                    origin = np.array([homograph_obj.origin_x, homograph_obj.origin_y])
                    rp1 = image_point_to_real_point(homograph_obj.homography_points, homograph_obj.unit_distance,
                                                    origin)
            if use_homograph and rp1 and rp2:
                distance = np.sqrt((rp2[0] - rp1[0]) ** 2 + (rp2[1] - rp1[1]) ** 2) + 0.5
                if homograph_obj.origin_x < pt2[0]:
                    distance = 15.0 + distance
                else:
                    distance = 15.0 - distance

            original_name = os.path.basename(video_obj.file.name)
            final_output_path = f"temp_media_store/processed_{original_name}"

            _encode_to_h264(opth, final_output_path)


            #----#
            #with open(video_path, 'rb') as f:
            #    video_obj.processed_file.save(os.path.basename(video_obj.file.name), File(f), save=False)

            if not distance:
                logger.info(f"[process_video_task] Finger tip detection failed: {petvideo_id}")
            else:
                print(distance)
            is_changed = False
            if distance:
                is_changed = video_obj.update_metrix(0, distance)
            if is_changed:
                _save_processed_file(video_obj, final_output_path, original_name)
            _remove_files(final_output_path)
            video_obj.is_video_processed = True
            video_obj.progress = 100
            video_obj.save()
            _remove_files(opth)
            _cleanup_local_video(video_obj)
            test_video_url(assessment_id, test_id, video_obj.participant_id, video_obj.processed_file.url)
            logger.info(f"[process_video_task] Finished processing PetVideo ID: {petvideo_id}")

        except Exception as e:
            logger.error(f"[process_video_task] Error processing PetVideo ID {petvideo_id}: {e}", exc_info=True)
        return


@background(schedule=0, remove_existing_tasks=True)
def process_video_task(petvideo_id, enable_color_marker_tracking=True, enable_start_end_detector=True, test_id="", assessment_id=""):
    if test_id == "" or assessment_id == "":
        logger.info(f"[process_video_task] INVALID TEST/ ASSESSMENT ID: {petvideo_id}")
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
        # ext = os.path.splitext(os.path.basename(video_obj.file.name))[1]
        # video_path = os.path.join(
        #     settings.TEMP_VIDEO_STORAGE,
        #     f"videot_{petvideo_id}{ext}"
        # )
        # if not os.path.exists(video_path):
        #     download_and_save_video(video_obj)
        # with open(video_path, 'rb') as f:
        #     video_obj.is_video_processed = True
        #     video_obj.progress = 100
        #     video_obj.processed_file.save(os.path.basename(video_obj.file.name), File(f), save=True)
        # ext = os.path.splitext(os.path.basename(video_obj.file.name))[1]
        # file_path = os.path.join(
        #     settings.TEMP_VIDEO_STORAGE,
        #     f"videot_{petvideo_id}{ext}"
        # )
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        # test_video_url(assessment_id, test_id, video_obj.participant_id, video_obj.processed_file.url)
        logger.info(f"[process_video_task] Video requires no processing time recorded: {petvideo_id}")
        return
    video_path = _ensure_local_video(video_obj)
    original_name = os.path.basename(video_obj.file.name)

    output_dir = "temp_media_store"
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

        homograph_obj = CalibrationDataModel.objects.filter(
            assessment_id=assessment_id,
            test_id=test_id
        ).first()

        use_homograph = homograph_obj.use_homograph if homograph_obj else False

        if not homograph_obj:
            print("Calibration was not successful")
            homograph_obj = SingletonHomographicMatrixModel.load()
            use_homograph = False


        print(homograph_obj.start_pixel, homograph_obj.end_pixel, pt2[0])
        # if homograph_obj.start_pixel_broad_jump != 1:
        #     pt1[0] = homograph_obj.start_pixel_broad_jump
        # distance_ft = distance_from_homography(pt1, pt2, H)
        distance_ft = homograph_obj.unit_distance * abs(pt2[0] - homograph_obj.start_pixel) / abs(
            homograph_obj.start_pixel - homograph_obj.end_pixel)
        rp1 = None
        rp2 = None
        if use_homograph:
            rp1 = image_point_to_real_point(homograph_obj.homography_points, homograph_obj.unit_distance,pt2)
            if homograph_obj.origin_x != 0 and homograph_obj.origin_y != 0:
                origin = np.array([homograph_obj.origin_x, homograph_obj.origin_y])
                rp2 = image_point_to_real_point(homograph_obj.homography_points, homograph_obj.unit_distance, origin)
        if rp1 and rp2 and use_homograph:
            distance_ft = np.sqrt((rp1[0] - rp2[0]) ** 2 + (rp1[1] - rp2[1]) ** 2)
        if homograph_obj.origin_y and homograph_obj.origin_y != 0:
            pt2[1] = homograph_obj.origin_y
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
            #cv2.circle(frame, trajectory[traj_cnt], 20, (0, 255, 0), 2)

            traj_cnt += 1
            if homograph_obj and homograph_obj.origin_x != 0 and homograph_obj != 0:
                if not success:
                    cv2.line(frame, [homograph_obj.origin_x, homograph_obj.origin_y], trajectory[end], (0, 0, 0), 2)
                    cv2.circle(frame, [homograph_obj.origin_x, homograph_obj.origin_y], 2, (0, 255, 0), 2)
                    cv2.circle(frame, trajectory[end], 2, (0, 255, 0), 2)
                    cv2.putText(frame, f"{round(distance_ft, 3)}m",
                                (homograph_obj.origin_x + 20, homograph_obj.origin_y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                elif  end and traj_cnt > end:
                    cv2.line(frame, [homograph_obj.origin_x, homograph_obj.origin_y], trajectory[end], (0, 0, 0), 2)
                    cv2.circle(frame, [homograph_obj.origin_x, homograph_obj.origin_y], 2, (0, 255, 0), 2)
                    cv2.circle(frame, trajectory[end], 2, (0, 255, 0), 2)
                    cv2.putText(frame, f"{round(distance_ft,3)}m", (homograph_obj.origin_x + 20, homograph_obj.origin_y - 6),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # for i, (x, y) in enumerate(scale_img):
            #     if 0 <= x < width and 0 <= y < height:
            #         cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            #         cv2.putText(frame, f"{i}ft", (x + 6, y - 6),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            frame = cv2.resize(frame, (width, height))
            out.write(frame)
        cap.release()
        out.release()
        # --- ffmpeg encode ---
        _encode_to_h264(temp_output_path, final_output_path)



        # video_obj.distance = distance_ft
        video_obj.is_video_processed = True
        video_obj.progress = 100
        is_changed = video_obj.update_metrix(0, distance_ft)
        if is_changed:
            _save_processed_file(video_obj, final_output_path, original_name)
        video_obj.save()
        _cleanup_local_video(video_obj)
        _remove_files(temp_output_path, final_output_path)
        test_video_url(assessment_id, test_id, video_obj.participant_id, video_obj.processed_file.url)
        logger.info(f"[process_video_task] Finished processing PetVideo ID: {petvideo_id}")

    except Exception as e:
        logger.error(f"[process_video_task] Error processing PetVideo ID {petvideo_id}: {e}", exc_info=True)


@background(schedule=0, remove_existing_tasks=True)
def process_15m_dash(petvideo_id, test_id, assessment_id):
    process_ttest_6x15_dash(petvideo_id, test_id, assessment_id)
    return
    logger.info(f"[process_video_task] STARTED TEST/ ASSESSMENT ID: {petvideo_id}")
    if test_id == "" or assessment_id == "":
        logger.info(f"[process_video_task] INVALID TEST/ ASSESSMENT ID: {petvideo_id}")
        return
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
    video_path = _ensure_local_video(video_obj)
    if video_obj.processed_file:
        video_obj.processed_file.delete(save=False)
        video_obj.processed_file = None
    #with open(video_path, 'rb') as f:
    #    video_obj.processed_file.save(os.path.basename(video_obj.file.name), File(f), save=True)
    output_dir = os.path.join(settings.TEMP_STORAGE, 'post_processed_video')
    os.makedirs(output_dir, exist_ok=True)
    try:
        video_obj.is_video_processed = False
        video_obj.progress = 0
        homograph_obj = CalibrationDataModel.objects.filter(
            assessment_id=assessment_id,
            test_id=test_id
        ).first()
        use_homograph = False
        if not homograph_obj:
            homograph_obj = SingletonHomographicMatrixModel.load()
        fno, duration, _ = detect_crossing_rightmost_ankle(video_path, homograph_obj.end_pixel, reference=homograph_obj.unit_distance,show=False)
        print(fno, duration)
        if not duration:
            duration = 0
        video_obj.duration = duration - 3.5

        original_name = os.path.basename(video_obj.file.name)

        final_output_path = f"temp_media_store/processed_{original_name}"
        if duration < 0.5:
            logger.info(f"[process_video_task] Detection failed retrying {petvideo_id}")
            fno, duration, _ = detect_crossing_person_box(video_path, homograph_obj.end_pixel, show=False, reference=homograph_obj.unit_distance)
            if duration and duration > 1:
                video_obj.duration = duration - 3.5
                pass
            else:
                with open(video_path, 'rb') as f:
                    video_obj.is_video_processed = True
                    video_obj.progress = 100
                    video_obj.processed_file.save(os.path.basename(video_obj.file.name), File(f), save=True)
                _cleanup_local_video(video_obj)
                logger.info(f"[process_video_task] Detection failed {petvideo_id}")
                return
        video_obj.distance = homograph_obj.unit_distance
        video_obj.is_video_processed = True
        video_obj.progress = 100
        _encode_to_h264("motion_output.mp4", final_output_path)

        _save_processed_file(video_obj, final_output_path, original_name)
        _remove_files(final_output_path)
        _cleanup_local_video(video_obj)
        video_obj.save()
        test_video_url(assessment_id, test_id, participant_id=video_obj.participant_id, vurl=video_obj.processed_file.url)
        logger.info(f"[process_video_task] Done processing PetVideo ID {petvideo_id}")

    except Exception as e:
        logger.error(f"[process_video_task] Error processing PetVideo ID {petvideo_id}: {e}", exc_info=True)


@background(schedule=0, remove_existing_tasks=True)
def process_plank(petvideo_id, test_id, assessment_id):
    logger.info(f"[process_video_task] STARTED PLANK/ ASSESSMENT ID: {petvideo_id}")
    if test_id == "" or assessment_id == "":
        logger.info(f"[process_video_task] INVALID TEST/ ASSESSMENT ID: {petvideo_id}")
        return
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
    video_path = _ensure_local_video(video_obj)
    if video_obj.processed_file:
        video_obj.processed_file.delete(save=False)
        video_obj.processed_file = None
    # with open(video_path, 'rb') as f:
    #    video_obj.processed_file.save(os.path.basename(video_obj.file.name), File(f), save=True)
    output_dir = os.path.join(settings.TEMP_STORAGE, 'post_processed_video')
    os.makedirs(output_dir, exist_ok=True)
    try:
        video_obj.is_video_processed = False
        video_obj.progress = 0


        original_name = os.path.basename(video_obj.file.name)
        opth = f"motion_output_{petvideo_id}.mp4"
        mark_right_side_pose(video_path, conf=0.2,video_obj=video_obj, output_path=opth)
        final_output_path = f"temp_media_store/processed_{original_name}"
        video_obj.distance = 0.0
        video_obj.is_video_processed = True
        video_obj.progress = 100
        _encode_to_h264(opth, final_output_path)

        _save_processed_file(video_obj, final_output_path, original_name)
        _remove_files(final_output_path)
        _cleanup_local_video(video_obj)
        video_obj.save()
        _remove_files(opth)
        test_video_url(assessment_id, test_id, participant_id=video_obj.participant_id,
                       vurl=video_obj.processed_file.url)
        logger.info(f"[process_video_task] Done processing PetVideo ID {petvideo_id}")

    except Exception as e:
        logger.error(f"[process_video_task] Error processing PetVideo ID {petvideo_id}: {e}", exc_info=True)


def process_ttest_6x15_dash(petvideo_id, test_id, assessment_id):
    logger.info(f"[process_video_task] STARTED TEST (runs) / ASSESSMENT ID: {petvideo_id}")
    if test_id == "" or assessment_id == "":
        logger.info(f"[process_video_task] INVALID TEST/ ASSESSMENT ID: {petvideo_id}")
        return
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
    video_path = _ensure_local_video(video_obj)

    #with open(video_path, 'rb') as f:
    #    video_obj.processed_file.save(os.path.basename(video_obj.file.name), File(f), save=True)
    output_dir = os.path.join(settings.TEMP_STORAGE, 'post_processed_video')
    os.makedirs(output_dir, exist_ok=True)
    try:
        video_obj.is_video_processed = False
        video_obj.progress = 0
        homograph_obj = CalibrationDataModel.objects.filter(
            assessment_id=assessment_id,
            test_id=test_id
        ).first()
        use_homograph = False
        if not homograph_obj:
            logger.info(f"[process_video_task] Not calibrated {petvideo_id}")
            return

        #fno, duration, _ = detect_crossing_rightmost_ankle(video_path, homograph_obj.end_pixel, reference=homograph_obj.unit_distance,show=False)

        duration =  - 3.5
        original_name = os.path.basename(video_obj.file.name)

        final_output_path = f"temp_media_store/processed_{original_name}"
        opth = f"motion_output_{petvideo_id}.mp4"
        is_changed = False
        if duration < 0.5:
            logger.info(f"[process_video_task] Detection started {petvideo_id}")
            fpx = [homograph_obj.origin_x, homograph_obj.origin_y]
            if homograph_obj.origin_x == 0 and homograph_obj == 0:
                fpx = [homograph_obj.end_pixel, None]
            fno, duration, _ = detect_crossing_person_box_reverse_nobuffer(video_path, fpx, show=False, video_obj=video_obj)
            print(fno, "Stop frame")
            if duration and duration > 1:
                #video_obj.duration = duration - 3.5
                is_changed = video_obj.update_metrix(duration - 3.5, None)
                write_video_until_frame(video_path, duration=-3.5 + duration, end_frame_idx=fno, x_B=homograph_obj.end_pixel, reference=homograph_obj.unit_distance, output_path=opth)
                pass
            else:
                with open(video_path, 'rb') as f:
                    video_obj.is_video_processed = True
                    video_obj.progress = 100
                    video_obj.processed_file.save(os.path.basename(video_obj.file.name), File(f), save=True)
                _cleanup_local_video(video_obj)
                logger.info(f"[process_video_task] Detection failed {petvideo_id}")
                return
        video_obj.distance = homograph_obj.unit_distance
        video_obj.is_video_processed = True
        video_obj.progress = 100
        _encode_to_h264(opth, final_output_path)
        if is_changed:
            _save_processed_file(video_obj, final_output_path, original_name)
        _remove_files(final_output_path, opth)
        _cleanup_local_video(video_obj)
        video_obj.save()
        test_video_url(assessment_id, test_id, participant_id=video_obj.participant_id, vurl=video_obj.processed_file.url)
        logger.info(f"[process_video_task] Done processing PetVideo ID {petvideo_id}")

    except Exception as e:
        logger.error(f"[process_video_task] Error processing PetVideo ID {petvideo_id}: {e}", exc_info=True)

