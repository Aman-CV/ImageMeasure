import json
import os
import subprocess
from botocore.exceptions import ClientError
from django.http import JsonResponse, FileResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
import cv2
import numpy as np
import time
from django.core.files import File
from sklearn.utils import deprecated
import tempfile
from .helper import merge_close_points, DEFAULT_HSV, TOL_S, TOL_H, TOL_V, order_points_anticlockwise, \
    process_frame_for_color_centers, correct_white_balance, stretch_contrast, find_yellow_point_lab
from .models import PetVideos, SingletonHomographicMatrixModel, CalibrationDataModel
from .sit_and_reach_helper_ import detect_carpet_segment_p
from .celery_tasks import (
    celery_process_sit_and_throw,
    celery_process_sit_and_reach,
    celery_process_video_task,
    celery_process_15m_dash,
    celery_process_plank,
)
from django.conf import settings
import base64

DEFAULT_HOMOGRAPH_POINTS = {
                "p1": {"fx": None, "fy": None},
                "p2": {"fx": None, "fy": None},
                "p3": {"fx": None, "fy": None},
                "p4": {"fx": None, "fy": None},
            }

FALL_BACK = {
    'flexibility' : 20,
    'lower body strength' : 1.4224,
    'default' : 1,
    'upper body strength' : 1.4224
}


def landing_page(request):
    return render(request, 'landing.html')


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).lower() in ('true', '1', 'yes', 'on')


def _video_defaults(video, participant_name, pet_type, duration_ms, to_be_processed, take_best=False):
    return {
        'name': video.name,
        'file': video,
        'participant_name': participant_name,
        'pet_type': pet_type,
        'duration': round(duration_ms / 1000, 3),
        'progress': 0 if to_be_processed else 100,
        'to_be_processed': to_be_processed,
        'is_video_processed': False if to_be_processed else True,
        'take_best': take_best,
    }


def _create_1fps_video_file(uploaded_video):
    """Create a 1fps mp4 from uploaded stream and return a django File plus temp paths."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
        for chunk in uploaded_video.chunks():
            temp_input.write(chunk)
        temp_input_path = temp_input.name

    temp_output_path = temp_input_path.replace('.mp4', '_1fps.mp4')
    command = [
        'ffmpeg',
        '-y',
        '-i', temp_input_path,
        '-vf', 'fps=1,setpts=N/30/TB',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-an',
        temp_output_path,
    ]
    subprocess.run(command, check=True, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_input_path, temp_output_path


def _cache_video_for_worker(video_obj):
    """Store an h264 worker-local copy at TEMP_VIDEO_STORAGE/videot_<id>.mp4."""
    os.makedirs(settings.TEMP_VIDEO_STORAGE, exist_ok=True)

    ext = os.path.splitext(video_obj.file.name)[1].lower() or '.mp4'
    raw_path = os.path.join(settings.TEMP_VIDEO_STORAGE, f"videot_{video_obj.id}_raw{ext}")
    final_path = os.path.join(settings.TEMP_VIDEO_STORAGE, f"videot_{video_obj.id}.mp4")

    with video_obj.file.open('rb') as src, open(raw_path, 'wb') as destination:
        for chunk in src.chunks():
            destination.write(chunk)
        destination.flush()
        os.fsync(destination.fileno())

    print(f"Debug: saving raw video -> {raw_path}")
    print("Raw video size:", os.path.getsize(raw_path))

    try:
        subprocess.run([
            'ffmpeg', '-i', raw_path,
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-c:a', 'aac', '-movflags', '+faststart', '-y',
            final_path
        ], check=True, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print('FFmpeg failed. Keeping raw file for debugging.')
        raise e

    if os.path.exists(raw_path):
        os.remove(raw_path)
        print('Deleted raw upload:', raw_path)


def _dispatch_processing_task(test_type, video_id, test_id, assessment_id, enable_color_marker_tracking, enable_start_end_detector):
    if test_type == 'upper body strength' or test_type == 'throw':
        celery_process_sit_and_throw.delay(video_id, test_id=test_id, assessment_id=assessment_id)
    elif test_type == 'flexibility' or test_type == 'reach':
        celery_process_sit_and_reach.delay(video_id, test_id=test_id, assessment_id=assessment_id)
    elif test_type in ('endurance', 'sprint speed', 'agility'):
        celery_process_15m_dash.delay(video_id, test_id=test_id, assessment_id=assessment_id)
    elif test_type == 'core strength':
        celery_process_plank.delay(video_id, test_id=test_id, assessment_id=assessment_id)
    else:
        celery_process_video_task.delay(
            video_id,
            enable_color_marker_tracking=enable_color_marker_tracking,
            enable_start_end_detector=enable_start_end_detector,
            test_id=test_id,
            assessment_id=assessment_id,
        )


@csrf_exempt
def upload_video(request):
    """
    Uploads the video to db,
    if video is of test which have distance as measurable quantity,
    program sends it to processing for distance calculations
    """
    if request.method != 'POST' or not request.FILES.get('video'):
        return JsonResponse({'status': 'error'}, status=400)

    video = request.FILES['video']
    participant_name = request.POST.get('participant_name', 'NoName')
    pet_type = request.POST.get('pet_type', 'BT')
    duration_ms = float(request.POST.get('duration', 0))
    to_be_processed = _as_bool(request.POST.get('to_be_processed', 'true'), default=True)
    test_id = request.POST.get('test_id', 'jump')
    participant_id = request.POST.get('participant_id', 'Dummy')
    assessment_id = request.POST.get('assessment_id', 'Dummy')
    enable_start_end_detector = _as_bool(request.POST.get('enable_start_end_detector', 'true'), default=True)
    enable_color_marker_tracking = _as_bool(request.POST.get('enable_color_marker_tracking', 'true'), default=True)
    take_best = _as_bool(request.POST.get('take_best', 'false'), default=False)
    type_param = request.POST.get('type_param', "core strength")
    if type_param is not None:
        type_param = type_param.lower()
    os.makedirs(settings.TEMP_VIDEO_STORAGE, exist_ok=True)

    if type_param == 'core strength':
        temp_input_path = None
        temp_output_path = None
        try:
            temp_input_path, temp_output_path = _create_1fps_video_file(video)
            with open(temp_output_path, 'rb') as f:
                obj, created = PetVideos.objects.update_or_create(
                    participant_id=participant_id,
                    test_id=test_id,
                    assessment_id=assessment_id,
                    defaults=_video_defaults(
                        File(f, name=video.name),
                        participant_name,
                        pet_type,
                        duration_ms,
                        to_be_processed,
                        take_best=take_best,
                        type_param=type_param
                    )
                )
        finally:
            if temp_input_path and os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if temp_output_path and os.path.exists(temp_output_path):
                os.remove(temp_output_path)
    else:
        obj, created = PetVideos.objects.update_or_create(
            participant_id=participant_id,
            test_id=test_id,
            assessment_id=assessment_id,
            defaults=_video_defaults(
                video,
                participant_name,
                pet_type,
                duration_ms,
                to_be_processed,
                take_best=take_best,
                type_param = type_param
            )
        )
        _cache_video_for_worker(obj)

    _dispatch_processing_task(
        obj.type_param,
        obj.id,
        test_id,
        assessment_id,
        enable_color_marker_tracking,
        enable_start_end_detector,
    )

    return JsonResponse({
        'status': 'success',
        'name': obj.name,
        'participant_name': obj.participant_name,
        'pet_type': obj.pet_type,
        'updated': not created
    })

def _parse_homograph_points(raw_points):
    if not raw_points:
        return DEFAULT_HOMOGRAPH_POINTS
    return json.loads(raw_points)


def _extract_middle_frame(video_file, test_id):
    cap = None
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp:
            for chunk in video_file.chunks():
                temp.write(chunk)
            temp.flush()
            os.fsync(temp.fileno())
            temp_path = temp.name

        time.sleep(0.2)
        for _ in range(3):
            cap = cv2.VideoCapture(temp_path)
            if cap.isOpened():
                break
            time.sleep(0.2)

        if not cap or not cap.isOpened():
            raise ValueError(f'Could not open uploaded video (path: {temp_path})')

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError('Video appears empty or unreadable')

        middle_frame_index = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        ret, frame = cap.read()
        # if test_id == 'vPbXoPK4' and frame is not None:
        #     frame = correct_white_balance(frame)

        if not ret or frame is None:
            raise ValueError('Could not extract frame from video')
        cv2.imwrite("media/calib_frame.jpg", frame)
        return cv2.resize(frame, (1280, 720))
    finally:
        if cap:
            cap.release()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _save_preview_to_singleton(singleton, frame, file_name='frame.jpg'):
    return
    # _, buffer = cv2.imencode('.jpg', frame)

    # if singleton.file:
    #     singleton.file.delete(save=False)
    # singleton.file.save(file_name, ContentFile(buffer.tobytes()), save=False)

    # if singleton.mask:
    #     singleton.mask.delete(save=False)
    # singleton.mask.save('mask.jpg', ContentFile(buffer.tobytes()), save=False)

    # singleton.save()


def _run_simple_calibration(frame, payload):
    # singleton = SingletonHomographicMatrixModel.load()
    h, w = frame.shape[:2]

    start_candidate = int(w * payload['position_factor2'])
    end_candidate = int(w * payload['position_factor'])
    start_pixel = min(start_candidate, end_candidate)
    end_pixel = max(start_candidate, end_candidate)

    # singleton.unit_distance = payload['unit_distance']
    # singleton.start_pixel = start_pixel
    # singleton.end_pixel = end_pixel

    homograph_points = payload['homograph_points']
    use_homograph = payload['use_homograph']

    if use_homograph and homograph_points:
        basic_bgr = np.uint8([
            [0, 0, 255],
            [0, 69, 255],
            [0, 255, 255],
            [0, 200, 0],
            [255, 0, 0],
            [130, 0, 75],
            [238, 130, 238],
        ])
        basic_lab = cv2.cvtColor(basic_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        for key, point in homograph_points.items():
            if point.get('fx') is None or point.get('fy') is None:
                continue

            x0 = int(float(point['fx']) * w)
            y0 = int(float(point['fy']) * h)

            x, y = find_yellow_point_lab(
                frame,
                x0,
                y0,
                basic_lab,
                yellow_idx=2,
                clahe=clahe,
                window_size=80
            )
            point['fx'] = float(x / w)
            point['fy'] = float(y / h)

            half_window = 40
            roi_x1 = max(0, x0 - half_window)
            roi_y1 = max(0, y0 - half_window)
            roi_x2 = min(w, x0 + half_window)
            roi_y2 = min(h, y0 + half_window)
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 1)
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(frame, key, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.line(frame, (int(payload['origin_x'] * w), 0), (int(payload['origin_x'] * w), h), (0, 255, 0), 2)

    # _save_preview_to_singleton(singleton, frame, file_name='frame.jpg')

    CalibrationDataModel.objects.update_or_create(
        test_id=payload['test_id'],
        assessment_id=payload['assessment_id'],
        defaults={
            'start_pixel': start_pixel,
            'end_pixel': end_pixel,
            'unit_distance': payload['unit_distance'],
            'use_homograph': use_homograph,
            'homography_points': homograph_points,
            'origin_x': int(payload['origin_x'] * w),
            'origin_y': int(payload['origin_y'] * h)
        }
    )
    return JsonResponse({'status': 'success', 'hpoints': homograph_points if homograph_points else {}})


def _run_homography_calibration(frame, payload):
    # singleton = SingletonHomographicMatrixModel.load()
    homograph_points = payload['homograph_points']
    use_homograph = payload['use_homograph']
    h, w = frame.shape[:2]

    # if singleton.hsv_value:
    #     h_val = singleton.hsv_value.get('h', DEFAULT_HSV[0])
    #     s_val = singleton.hsv_value.get('s', DEFAULT_HSV[1])
    #     v_val = singleton.hsv_value.get('v', DEFAULT_HSV[2])
    # else:
    h_val, s_val, v_val = DEFAULT_HSV

    p1 = [int(homograph_points['p1']['fx'] * w), int(homograph_points['p1']['fy'] * h)]
    selected_point = p1
    points, pt2 = process_frame_for_color_centers(frame, selected_point=selected_point)
    points = merge_close_points(points, threshold=25)
    points_sorted = sorted(points, key=lambda p: p[1], reverse=True)
    end_point_of_mat = 'Normal Calibration Successful'

    unit_distance = payload['unit_distance']
    if len(points) != 4:
        if len(pt2) == 4:
            points = pt2
            end_point_of_mat = 'Markers not detected using end points of mat'
            points_sorted = sorted(points, key=lambda p: p[1], reverse=True)
            unit_distance = FALL_BACK.get(payload['type_param'], 1)
        else:
            return JsonResponse({
                'status': 'error',
                'message': f'Failed to detect exactly 4 points. Detected: {len(points)}'
            }, status=400)

    if len(points) < 6:
        points = points_sorted[:4]
        world_pts = np.array([
            [0, 0],
            [unit_distance, 0],
            [unit_distance, unit_distance],
            [0, unit_distance]
        ], dtype=np.float32)
    else:
        points = points_sorted[:6]
        world_pts = np.array([
            [0, 0],
            [unit_distance, 0],
            [unit_distance + unit_distance, 0],
            [unit_distance + unit_distance, unit_distance],
            [unit_distance, unit_distance],
            [0, unit_distance]
        ], dtype=np.float32)

    order_points = np.array(order_points_anticlockwise(points))
    H, _ = cv2.findHomography(order_points, world_pts)
    homography_matrix = H.tolist()

    # homography_obj = SingletonHomographicMatrixModel.load()
    # json_content = json.dumps(homography_matrix)
    # if homography_obj.matrix:
    #     homography_obj.matrix.delete(save=False)
    # homography_obj.matrix.save('homography.json', ContentFile(json_content), save=False)
    # homography_obj.unit_distance = unit_distance
    # homography_obj.start_pixel_broad_jump = order_points[0][0]

    for idx, (x, y) in enumerate(order_points):
        cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(frame, str(idx + 1), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (233, 0, 2), 1)

    # _save_preview_to_singleton(homography_obj, frame, file_name='mask.jpg')

    for i, (key, point) in enumerate(homograph_points.items()):
        if i >= len(order_points):
            break
        point['fx'] = float(order_points[i][0] / float(w))
        point['fy'] = float(order_points[i][1] / float(h))

    CalibrationDataModel.objects.update_or_create(
        test_id=payload['test_id'],
        assessment_id=payload['assessment_id'],
        defaults={
            'start_pixel': 0,
            'end_pixel': 1,
            'unit_distance': unit_distance,
            'use_homograph': use_homograph,
            'homography_points': homograph_points,
            'origin_x': int(payload['origin_x'] * w),
            'origin_y': int(payload['origin_y'] * h)
        }
    )

    return JsonResponse({
        'status': 'success',
        'hpoints': homograph_points if homograph_points else {},
        'message': end_point_of_mat
    })

@csrf_exempt
def upload_calibration_video(request):
    """
    recieves a video , extract on frame from video and marks calibration points
    """
    if request.method != 'POST' or not request.FILES.get('video'):
        return JsonResponse({'status': 'error', 'message': 'No image uploaded'}, status=400)

    payload = dict(
        video_file=request.FILES['video'],
        test_id=request.POST.get('test_id', 'not_sit_and_reach'),
        unit_distance=float(request.POST.get('square_size', 2.5908)),
        position_factor=float(request.POST.get('position_factor', 0.5)),
        position_factor2=float(request.POST.get('position_factor2', 0.15)),
        assessment_id=request.POST.get('assessment_id', 'notvalid'),
        use_homograph=_as_bool(request.POST.get('use_homograph', 'false'), default=False),
        use_sam_homograph=_as_bool(request.POST.get('use_sam_homograph', 'false'), default=False),
        origin_x=float(request.POST.get('origin_x', 0)),
        origin_y=float(request.POST.get('origin_y', 0)),
        type_param=request.POST.get('type_param', None)
    )
    if payload['type_param'] is not None:
        payload['type_param'] = payload['type_param'].lower()
    payload['homograph_points'] = _parse_homograph_points(request.POST.get('hpoints', None))

    try:
        frame = _extract_middle_frame(payload['video_file'], payload['test_id'])
    except ValueError as exc:
        return JsonResponse({'status': 'error', 'message': str(exc)}, status=400)

    #simple_test_ids = {'Vnb7E6L6', 'VpKl80KM', 'BwbJyXKl', 'G6bWk0bW', 'vPbXoPK4', 'lzb1PEKm'}
    simple_test_type = {"upper body strength", "lower body strength", "sprint speed", "agility", "flexibility", "endurance"}
    if not payload['use_sam_homograph'] and payload['type_param'] in simple_test_type:
        return _run_simple_calibration(frame, payload)

    return _run_homography_calibration(frame, payload)

# @csrf_exempt
# def upload_calibration_video_deprecated(request):
#     if request.method == 'POST' and request.FILES.get('video'):
#         video_file = request.FILES['video']
#         unit_distance = float(request.POST.get('square_size', 0.984252))

#         file_bytes = np.asarray(bytearray(video_file.read()), dtype=np.uint8)
#         cap = cv2.VideoCapture(cv2.CAP_FFMPEG)
#         cap.open(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))

#         if not cap.isOpened():
#             import tempfile
#             with tempfile.NamedTemporaryFile(suffix='.mp4') as temp:
#                 temp.write(file_bytes)
#                 temp.flush()
#                 cap = cv2.VideoCapture(temp.name)
#                 if not cap.isOpened():
#                     return JsonResponse({
#                         'status': 'error',
#                         'message': 'Failed to read uploaded video'
#                     }, status=400)

#                 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#                 middle_frame_index = total_frames // 2
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
#                 ret, frame = cap.read()
#                 cap.release()

#         else:
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             middle_frame_index = total_frames // 2
#             cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
#             ret, frame = cap.read()
#             cap.release()

#         if not ret or frame is None:
#             return JsonResponse({
#                 'status': 'error',
#                 'message': 'Could not extract frame from video'
#             }, status=400)
#         frame = cv2.resize(frame, (1280, 720))
#         cv2.imwrite("media/cal.jpg", frame)
#         # HSV mask logic (assume DEFAULT_HSV, TOL_H, TOL_S, TOL_V are defined)
#         hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         singleton = SingletonHomographicMatrixModel.load()
#         # Load HSV from model if set, else use DEFAULT_HSV
#         if singleton.hsv_value:  # will be {} if not set
#             h = singleton.hsv_value.get('h', DEFAULT_HSV[0])
#             s = singleton.hsv_value.get('s', DEFAULT_HSV[1])
#             v = singleton.hsv_value.get('v', DEFAULT_HSV[2])
#         else:
#             h, s, v = DEFAULT_HSV
#         lower = np.array([max(h - TOL_H, 0), max(s - TOL_S, 0), max(v - TOL_V, 0)])
#         upper = np.array([min(h + TOL_H, 179), min(s + TOL_S, 255), min(v + TOL_V, 255)])
#         mask = cv2.inRange(hsv_frame, lower, upper)

#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         points = []
#         for cnt in contours:
#             M = cv2.moments(cnt)
#             if M["m00"] > 0:
#                 cx = M["m10"] / M["m00"]
#                 cy = M["m01"] / M["m00"]
#                 points.append((cx, cy))

#         points = merge_close_points(points, threshold=10)  # Your custom logic
#         points_sorted = sorted(points, key=lambda p: p[1], reverse=True)
#         if len(points) < 4:
#             return JsonResponse({
#                 'status': 'error',
#                 'message': f'Failed to detect exactly 4 points. Detected: {len(points)}'
#             }, status=400)
#         cv2.imwrite("media/tihis.jpg", frame)
#         if len(points) < 6:
#             points = points_sorted[:4]
#             world_pts = np.array([
#                 [0, 0],
#                 [unit_distance, 0],
#                 [unit_distance, unit_distance],
#                 [0, unit_distance]
#             ], dtype=np.float32)
#         else:
#             points = points_sorted[:6]
#             world_pts = np.array([
#                 [0, 0],
#                 [unit_distance, 0],
#                 [unit_distance + unit_distance, 0],
#                 [unit_distance + unit_distance, unit_distance],
#                 [unit_distance, unit_distance],
#                 [0, unit_distance]
#             ], dtype=np.float32)

#         order_points = np.array(order_points_anticlockwise(points))

#         H, _ = cv2.findHomography(order_points, world_pts)
#         homography_matrix = H.tolist()

#         homography_obj = SingletonHomographicMatrixModel.load()
#         json_content = json.dumps(homography_matrix)
#         if homography_obj.matrix:
#             homography_obj.matrix.delete(save=False)
#         homography_obj.matrix.save(
#             'homography.json',
#             ContentFile(json_content),
#             save=False
#         )
#         homography_obj.unit_distance = unit_distance

#         # Mark detected points on frame
#         for idx, (x, y) in enumerate(order_points):
#             cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)
#             cv2.putText(
#                 frame,
#                 str(idx + 1),
#                 (int(x) + 5, int(y) - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (233, 0, 2),
#                 1
#             )
#         _, buffer = cv2.imencode('.jpg', frame)
#         if homography_obj.file:
#             homography_obj.file.delete(save=False)
#         homography_obj.file.save(
#             'mask.jpg',
#             ContentFile(buffer.tobytes()),
#             save=True
#         )

#         homography_obj.save()

#         return JsonResponse({
#             'status': 'success',
#         })

#     return JsonResponse({
#         'status': 'error',
#         'message': 'No image uploaded'
#     }, status=400)


# @csrf_exempt
# def process_image(request):
#     """
#     DO NOT USE THIS ENDPOINT. This was an experimental endpoint to test HSV based color detection for calibration and is now deprecated. It may be removed in future releases.
#     """
#     if request.method != 'POST':
#         return JsonResponse({'error': 'Only POST allowed'}, status=405)

#     try:
#         img_file = request.FILES['image']
#         x = int(request.POST['x'])
#         y = int(request.POST['y'])
#         print(x, y)
#         file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#         if img is None:
#             return JsonResponse({'error': 'Invalid image'}, status=400)
#         print(img.shape)
#         cv2.imwrite("media/ths.jpg", img)
#         hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         h, s, v = hsv_frame[y, x]
#         h, s, v = int(h), int(s), int(v)
#         lower = np.array([max(h - TOL_H, 0), max(s - TOL_S, 0), max(v - TOL_V, 0)])
#         upper = np.array([min(h + TOL_H, 179), min(s + TOL_S, 255), min(v + TOL_V, 255)])
#         mask = cv2.inRange(hsv_frame, lower, upper)
#         highlight = cv2.convertScaleAbs(img, alpha=1.8, beta=40)
#         output_img = img.copy()
#         output_img[mask > 0] = highlight[mask > 0]
#         cv2.circle(output_img, (x, y), 4, (25, 48, 228), -1)
#         cv2.imwrite("media/this.jpg", output_img)
#         _, buffer = cv2.imencode('.jpg', output_img)
#         encoded_image = base64.b64encode(buffer).decode('utf-8')

#         singleton = SingletonHomographicMatrixModel.load()
#         singleton.hsv_value = {'h': int(h), 's': int(s), 'v': int(v)}
#         singleton.tracker_hsv_value = {'h': int(h), 's': int(s), 'v': int(v)}
#         singleton.save()

#         return JsonResponse({
#             'hsv': {'h': int(h), 's': int(s), 'v': int(v)},
#             'image_base64': f"data:image/jpeg;base64,{encoded_image}"
#         })

#     except Exception as e:
#         return JsonResponse({'error': str(e)}, status=500)


def list_videos_by_assessment_and_test(request):
    assessment_id = request.GET.get('assessment_id')  or 'dummy'
    test_id = request.GET.get('test_id') or 'jump'

    videos = PetVideos.objects.filter(
        assessment_id=assessment_id,
        test_id=test_id
    ).order_by('-uploaded_at')
    data = [{
        'name': v.name,
        'file': v.file.url if v.file else None,
        'distance': v.distance,
        'participant_name': v.participant_name,
        'pet_type': v.pet_type,
        'id': v.id,
        'is_processed': v.is_video_processed,
        'progress': v.progress,
        'duration': v.duration,
        'to_be_processed': v.to_be_processed,
        'participant_id': v.participant_id
    } for v in videos]

    return JsonResponse({'status':'true', 'message': 'success', 'data': data})



def list_videos(request):
    videos = PetVideos.objects.all().order_by('-uploaded_at')
    data = [{'name': v.name, 'file': v.file.url, 'distance': v.distance,
             'participant_name': v.participant_name, "pet_type": v.pet_type, 'id': v.id, 'is_processed': v.is_video_processed, "progress": v.progress, 'duration': v.duration, 'to_be_processed': v.to_be_processed} for v
            in videos]
    return JsonResponse({'videos': data})


def video_stream(request, video_id):
    try:
        video = get_object_or_404(PetVideos, id=video_id)

        if not video.processed_file:
            return JsonResponse(
                {"status": "error", "message": "Processed video not available"},
                status=404
            )

        file_obj = video.processed_file.open('rb')

        return FileResponse(
            file_obj,
            content_type='video/mp4'
        )

    except PetVideos.DoesNotExist:
        return JsonResponse(
            {"status": "error", "message": "Video not found"},
            status=404
        )

    except ClientError as e:
        return JsonResponse(
            {
                "status": "error",
                "message": "Failed to access video storage",
                "detail": str(e)
            },
            status=500
        )

    except Exception as e:
        return JsonResponse(
            {
                "status": "error",
                "message": "Unexpected error while streaming video"
            },
            status=500
        )

# def video_stream(request, filename):
#     file_path = os.path.join(settings.TEMP_STORAGE, 'post_processed_video', filename)
#     if not os.path.exists(file_path):
#         return JsonResponse({"status": "error", "message": "Video not found"}, status=404)
#
#     return FileResponse(open(file_path, 'rb'), content_type='video/mp4')

def get_video_detail(request):
    video_id = request.GET.get('id')
    if not video_id:
        return JsonResponse(
            {'status': 'error', 'message': 'No video ID provided'},
            status=400
        )

    try:
        video = PetVideos.objects.get(id=video_id)

        if not video.processed_file:
            return JsonResponse(
                {'status': 'error', 'message': 'Processed video not available'},
                status=404
            )

        video_url = request.build_absolute_uri(
            f'/stream_video/{video.id}/'
        )

        return JsonResponse({
            'status': 'success',
            'processed_video_url': video_url
        })

    except PetVideos.DoesNotExist:
        return JsonResponse(
            {'status': 'error', 'message': 'Video not found'},
            status=404
        )

# def get_video_detail(request):
#     video_id = request.GET.get('id')
#     if not video_id:
#         return JsonResponse({'status': 'error', 'message': 'No video ID provided'}, status=400)
#
#     try:
#         video = PetVideos.objects.get(id=video_id)
#         if not video.processed_file:
#             return JsonResponse({'status': 'error', 'message': 'Processed video not available'}, status=404)
#
#         filename = os.path.basename(video.processed_file.name)
#         video_url = request.build_absolute_uri(f'/stream_video/{filename}/')
#
#         return JsonResponse({
#             'status': 'success',
#             'processed_video_url': video_url
#         })
#     except PetVideos.DoesNotExist:
#         return JsonResponse({'status': 'error', 'message': 'Video not found'}, status=404)


# def get_homograph(request):
#     test_id = request.GET.get('test_id', "not_sit_and_reach")
#     homograph_obj = SingletonHomographicMatrixModel.load()
#     if test_id != "vPbXoPK4" and test_id != "BwbJyXKl":
#         response_data = {
#             'square_size': homograph_obj.unit_distance,
#             'matrix_url': homograph_obj.matrix.url if homograph_obj.matrix else "",
#             'image_url': homograph_obj.file.url if homograph_obj.file else "",
#         }
#     else:
#         response_data = {
#             'square_size': homograph_obj.unit_distance,
#             'matrix_url': homograph_obj.matrix.url if homograph_obj.matrix else "",
#             'image_url': homograph_obj.mask.url if homograph_obj.mask else "",
#         }
#     return JsonResponse({
#         'status' : 'success',
#         'calibration_info': response_data
#     })

