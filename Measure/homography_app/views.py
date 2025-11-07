import json
import os
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
import cv2
import numpy as np
import time
from sklearn.utils import deprecated
import tempfile
from .helper import merge_close_points, DEFAULT_HSV, TOL_S, TOL_H, TOL_V, order_points_anticlockwise, \
    process_frame_for_color_centers, correct_white_balance
from .models import PetVideos, SingletonHomographicMatrixModel
from .task import process_video_task
from django.conf import settings
import base64

@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']

        # Get extra fields from POST
        participant_name = request.POST.get('participant_name', 'NoName')
        pet_type = request.POST.get('pet_type', 'BT')
        duration = float(request.POST.get('duration', 0))
        to_be_processed = False
        if pet_type == 'BROAD_JUMP' or pet_type == "SIT_THROW" or pet_type == "SIT_REACH":
            to_be_processed = True
        obj = PetVideos.objects.create(
            name=video.name,
            file=video,
            participant_name=participant_name,
            pet_type=pet_type,
            duration=duration,
            progress= 0 if to_be_processed else 100,
            to_be_processed = to_be_processed
        )
        process_video_task(obj.id)
        return JsonResponse({
            'status': 'success',
            'name': obj.name,
            'participant_name': obj.participant_name,
            'pet_type': obj.pet_type
        })

    return JsonResponse({'status': 'error'}, status=400)

@csrf_exempt
def upload_calibration_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        unit_distance = float(request.POST.get('square_size', 0.984252))

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp:
            for chunk in video_file.chunks():
                temp.write(chunk)
            temp.flush()
            os.fsync(temp.fileno())
            temp_path = temp.name
        time.sleep(0.2)
        cap = None
        for _ in range(3):
            cap = cv2.VideoCapture(temp_path)
            if cap.isOpened():
                break
            time.sleep(0.2)

        if not cap or not cap.isOpened():
            os.remove(temp_path)
            return JsonResponse({
                'status': 'error',
                'message': f'Could not open uploaded video (path: {temp_path})'
            }, status=400)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            os.remove(temp_path)
            return JsonResponse({
                'status': 'error',
                'message': 'Video appears empty or unreadable'
            }, status=400)

        middle_frame_index = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        ret, frame = cap.read()
        frame = correct_white_balance(frame)
        cap.release()
        os.remove(temp_path)

        if not ret or frame is None:
            return JsonResponse({
                'status': 'error',
                'message': 'Could not extract frame from video'
            }, status=400)
        frame = cv2.resize(frame, (1280, 720))
        cv2.imwrite("cal.jpg", frame)
        singleton = SingletonHomographicMatrixModel.load()
        if singleton.hsv_value:  # will be {} if not set
            h = singleton.hsv_value.get('h', DEFAULT_HSV[0])
            s = singleton.hsv_value.get('s', DEFAULT_HSV[1])
            v = singleton.hsv_value.get('v', DEFAULT_HSV[2])
        else:
            h, s, v = DEFAULT_HSV
        points = process_frame_for_color_centers(frame, selected_point=[640, 540], target_hsv=(h, s, v))

        points = merge_close_points(points, threshold=10)  # Your custom logic
        points_sorted = sorted(points, key=lambda p: p[1], reverse=True)
        if len(points) < 4:
            return JsonResponse({
                'status': 'error',
                'message': f'Failed to detect exactly 4 points. Detected: {len(points)}'
            }, status=400)
        cv2.imwrite("tihis.jpg", frame)
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

        homography_obj = SingletonHomographicMatrixModel.load()
        json_content = json.dumps(homography_matrix)
        if homography_obj.matrix:
            homography_obj.matrix.delete(save=False)
        homography_obj.matrix.save(
            'homography.json',
            ContentFile(json_content),
            save=False
        )
        homography_obj.unit_distance = unit_distance

        for idx, (x, y) in enumerate(order_points):
            cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(
                frame,
                str(idx + 1),
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (233, 0, 2),
                1
            )
        _, buffer = cv2.imencode('.jpg', frame)
        if homography_obj.file:
            homography_obj.file.delete(save=False)
        homography_obj.file.save(
            'calibrated.jpg',
            ContentFile(buffer.tobytes()),
            save=True
        )

        homography_obj.save()

        return JsonResponse({
            'status': 'success',
        })

    return JsonResponse({
        'status': 'error',
        'message': 'No image uploaded'
    }, status=400)

@csrf_exempt
def upload_calibration_video_deprecated(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        unit_distance = float(request.POST.get('square_size', 0.984252))

        file_bytes = np.asarray(bytearray(video_file.read()), dtype=np.uint8)
        cap = cv2.VideoCapture(cv2.CAP_FFMPEG)
        cap.open(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))

        if not cap.isOpened():
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4') as temp:
                temp.write(file_bytes)
                temp.flush()
                cap = cv2.VideoCapture(temp.name)
                if not cap.isOpened():
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Failed to read uploaded video'
                    }, status=400)

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                middle_frame_index = total_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
                ret, frame = cap.read()
                cap.release()

        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame_index = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
            ret, frame = cap.read()
            cap.release()

        if not ret or frame is None:
            return JsonResponse({
                'status': 'error',
                'message': 'Could not extract frame from video'
            }, status=400)
        frame = cv2.resize(frame, (1280, 720))
        cv2.imwrite("cal.jpg", frame)
        # HSV mask logic (assume DEFAULT_HSV, TOL_H, TOL_S, TOL_V are defined)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        singleton = SingletonHomographicMatrixModel.load()
        # Load HSV from model if set, else use DEFAULT_HSV
        if singleton.hsv_value:  # will be {} if not set
            h = singleton.hsv_value.get('h', DEFAULT_HSV[0])
            s = singleton.hsv_value.get('s', DEFAULT_HSV[1])
            v = singleton.hsv_value.get('v', DEFAULT_HSV[2])
        else:
            h, s, v = DEFAULT_HSV
        lower = np.array([max(h - TOL_H, 0), max(s - TOL_S, 0), max(v - TOL_V, 0)])
        upper = np.array([min(h + TOL_H, 179), min(s + TOL_S, 255), min(v + TOL_V, 255)])
        mask = cv2.inRange(hsv_frame, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                points.append((cx, cy))

        points = merge_close_points(points, threshold=10)  # Your custom logic
        points_sorted = sorted(points, key=lambda p: p[1], reverse=True)
        if len(points) < 4:
            return JsonResponse({
                'status': 'error',
                'message': f'Failed to detect exactly 4 points. Detected: {len(points)}'
            }, status=400)
        cv2.imwrite("tihis.jpg", frame)
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

        homography_obj = SingletonHomographicMatrixModel.load()
        json_content = json.dumps(homography_matrix)
        if homography_obj.matrix:
            homography_obj.matrix.delete(save=False)
        homography_obj.matrix.save(
            'homography.json',
            ContentFile(json_content),
            save=False
        )
        homography_obj.unit_distance = unit_distance

        # Mark detected points on frame
        for idx, (x, y) in enumerate(order_points):
            cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(
                frame,
                str(idx + 1),
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (233, 0, 2),
                1
            )
        _, buffer = cv2.imencode('.jpg', frame)
        if homography_obj.file:
            homography_obj.file.delete(save=False)
        homography_obj.file.save(
            'calibrated.jpg',
            ContentFile(buffer.tobytes()),
            save=True
        )

        homography_obj.save()

        return JsonResponse({
            'status': 'success',
        })

    return JsonResponse({
        'status': 'error',
        'message': 'No image uploaded'
    }, status=400)


@csrf_exempt
def process_image(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    try:
        img_file = request.FILES['image']
        x = int(request.POST['x'])
        y = int(request.POST['y'])
        print(x, y)
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return JsonResponse({'error': 'Invalid image'}, status=400)
        print(img.shape)
        cv2.imwrite("ths.jpg", img)
        hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_frame[y, x]
        h, s, v = int(h), int(s), int(v)
        lower = np.array([max(h - TOL_H, 0), max(s - TOL_S, 0), max(v - TOL_V, 0)])
        upper = np.array([min(h + TOL_H, 179), min(s + TOL_S, 255), min(v + TOL_V, 255)])
        mask = cv2.inRange(hsv_frame, lower, upper)
        highlight = cv2.convertScaleAbs(img, alpha=1.8, beta=40)
        output_img = img.copy()
        output_img[mask > 0] = highlight[mask > 0]
        cv2.circle(output_img, (x, y), 4, (25, 48, 228), -1)
        cv2.imwrite("this.jpg", output_img)
        _, buffer = cv2.imencode('.jpg', output_img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        # ✅ Save HSV value in singleton model
        singleton = SingletonHomographicMatrixModel.load()
        singleton.hsv_value = {'h': int(h), 's': int(s), 'v': int(v)}
        singleton.tracker_hsv_value = {'h': int(h), 's': int(s), 'v': int(v)}
        singleton.save()

        return JsonResponse({
            'hsv': {'h': int(h), 's': int(s), 'v': int(v)},
            'image_base64': f"data:image/jpeg;base64,{encoded_image}"
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)



def list_videos(request):
    videos = PetVideos.objects.all().order_by('-uploaded_at')
    data = [{'name': v.name, 'file': v.file.url, 'distance': v.distance,
             'participant_name': v.participant_name, "pet_type": v.pet_type, 'id': v.id, 'is_processed': v.is_video_processed, "progress": v.progress, 'duration': v.duration, 'to_be_processed': v.to_be_processed} for v
            in videos]
    return JsonResponse({'videos': data})


def video_stream(request, filename):
    file_path = os.path.join(settings.MEDIA_ROOT, 'post_processed_video', filename)
    if not os.path.exists(file_path):
        return JsonResponse({"status": "error", "message": "Video not found"}, status=404)

    return FileResponse(open(file_path, 'rb'), content_type='video/mp4')


def get_video_detail(request):
    video_id = request.GET.get('id')
    if not video_id:
        return JsonResponse({'status': 'error', 'message': 'No video ID provided'}, status=400)

    try:
        video = PetVideos.objects.get(id=video_id)
        if not video.processed_file:
            return JsonResponse({'status': 'error', 'message': 'Processed video not available'}, status=404)

        filename = os.path.basename(video.processed_file.name)
        video_url = request.build_absolute_uri(f'/stream_video/{filename}/')

        return JsonResponse({
            'status': 'success',
            'processed_video_url': video_url
        })
    except PetVideos.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Video not found'}, status=404)


def get_homograph(request):
    homograph_obj = SingletonHomographicMatrixModel.load()
    response_data = {
        'square_size': homograph_obj.unit_distance,
        'matrix_url': homograph_obj.matrix.url if homograph_obj.matrix else "",
        'image_url': homograph_obj.file.url if homograph_obj.file else "",
    }

    return JsonResponse({
        'status' : 'success',
        'calibration_info': response_data
    })

