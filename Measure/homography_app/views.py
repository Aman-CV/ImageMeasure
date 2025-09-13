import json
import os

from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
import cv2
import numpy as np
from .helper import merge_close_points, DEFAULT_HSV, TOL_S, TOL_H, TOL_V
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
        obj = PetVideos.objects.create(
            name=video.name,
            file=video,
            participant_name=participant_name,
            pet_type=pet_type
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
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        unit_distance = float(request.POST.get('square_size', 60))  # Default to 60 if not provided
        np_img = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

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

        points = merge_close_points(points, threshold=40)  # Your custom logic

        if len(points) != 4:
            return JsonResponse({
                'status': 'error',
                'message': f'Failed to detect exactly 4 points. Detected: {len(points)}'
            }, status=400)

        # Proceed to compute homography and save
        pts = np.array(points, dtype=np.float32)
        total_x = sum([pt[0] for pt in pts]) / 4.
        total_y = sum([pt[1] for pt in pts]) / 4.
        pts = pts - np.array([total_x, total_y])
        order_points = np.zeros((4, 2))
        for pt in pts:
            if pt[0] <= 0 and pt[1] <= 0:
                order_points[3] = pt
            elif pt[0] > 0 > pt[1]:
                order_points[2] = pt
            elif pt[0] > 0 and pt[1] > 0:
                order_points[1] = pt
            else:
                order_points[0] = pt
        order_points += np.array([total_x, total_y])

        world_pts = np.array([
            [0, 0],
            [unit_distance, 0],
            [unit_distance, unit_distance],
            [0, unit_distance]
        ], dtype=np.float32)

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
                (0, 255, 0),
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

        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return JsonResponse({'error': 'Invalid image'}, status=400)
        hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_frame[y, x]
        h, s, v = int(h), int(s), int(v)
        lower = np.array([max(h - TOL_H, 0), max(s - TOL_S, 0), max(v - TOL_V, 0)])
        upper = np.array([min(h + TOL_H, 179), min(s + TOL_S, 255), min(v + TOL_V, 255)])
        mask = cv2.inRange(hsv_frame, lower, upper)
        highlight = cv2.convertScaleAbs(img, alpha=1.8, beta=40)
        output_img = img.copy()
        output_img[mask > 0] = highlight[mask > 0]

        _, buffer = cv2.imencode('.jpg', output_img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        # ✅ Save HSV value in singleton model
        singleton = SingletonHomographicMatrixModel.load()
        singleton.hsv_value = {'h': int(h), 's': int(s), 'v': int(v)}
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
             'participant_name': v.participant_name, "pet_type": v.pet_type, 'id': v.id, 'is_processed': v.is_video_processed} for v
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

