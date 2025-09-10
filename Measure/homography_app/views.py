import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
import cv2
import numpy as np
from .helper import merge_close_points

from .models import PetVideos, SingletonHomographicMatrixModel

DEFAULT_HSV = np.array([26, 116, 152], dtype=np.uint8)
TOL_H, TOL_S, TOL_V = 10, 50, 50


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

        points = merge_close_points(points)  # Your custom logic

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


def list_videos(request):
    videos = PetVideos.objects.all().order_by('-uploaded_at')
    data = [{'name': v.name, 'file': v.file.url, 'distance': v.distance, 'participant_name': v.participant_name, "pet_type": v.pet_type} for v
            in videos]
    return JsonResponse({'videos': data})


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

