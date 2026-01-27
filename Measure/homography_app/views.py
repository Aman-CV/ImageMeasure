import json
import os
import subprocess
from botocore.exceptions import ClientError
from django.http import JsonResponse, FileResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
import cv2
import numpy as np
import time
from sklearn.utils import deprecated
import tempfile
from .helper import merge_close_points, DEFAULT_HSV, TOL_S, TOL_H, TOL_V, order_points_anticlockwise, \
    process_frame_for_color_centers, correct_white_balance, stretch_contrast
from .models import PetVideos, SingletonHomographicMatrixModel, CalibrationDataModel
from .sit_and_reach_helper_ import detect_carpet_segment_p
from .task import process_video_task, process_sit_and_throw, process_sit_and_reach
from django.conf import settings
import base64


@csrf_exempt
def upload_video(request):
    """
    Uploads the video to db,
    if video is of test which have distance as measurable quantity,
    program sends it to processing for distance calculations
    """
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']

        # Get extra fields from POST
        participant_name = request.POST.get('participant_name', 'NoName')
        pet_type = request.POST.get('pet_type', 'BT')
        duration = float(request.POST.get('duration', 0))
        to_be_processed_str = request.POST.get('to_be_processed', 'true').lower()
        to_be_processed = to_be_processed_str in ('true', '1', 'yes', 'on')
        test_id = request.POST.get('test_id', "jump")
        participant_id = request.POST.get('participant_id', 'Dummy')
        assessment_id = request.POST.get('assessment_id', 'Dummy')
        enable_start_end_detector = request.POST.get('enable_start_end_detector', 'true').lower()
        enable_color_marker_tracking = request.POST.get('enable_color_marker_tracking', 'true').lower()
        enable_start_end_detector = enable_start_end_detector in ('true', '1', 'yes', 'on')
        enable_color_marker_tracking = enable_color_marker_tracking in ('true', '1', 'yes', 'on')

        os.makedirs(settings.TEMP_VIDEO_STORAGE, exist_ok=True)



        obj, created = PetVideos.objects.update_or_create(
            participant_id=participant_id,
            test_id=test_id,
            assessment_id=assessment_id,
            defaults={
                'name': video.name,
                'file': video,
                'participant_name': participant_name,
                'pet_type': pet_type,
                'duration': round(duration / 1000, 3),
                'progress': 0 if to_be_processed else 100,
                'to_be_processed': to_be_processed,
            }
        )
        ext = os.path.splitext(video.name)[1].lower()
        raw_path = os.path.join(
            settings.TEMP_VIDEO_STORAGE,
            f"videot_{obj.id}_raw{ext}"
        )

        print(f"Debug: saving raw video → {raw_path}")

        with open(raw_path, "wb") as destination:
            for chunk in video.chunks():
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())

        print("Raw video size:", os.path.getsize(raw_path))
        final_path = os.path.join(
            settings.TEMP_VIDEO_STORAGE,
            f"videot_{obj.id}.mp4"
        )

        try:
            subprocess.run([
                'ffmpeg', '-i', raw_path,
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                '-c:a', 'aac', '-movflags', '+faststart', '-y',
                final_path
            ], check=True)


            if os.path.exists(raw_path):
                os.remove(raw_path)
                print("Deleted raw upload:", raw_path)

        except subprocess.CalledProcessError as e:
            print("FFmpeg failed. Keeping raw file for debugging.")
            raise e

        # ext = os.path.splitext(video.name)[1]
        # file_path = os.path.join(
        #     settings.TEMP_VIDEO_STORAGE,
        #     f"videot_{obj.id}{ext}"
        # )
        # print(f"Debug: videot_{obj.id}{ext}")
        # with open(file_path, 'wb') as destination:
        #     for chunk in video.chunks():
        #         destination.write(chunk)

        if test_id == "BwbJyXKl":
            process_sit_and_throw(obj.id, test_id=test_id, assessment_id =assessment_id)
        elif test_id == "vPbXoPK4":
            print("I was here")
            process_sit_and_reach(obj.id, test_id=test_id, assessment_id=assessment_id)
        else:
            process_video_task(obj.id, enable_color_marker_tracking=enable_color_marker_tracking, enable_start_end_detector=enable_start_end_detector, test_id=test_id, assessment_id=assessment_id)
        return JsonResponse({
            'status': 'success',
            'name': obj.name,
            'participant_name': obj.participant_name,
            'pet_type': obj.pet_type,
            'updated': not created
        })

    return JsonResponse({'status': 'error'}, status=400)

@csrf_exempt
def upload_calibration_video(request):
    """
    recieves a video , extract on frame from video and marks calibration points
    """
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        test_id = request.POST.get('test_id', "not_sit_and_reach")
        unit_distance = float(request.POST.get('square_size', 2.5908))
        position_factor = float(request.POST.get('position_factor', 1.5))
        assessment_id = request.POST.get('assessment_id', 'notvalid')
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
        if test_id == "vPbXoPK4":
            frame = correct_white_balance(frame)
        cap.release()
        os.remove(temp_path)

        if not ret or frame is None:
            return JsonResponse({
                'status': 'error',
                'message': 'Could not extract frame from video'
            }, status=400)
        frame = cv2.resize(frame, (1280, 720))
        cv2.imwrite("media/cal_orig.jpg", frame)
        # if test_id == "vPbXoPK4":
        #     frame = stretch_contrast(frame)
        cv2.imwrite("media/cal.jpg", frame)
        if test_id == "notvalid":
            mask, _, _  = detect_carpet_segment_p(frame)
            singleton = SingletonHomographicMatrixModel.load()
            _, buffer = cv2.imencode('.jpg', mask)
            singleton.mask.save(
                'mask.jpg',
                ContentFile(buffer.tobytes()),
                save=True
            )
            return JsonResponse({
                'status': 'success',
            })
        if test_id == "BwbJyXKl" or test_id == "G6bWk0bW" or test_id == "vPbXoPK4":
            singleton = SingletonHomographicMatrixModel.load()
            h, w = frame.shape[:2]

            x1 = int(w * 0.15)
            x2 = int(w * (1 - position_factor))
            singleton.unit_distance = unit_distance
            singleton.end_pixel = max(x1, x2)
            singleton.start_pixel = min(x1, x2)
            cv2.line(frame, (x1, 0), (x1, h), (0, 255, 0), 2)
            cv2.line(frame, (x2, 0), (x2, h), (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            singleton.file.save(
                'frame.jpg',
                ContentFile(buffer.tobytes()),
                save=True
            )
            singleton.mask.save(
                'mask.jpg',
                ContentFile(buffer.tobytes()),
                save=True
            )
            obj, created = CalibrationDataModel.objects.update_or_create(
                test_id=test_id,
                assessment_id=assessment_id,
                defaults={
                  'start_pixel' : min(x1, x2),
                    'end_pixel' : max(x1, x2),
                    'unit_distance' : unit_distance
                }
            )
            return JsonResponse({
                'status': 'success',
            })
        singleton = SingletonHomographicMatrixModel.load()
        if singleton.hsv_value:  # will be {} if not set
            h = singleton.hsv_value.get('h', DEFAULT_HSV[0])
            s = singleton.hsv_value.get('s', DEFAULT_HSV[1])
            v = singleton.hsv_value.get('v', DEFAULT_HSV[2])
        else:
            h, s, v = DEFAULT_HSV
        selected_point = [int(1280 * 0.422), int(720 * 0.75)]
        print(selected_point)
        points = process_frame_for_color_centers(frame, selected_point=selected_point, target_hsv=(h, s, v))
        print(points)
        points = merge_close_points(points, threshold=10)
        print(points)
        points_sorted = sorted(points, key=lambda p: p[1], reverse=True)
        if len(points) < 4:
            return JsonResponse({
                'status': 'error',
                'message': f'Failed to detect exactly 4 points. Detected: {len(points)}'
            }, status=400)
        cv2.imwrite("media/tihis.jpg", frame)
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
        print(order_points, world_pts)
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
        homography_obj.start_pixel_broad_jump = order_points[0][0] #taking first pixel x corrd as start point
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

        homography_obj.file.save(
            'mask.jpg',
            ContentFile(buffer.tobytes()),
            save=False
        )
        homography_obj.mask.save(
            'mask.jpg',
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
        cv2.imwrite("media/cal.jpg", frame)
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
        cv2.imwrite("media/tihis.jpg", frame)
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
            'mask.jpg',
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
    """
    Recieves an image and coorinate of point on image, stores the color of that point
    """
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
        cv2.imwrite("media/ths.jpg", img)
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
        cv2.imwrite("media/this.jpg", output_img)
        _, buffer = cv2.imencode('.jpg', output_img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

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


def get_homograph(request):
    test_id = request.GET.get('test_id', "not_sit_and_reach")
    homograph_obj = SingletonHomographicMatrixModel.load()
    if test_id != "vPbXoPK4" and test_id != "BwbJyXKl":
        response_data = {
            'square_size': homograph_obj.unit_distance,
            'matrix_url': homograph_obj.matrix.url if homograph_obj.matrix else "",
            'image_url': homograph_obj.file.url if homograph_obj.file else "",
        }
    else:
        response_data = {
            'square_size': homograph_obj.unit_distance,
            'matrix_url': homograph_obj.matrix.url if homograph_obj.matrix else "",
            'image_url': homograph_obj.mask.url if homograph_obj.mask else "",
        }
    return JsonResponse({
        'status' : 'success',
        'calibration_info': response_data
    })


