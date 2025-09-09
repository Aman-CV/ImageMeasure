import json
import os

from django.core.files.base import ContentFile
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage, default_storage
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import base64
from .helper import merge_close_points, save_homography_as_json, detect_and_measure_image
from django.conf import settings

DEFAULT_HSV = np.array([26, 116, 152], dtype=np.uint8)
TOL_H, TOL_S, TOL_V = 10, 50, 50


@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        file_path = default_storage.save(f'videos/{video_file.name}', ContentFile(video_file.read()))
        return JsonResponse({'message': 'Video uploaded', 'file_path': file_path})
    return JsonResponse({'error': 'No video uploaded'}, status=400)



def mark_distance_image(request):
    # Load homography matrix from JSON
    json_path = os.path.join(settings.BASE_DIR, 'homography_app/homography_data/homography.json')
    with open(json_path, "r") as f:
        data = json.load(f)
    H = np.array(data["homography_matrix"], dtype=np.float32)
    print(H)
    if request.method == 'POST' and 'x1' in request.POST and 'y1' in request.POST and 'x2' in request.POST and 'y2' in request.POST:
        # AJAX request from JS to calculate distance
        x1 = int(request.POST['x1'])
        y1 = int(request.POST['y1'])
        x2 = int(request.POST['x2'])
        y2 = int(request.POST['y2'])

        p1 = np.array([[[x1, y1]]], dtype=np.float32)
        p2 = np.array([[[x2, y2]]], dtype=np.float32)
        print(p1, p2)
        wp1 = cv2.perspectiveTransform(p1, H)[0][0]
        wp2 = cv2.perspectiveTransform(p2, H)[0][0]
        dist_cm = np.linalg.norm(wp1 - wp2)

        print(f"Distance calculated: {dist_cm:.2f} cm")

        return JsonResponse({
            "status": True,
            "distance": float(round(dist_cm, 2))  # convert to native float
        })

    # Handle image upload
    uploaded_image_url = None
    if request.method == 'POST' and 'image_file' in request.FILES:
        image_file = request.FILES['image_file']
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        uploaded_image_url = fs.url(filename)

    context = {
        "uploaded_image_url": uploaded_image_url
    }
    return render(request, 'mark_distance_image.html', context)

def mark_distance(request):
    global clicked_points
    context = {}
    if request.method == 'POST':
        x1 = int(request.POST.get('x1'))
        y1 = int(request.POST.get('y1'))
        x2 = int(request.POST.get('x2'))
        y2 = int(request.POST.get('y2'))

        # Load homography matrix
        json_path = os.path.join(settings.BASE_DIR, "homography_app/homography_data/homography.json")

        # Load homography matrix from JSON
        with open(json_path, "r") as f:
            data = json.load(f)

        homography_list = data.get("homography_matrix")

        # Convert to numpy array
        H = np.array(homography_list, dtype=np.float32)


        p1 = np.array([[[x1, y1]]], dtype=np.float32)
        p2 = np.array([[[x2, y2]]], dtype=np.float32)

        wp1 = cv2.perspectiveTransform(p1, H)[0][0]
        wp2 = cv2.perspectiveTransform(p2, H)[0][0]
        dist_cm = np.linalg.norm(wp1 - wp2)

        context['distance'] = round(dist_cm, 2)

    return render(request, 'mark_distance.html', context)

def upload_image(request):
    result = None
    detected_image = None

    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        np_img = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # HSV mask
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h,s,v = DEFAULT_HSV
        lower = np.array([max(h-TOL_H,0), max(s-TOL_S,0), max(v-TOL_V,0)])
        upper = np.array([min(h+TOL_H,179), min(s+TOL_S,255), min(v+TOL_V,255)])
        mask = cv2.inRange(hsv_frame, lower, upper)

        # Contours and centroids
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"]>0:
                cx = M["m10"]/M["m00"]
                cy = M["m01"]/M["m00"]
                points.append((cx, cy))

        points = merge_close_points(points)

        if len(points) != 4:
            result = f"Failed to detect 4 points. Detected: {len(points)}"
        else:
            # Order points
            pts = np.array(points, dtype=np.float32)
            total_x = sum([pt[0] for pt in pts]) / 4.
            total_y = sum([pt[1] for pt in pts]) / 4.
            print(pts)
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



            world_pts = np.array([[0,0],[60,0],[60,60],[0,60]], dtype=np.float32)
            H, _ = cv2.findHomography(order_points, world_pts)

            result = f"Homography matrix:\n{H}"
            save_homography_as_json(H)
            # Draw points on image
            for idx, (x, y) in enumerate(order_points):
                cv2.circle(frame, (int(x), int(y)), 6, (0,0,255), -1)
                cv2.putText(frame, str(idx+1), (int(x)+5,int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

            # Convert image to base64 to show in HTML
            _, buffer = cv2.imencode('.jpg', frame)
            detected_image = base64.b64encode(buffer).decode('utf-8')
            detected_image = f"data:image/jpeg;base64,{detected_image}"

    return render(request, 'upload.html', {'result': result, 'detected_image': detected_image})

def measure_distance_view(request):
    result_image_path = None
    message = None

    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]

        # Save uploaded image temporarily
        upload_path = os.path.join(settings.MEDIA_ROOT, image_file.name)
        with open(upload_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Load homography matrix
        json_path = os.path.join(settings.BASE_DIR, "homography_app/homography_data/homography.json")

        # Load homography matrix from JSON
        with open(json_path, "r") as f:
            data = json.load(f)

        homography_list = data.get("homography_matrix")

        # Convert to numpy array
        H = np.array(homography_list, dtype=np.float32)

        # Run detection and measurement
        output_path = os.path.join(settings.MEDIA_ROOT, f"result_{image_file.name}")
        dist_cm = detect_and_measure_image(upload_path, output_path, homography=H)

        if dist_cm is not None:
            message = f"Distance measured: {dist_cm:.1f} cm"
            result_image_path = f"/media/result_{image_file.name}"
        else:
            message = "Failed to detect exactly 2 points."

    return render(request, "measure_distance.html", {
        "result_image": result_image_path,
        "message": message
    })