import numpy as np
import cv2

DEFAULT_HSV = np.array([172, 180, 180], dtype=np.uint8)
TOL_H, TOL_S, TOL_V = 20, 50, 50

from scipy import interpolate, signal
from scipy.signal import savgol_filter, find_peaks
import json
import math
import requests
from dotenv import load_dotenv
import os

load_dotenv()

TOKEN = os.getenv("VIDEO_UPLOAD_TOKEN")

def test_video_url(assessment_id, test_id, participant_id, vurl):
    # domain = "http://127.0.0.1:8000"
    domain = "http://ec2-13-126-18-144.ap-south-1.compute.amazonaws.com"
    url = f"{domain}/api/coaching/assessment/member/video_upload/"
    # Test payload with multiple users
    payload = {
        "variant_scores": [
            {
                "assessment_id": assessment_id,
                "member_id":  participant_id,
                "variant_id": test_id,
                "video_url": vurl
            }
        ]
    }

    headers = {
        'Content-Type': 'application/json',
        "Authorization": TOKEN
    }
    try:
        print("Sending POST request to:", url)
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=4))
    except Exception as e:
        print(f"Error: {str(e)}")



def correct_white_balance(img, strength=1.8):
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Compute averages for a and b channels
    avg_a = np.average(lab[:, :, 1])
    avg_b = np.average(lab[:, :, 2])

    # Calculate correction strength (lower = gentler)
    correction_scale = (lab[:, :, 0] / 255.0) * strength

    # Apply correction with clamping to avoid extreme shifts
    lab[:, :, 1] = np.clip(lab[:, :, 1] - ((avg_a - 128) * correction_scale), 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] - ((avg_b - 128) * correction_scale), 0, 255)

    # Convert back to BGR
    balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return balanced


def correct_white_balance_deprecated(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    avg_a = np.average(result[:, :, 1])

    avg_b = np.average(result[:, :, 2])

    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 2.)

    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 2.)

    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)



def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))
    return x1, y1, x2, y2



def get_ankle_mask(frame, clahe=None):
    # Get image dimensions
    h, w = frame.shape[:2]
    half_h = 0
    lower_half = frame[half_h:h, :]

    # Define 7 basic BGR colors
    basic_bgr = np.uint8([
        [0, 0, 255],    # red
        [0, 69, 255],   # orange
        [0, 255, 255],  # yellow
        [0, 200, 0],    # green
        [255, 0, 0],    # blue
        [130, 0, 75],   # indigo
        [238, 130, 238],# violet
    ])

    basic_lab = cv2.cvtColor(basic_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    YELLOW_IDX = 2  # index for yellow in the array

    # Convert frame to LAB
    lab = cv2.cvtColor(lower_half, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Optional illumination correction
    if clahe is not None:
        L = clahe.apply(L)
        lab = cv2.merge((L, A, B))

    # Use only chroma channels (a,b)
    hh, ww = L.shape
    ab = lab[:, :, 1:3].reshape((-1, 2)).astype(np.int16)
    basic_ab = basic_lab[:, 1:3].astype(np.int16)

    # Compute squared distances between each pixel and the 7 base colors
    dists = np.sum((ab[:, None, :] - basic_ab[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1).astype(np.uint8).reshape(hh, ww)

    # Build yellow mask
    mask_yellow = (labels == YELLOW_IDX).astype(np.uint8) * 255

    # Clean up
    mask_yellow = cv2.medianBlur(mask_yellow, 5)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Create full-size mask with zeros on top half
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[half_h:h, :] = mask_yellow

    return full_mask





def ankle_crop_color_detection(frame, CLAHE=None, model=None, CROP_HALF=32):
    h, w = frame.shape[:2]

    # Run YOLO pose detection
    results = model.predict(frame, conf=0.25, verbose=False)

    highest_y = -1
    best_person_ankles = []

    for r in results:
        if hasattr(r, 'keypoints') and r.keypoints is not None:
            kps = r.keypoints.xy  # (num_people, num_kpts, 2)

            for person_kps in kps:
                person_ankles = []
                max_y_this_person = -1

                # (15 = left, 16 = right)
                for idx in [15, 16]:
                    if idx < len(person_kps):
                        x_px, y_px = person_kps[idx]
                        person_ankles.append((int(x_px), int(y_px)))
                        if y_px > max_y_this_person:
                            max_y_this_person = y_px

                if max_y_this_person > highest_y:
                    highest_y = max_y_this_person
                    best_person_ankles = person_ankles

    ankle_keypoints = best_person_ankles

    mask_full = np.zeros((h, w), dtype=np.uint8)

    for ax, ay in ankle_keypoints:
        x1 = ax - CROP_HALF
        y1 = ay - CROP_HALF
        x2 = ax + CROP_HALF
        y2 = ay + CROP_HALF
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

        # Crop around ankle
        crop = frame[y1:y2, x1:x2]

        yellow_mask = get_ankle_mask(crop, clahe=CLAHE)

        mask_full[y1:y2, x1:x2] = cv2.bitwise_or(mask_full[y1:y2, x1:x2], yellow_mask)


    return mask_full, ankle_keypoints


def stretch_contrast(frame):
    f = frame.astype(np.float32)

    f = f / 255.0


    stretched = 0.5 + (f - 0.5) * 1.2

    stretched = np.clip(stretched, 0, 1)

    return (stretched * 255).astype(np.uint8)


def detect_yellow_mask_lab(frame, clahe=None):
    # Get image dimensions
    h, w = frame.shape[:2]
    half_h = h // 2

    # Work only on the lower half
    lower_half = frame[half_h:h, :]

    # Define 7 basic BGR colors
    basic_bgr = np.uint8([
        [0, 0, 255],    # red
        [0, 69, 255],   # orange
        [0, 255, 255],  # yellow
        [0, 255, 0],    # green
        [255, 0, 0],    # blue
        [130, 0, 75],   # indigo
        [238, 130, 238],# violet
        [0, 220, 0]  # dg

    ])

    basic_lab = cv2.cvtColor(basic_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    YELLOW_IDX = 2  # index for yellow in the array

    # Convert frame to LAB
    lab = cv2.cvtColor(lower_half, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Optional illumination correction
    if clahe is not None:
        L = clahe.apply(L)
        lab = cv2.merge((L, A, B))

    # Use only chroma channels (a,b)
    hh, ww = L.shape
    ab = lab[:, :, 1:3].reshape((-1, 2)).astype(np.int16)
    basic_ab = basic_lab[:, 1:3].astype(np.int16)

    # Compute squared distances between each pixel and the 7 base colors
    dists = np.sum((ab[:, None, :] - basic_ab[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1).astype(np.uint8).reshape(hh, ww)

    # Build yellow mask
    mask_yellow = (labels == YELLOW_IDX).astype(np.uint8) * 255

    # Clean up
    mask_yellow = cv2.medianBlur(mask_yellow, 5)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Create full-size mask with zeros on top half
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[half_h:h, :] = mask_yellow

    return full_mask




def merge_close_regions(regions, max_gap=2):
    if not regions:
        return []

    # Sort regions by start index
    regions = sorted(regions, key=lambda x: x[0])
    merged = [regions[0]]

    for start, end in regions[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= max_gap:
            # Merge if gap <= max_gap
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


def get_flat_start(y, window=30):

    flat_regions = []
    i = 0
    while i <= len(y) - window:
        segment = y[i:i+window]
        if np.all(segment == 0):
            flat_regions.append((i, i + window))
            i += window
        else:
            i += 1
    flat_regions = merge_close_regions(flat_regions)
    if len(flat_regions) < 2:
        return False, [[0, 1], [0, 1]]
    else:
        mid = len(flat_regions) // 2
        return True, [flat_regions[mid-1], flat_regions[mid]]

def order_points_anticlockwise(points):
    if len(points) < 3:
        raise ValueError("Need at least 3 points to order anticlockwise.")

    max_y = max(p[1] for p in points)
    candidates = [p for p in points if abs(p[1] - max_y) < 10]
    start = min(candidates, key=lambda p: p[0])
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    def angle(p):
        return math.atan2(p[1] - cy, p[0] - cx)

    sorted_points = sorted(points, key=angle, reverse=True)

    if start in sorted_points:
        i = sorted_points.index(start)
        sorted_points = sorted_points[i:] + sorted_points[:i]

    return sorted_points


def distance_from_homography(pt1, pt2, H):
    # Convert to proper shape for cv2.perspectiveTransform → (N, 1, 2)
    pts = np.array([pt1, pt2], dtype=np.float32).reshape(-1, 1, 2)

    # Transform both points using the homography
    world_pts = cv2.perspectiveTransform(pts, H)

    # Compute Euclidean distance in world coordinates
    p1, p2 = world_pts[0, 0], world_pts[1, 0]
    distance = np.linalg.norm(p1 - p2)

    return float(distance)

def detect_biggest_jump(dy, smooth_window=11, smooth_poly=2, start_thresh=-1.5, end_thresh=0):
    start , end = 0 , len(dy)
    while start < len(dy) and dy[start] == 0:
        start += 1

    while end > start and dy[end - 1] == 0:
        end -= 1

    return start - 3, end + 3



import cv2
import numpy as np



def world_to_image(points, H):
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)  # (N,1,2)
    projected = cv2.perspectiveTransform(pts, H)
    return projected.reshape(-1, 2).astype(int)

def filter_and_smooth(coords, window_size=5, threshold=5):
    coords = np.array(coords, dtype=np.float64)
    N = len(coords)
    if N == 0:
        return coords
    coords_filtered = coords.copy()
    for i in range(N):
        start = max(0, i - window_size)
        end = min(N, i + window_size + 1)

        local_median = np.median(coords[start:end], axis=0)
        distance = np.linalg.norm(coords[i] - local_median)

        if distance > threshold or (coords[i][1] == 720 and coords[i][0] == 1280):
            coords_filtered[i] = np.array([np.nan, np.nan])
    valid_mask = ~np.isnan(coords_filtered[:, 0])
    x_valid = np.where(valid_mask)[0]
    y_valid = coords_filtered[valid_mask]

    if len(x_valid) < 4:
        kind = 'linear'
    else:
        kind = 'cubic'

    interp_func_x = interpolate.interp1d(x_valid, y_valid[:, 0], kind=kind, fill_value="extrapolate")
    interp_func_y = interpolate.interp1d(x_valid, y_valid[:, 1], kind=kind, fill_value="extrapolate")

    x_all = np.arange(N)
    coords_interpolated = np.vstack((interp_func_x(x_all), interp_func_y(x_all))).T

    smoothed_x = signal.savgol_filter(coords_interpolated[:, 0], window_length=7, polyorder=2, mode='nearest')
    smoothed_y = signal.savgol_filter(coords_interpolated[:, 1], window_length=7, polyorder=2, mode='nearest')

    smoothed_coords = np.vstack((smoothed_x, smoothed_y)).T

    return smoothed_coords



def equalize_image(img, clahe):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    enhanced_frame = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return enhanced_frame

def merge_close_points(points, threshold=10):
    merged = []
    points = points.copy()
    while points:
        base = points.pop(0)
        close_pts = [base]
        remaining = []
        for pt in points:
            if np.linalg.norm(np.array(base) - np.array(pt)) < threshold:
                close_pts.append(pt)
            else:
                remaining.append(pt)
        avg_x = int(np.mean([p[0] for p in close_pts]))
        avg_y = int(np.mean([p[1] for p in close_pts]))
        merged.append((avg_x, avg_y))
        points = remaining
    return merged

import json

def save_homography_as_json(H, path='homography_app/homography_data/homography.json'):
    H_list = H.tolist()  # Convert NumPy array to a Python list
    data = {
        "homography_matrix": H_list
    }
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"Homography saved to {path}")


def load_homography_from_json(path='homography_app/homography_data/homography.json'):
    with open(path, 'r') as f:
        data = json.load(f)
        H = np.array(data["homography_matrix"], dtype=np.float32)
    return H


def detect_and_measure_image(image_path, output_path, homography):
    picked_hsv = np.array([26, 116, 152])
    tol_h, tol_s, tol_v = 10, 50, 50
    lower = np.array([max(picked_hsv[0] - tol_h, 0),
                      max(picked_hsv[1] - tol_s, 0),
                      max(picked_hsv[2] - tol_v, 0)])
    upper = np.array([min(picked_hsv[0] + tol_h, 179),
                      min(picked_hsv[1] + tol_s, 255),
                      min(picked_hsv[2] + tol_v, 255)])

    frame = cv2.imread(image_path)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_points = []

    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            detected_points.append((cx, cy))

    # Merge close points
    def merge_close_points(points, threshold=10):
        merged = []
        points = points.copy()
        while points:
            base = points.pop(0)
            close_pts = [base]
            remaining = []
            for pt in points:
                if np.linalg.norm(np.array(base) - np.array(pt)) < threshold:
                    close_pts.append(pt)
                else:
                    remaining.append(pt)
            avg_x = int(np.mean([p[0] for p in close_pts]))
            avg_y = int(np.mean([p[1] for p in close_pts]))
            merged.append((avg_x, avg_y))
            points = remaining
        return merged

    detected_points = merge_close_points(detected_points)

    display_img = frame.copy()

    if len(detected_points) == 2:
        p1 = np.array([[detected_points[0]]], dtype=np.float32)
        p2 = np.array([[detected_points[1]]], dtype=np.float32)

        wp1 = cv2.perspectiveTransform(p1, homography)[0][0]
        wp2 = cv2.perspectiveTransform(p2, homography)[0][0]

        dist_cm = np.linalg.norm(wp1 - wp2)

        # Draw
        cv2.circle(display_img, detected_points[0], 6, (0, 0, 255), -1)
        cv2.circle(display_img, detected_points[1], 6, (0, 0, 255), -1)
        cv2.line(display_img, detected_points[0], detected_points[1], (0, 255, 0), 2)

        mid_x = (detected_points[0][0] + detected_points[1][0]) // 2
        mid_y = (detected_points[0][1] + detected_points[1][1]) // 2
        cv2.putText(display_img, f"{dist_cm:.1f} cm", (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imwrite(output_path, display_img)
        return dist_cm
    else:
        return None



def detect_carpet_segment(frame, selected_point=None):

    h, w, _ = frame.shape
    lower_half = frame[h // 2:, :]
    print(selected_point)
    hsv = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV)
    if not selected_point:
        DEFAULT_HSV = np.array([80, 80, 40], dtype=np.uint8)
    else:
        x, y = selected_point
        DEFAULT_HSV = hsv[y - 360, x]
        print("h")
    print(DEFAULT_HSV)
    TOL_H, TOL_S, TOL_V = 15, 50, 50

    center = DEFAULT_HSV.astype(int)

    # --- compute tolerance-based range safely ---
    lower_hsv = np.array([
        max(center[0] - TOL_H, 179),
        max(center[1] - TOL_S, 255),
        max(center[2] - TOL_V, 255)
    ], dtype=np.uint8)

    upper_hsv = np.array([
        min(center[0] + TOL_H, 179),
        min(center[1] + TOL_S, 255),
        min(center[2] + TOL_V, 255)
    ], dtype=np.uint8)

    lower_hsv = np.array([
        0, 0, 0
    ], dtype=np.uint8)

    upper_hsv = np.array([
        179, 255, 85
    ], dtype=np.uint8)
    #--- Mask for target color ---
    print(lower_hsv)
    print(upper_hsv)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # --- Ignore white/desaturated areas ---
    s_channel = hsv[:, :, 1]
 #   mask[s_channel < 40] = 0

    # --- Clean mask carefully ---
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.uint8(labels == largest_label) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segment_mask = np.zeros((h, w), dtype=np.uint8)

    if contours:
        selected_contour = None

        if selected_point:
            px, py = selected_point
            if py >= h // 2:
                py_adj = py - h // 2
                min_dist = float("inf")
                for cnt in contours:
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dist = np.hypot(px - cx, py_adj - cy)
                    if dist < min_dist:
                        min_dist = dist
                        selected_contour = cnt
            else:
                selected_contour = max(contours, key=cv2.contourArea)
        else:
            selected_contour = max(contours, key=cv2.contourArea)

        if selected_contour is not None:
            cv2.drawContours(segment_mask[h // 2:, :], [selected_contour], -1, 255, -1)

    segmented_region = cv2.bitwise_and(frame, frame, mask=segment_mask)

    return segment_mask, segmented_region

def separate_color_by_hsv_deprecated(segmented_region, target_hsv=(110, 122, 140), tol_h=20, tol_s=80, tol_v=80):

    hsv = cv2.cvtColor(segmented_region, cv2.COLOR_BGR2HSV)

    h, s, v = target_hsv
    lower_hsv = np.array([
        max(h - tol_h, 0),
        max(s - tol_s, 0),
        max(v - tol_v, 0)
    ], dtype=np.uint8)
    upper_hsv = np.array([
        min(h + tol_h, 179),
        min(s + tol_s, 255),
        min(v + tol_v, 255)
    ], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    separated = cv2.bitwise_and(segmented_region, segmented_region, mask=mask)

    return mask, separated


def separate_color_by_hsv(
    segmented_region,
    target_hsv=(110, 122, 140),
    tol_h=40, tol_s=80, tol_v=80
):

    # Convert to HSV
    hsv = cv2.cvtColor(segmented_region, cv2.COLOR_BGR2HSV)
    h, s, v = target_hsv

    # --- Build a fixed color palette ---
    # target color
    target_color = np.uint8([[target_hsv]])
    target_bgr = cv2.cvtColor(target_color, cv2.COLOR_HSV2BGR)[0, 0]
    target_bgr = np.array([0, 230, 230], dtype = np.uint8)
    target_bgr2 = np.array([30, 150, 200], dtype = np.uint8)

    # two dark tones
    dark_gray = np.array([50, 50, 50], dtype=np.uint8)
    black = np.array([0, 0, 0], dtype=np.uint8)

    palette = np.array([target_bgr, target_bgr2, dark_gray, black], dtype=np.uint8)

    img = segmented_region.reshape((-1, 3)).astype(np.float32)

    distances = np.linalg.norm(img[:, None, :] - palette[None, :, :].astype(np.float32), axis=2)
    labels = np.argmin(distances, axis=1)

    target_cluster = 0  # index 0 = target color
    mask = np.where((labels == target_cluster) | (labels == 1), 255, 0).astype(np.uint8)
    mask = mask.reshape(segmented_region.shape[:2])

        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

    separated = cv2.bitwise_and(segmented_region, segmented_region, mask=mask)

    return mask, separated


def get_mask_centers(mask, return_largest=False):

    mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None if return_largest else []

    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    if not centers:
        return None if return_largest else []

    if return_largest:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    else:
        return centers

def process_frame_for_color_centers(frame, selected_point=None, target_hsv=(110, 122, 140)):


    segment_mask, segmented_region = detect_carpet_segment(frame, selected_point=selected_point)
    cv2.imwrite("media/temp.jpg", segmented_region)
    if segmented_region is None or np.count_nonzero(segment_mask) == 0:
        print("No carpet region detected.")
        return []

    mask, separated = separate_color_by_hsv(segmented_region, target_hsv=target_hsv)
    cv2.imwrite("media/temp1.jpg", separated)
    cv2.imwrite("media/temp2.jpg", mask)


    centers = get_mask_centers(mask, return_largest=False)

    return centers


def highest_peak_by_adjacent_minima(y, xvs, plot=False):

    y = np.asarray(y)
    x = np.arange(len(y))

    peaks, _ = find_peaks(y)
    if len(peaks) < 2:
        return None, None, None, None, None

    peak_idx = peaks[np.argmax(y[peaks])]

    minima, _ = find_peaks(-y)
    if len(minima) < 2:
        return None, None, None, None, None
    left_min = minima[minima < peak_idx]
    start_idx = left_min[-1] if len(left_min) > 0 else 0

    right_min = minima[minima > peak_idx]
    end_idx = right_min[0] if len(right_min) > 0 else len(y) - 1

    # ---- Plot ----
    if plot:
        pass
        # plt.figure()
        # plt.plot(x, y, label="Signal")
        # plt.plot(x[peak_idx], y[peak_idx], "ro", label="Highest Peak")
        # plt.plot(x[start_idx], y[start_idx], "go", label="Start (min)")
        # plt.plot(x[end_idx], y[end_idx], "go", label="End (min)")
        # plt.fill_between(
        #     x[start_idx:end_idx + 1],
        #     y[start_idx:end_idx + 1],
        #     alpha=0.3,
        #     label="Peak Region"
        # )
        # plt.legend()
        # plt.title("Peak Boundaries via Adjacent Minima")
        # plt.show()

    return peak_idx, start_idx, end_idx, xvs[start_idx], xvs[end_idx]


import cv2
import numpy as np


def image_point_to_real_point(homograph_points, unit_distance, p, w = 1280, h = 720):

    if not homograph_points:
        raise ValueError("homograph_points is empty or None")

    img_pts = np.array([
        [homograph_points["p1"]["fx"] * w, homograph_points["p1"]["fy"] * h],
        [homograph_points["p2"]["fx"] * w, homograph_points["p2"]["fy"] * h],
        [homograph_points["p3"]["fx"] * w, homograph_points["p3"]["fy"] * h],
        [homograph_points["p4"]["fx"] * w, homograph_points["p4"]["fy"] * h],
    ], dtype=np.float32)

    world_pts = np.array([
        [0, 0],
        [unit_distance, 0],
        [unit_distance, unit_distance],
        [0, unit_distance],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(img_pts, world_pts)
    if H is None:
        raise ValueError("Homography computation failed")

    pt = np.array([[p]], dtype=np.float32)  # shape (1,1,2)
    real_pt = cv2.perspectiveTransform(pt, H)

    X, Y = real_pt[0][0]
    return float(X), float(Y)


def find_yellow_point_lab(
    frame,
    x,
    y,
    basic_lab,
    yellow_idx=2,
    window_size=32,
    clahe=None,
    min_pixels=20,
):
    """
    Returns (new_x, new_y) if yellow is found near (x, y),
    otherwise returns the original (x, y).
    """

    h, w = frame.shape[:2]
    half = window_size // 2

    x1 = max(x - half, 0)
    y1 = max(y - half, 0)
    x2 = min(x + half, w)
    y2 = min(y + half, h)

    roi = frame[y1:y2, x1:x2]

    # Convert to LAB
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    if clahe is not None:
        L, A, B = cv2.split(lab)
        L = clahe.apply(L)
        lab = cv2.merge((L, A, B))

    hh, ww = lab.shape[:2]

    # Use only a,b channels
    ab = lab[:, :, 1:3].reshape((-1, 2)).astype(np.int16)
    basic_ab = basic_lab[:, 1:3].astype(np.int16)

    # Nearest base color
    dists = np.sum((ab[:, None, :] - basic_ab[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1).reshape(hh, ww)

    # Yellow mask
    mask = (labels == yellow_idx).astype(np.uint8) * 255

    # Cleanup
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    if cv2.countNonZero(mask) < min_pixels:
        return x, y

    # Centroid
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return x, y

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return x1 + cx, y1 + cy
