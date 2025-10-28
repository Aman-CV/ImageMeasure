import numpy as np


DEFAULT_HSV = np.array([172, 180, 180], dtype=np.uint8)
TOL_H, TOL_S, TOL_V = 20, 50, 50

from scipy import interpolate, signal
from scipy.signal import savgol_filter, find_peaks
import json
import math



def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))
    return x1, y1, x2, y2



def get_ankle_mask(frame, clahe=None):
    h, w = frame.shape[:2]
    half_h = 0
    # target full ankle
    lower_half = frame[half_h:h, :]

    #  basic BGR colors
    basic_bgr = np.uint8([
        [0, 0, 255],    # red
        [0, 69, 255],   # orange
        [0, 255, 255],  # yellow
        [0, 255, 0],    # green
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
    ankle_keypoints = []

    for r in results:
        if hasattr(r, 'keypoints') and r.keypoints is not None:
            kps = r.keypoints.xy  # (num_people, num_kpts, 2)
            for person_kps in kps:
                for idx in [15, 16]:  # 16  = right ankkll
                    if idx < len(person_kps):
                        x_px, y_px = person_kps[idx]
                        ankle_keypoints.append((int(x_px), int(y_px)))

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


    return mask_full




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

        if distance > threshold:
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
