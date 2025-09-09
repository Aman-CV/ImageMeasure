import numpy as np
import cv2

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

import numpy as np
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
