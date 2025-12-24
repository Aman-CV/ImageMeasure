import cv2
import numpy as np


def estimate_distance_between_points(centers, known_distance_cm=15):

    import math

    if len(centers) < 3:
        return None

    P1, P2, P3 = centers

    # pixel distances
    d12_pix = math.hypot(P1[0]-P2[0], P1[1]-P2[1])
    d23_pix = math.hypot(P2[0]-P3[0], P2[1]-P3[1])

    if d12_pix == 0:
        return None

    scale = known_distance_cm / d12_pix
    est_d23_cm = d23_pix * scale
    print(est_d23_cm)
    return est_d23_cm

def detect_yellow_strip_positions_mask(frame, mask, hz, margin_ratio=0.02, max_hits=3):


    h, w = frame.shape[:2]

    margin = int(h * margin_ratio)
    y_top = max(0, hz - margin)
    y_bottom = min(h, hz + margin)

    strip = frame[y_top:y_bottom]
    strip_mask = mask[y_top:y_bottom]

    strip = cv2.bitwise_and(strip, strip, mask=strip_mask)

    yellow = np.array([0, 255, 255], dtype=np.uint8)
    dark_gray = np.array([40, 40, 40], dtype=np.uint8)
    grey = np.array([80, 80, 80], dtype=np.uint8)
    black = np.array([0, 0, 0], dtype=np.uint8)

    palette = np.array([yellow, dark_gray, black, yellow], dtype=np.uint8)

    pixels = strip.reshape((-1, 3)).astype(np.float32)
    distances = np.linalg.norm(
        pixels[:, None, :] - palette[None, :, :].astype(np.float32), axis=2
    )
    labels = np.argmin(distances, axis=1)
    label_img = labels.reshape(strip.shape[:2])

    yellow_indices = np.where(np.all(palette == yellow, axis=1))[0]
    yellow_mask = np.zeros(strip.shape[:2], dtype=np.uint8)
    for idx in yellow_indices:
        yellow_mask[label_img == idx] = 255

    yellow_mask_full = np.zeros((h, w), dtype=np.uint8)
    yellow_mask_full[y_top:y_bottom, :] = yellow_mask
    cv2.imwrite("media/this_strip.jpg", yellow_mask_full)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    yellow_mask_full = cv2.morphologyEx(yellow_mask_full, cv2.MORPH_OPEN, kernel)
    cv2.imwrite("media/this_strip_morph.jpg", yellow_mask_full)

    return yellow_mask_full

def find_three_centers_from_mask(yellow_mask_full):
    contours, _ = cv2.findContours(
        yellow_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contour_centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h != 0 else 0
        if aspect_ratio < 0.4:
            continue
        contour_centroids.append((cnt, cx, cy))

    if not contour_centroids:
        return []

    contour_centroids.sort(key=lambda c: c[1], reverse=True)

    centers = [(cx, cy) for (_, cx, cy) in contour_centroids[:3]]

    return centers




def detect_carpet_segment_p(frame, p=0.75):


    h, w, _ = frame.shape

    center_x = w // 2
    center_y = int(p * h)
    selected_point = (center_x, center_y)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([0, 0, 0], dtype=np.uint8)
    upper_hsv = np.array([179, 255, 85], dtype=np.uint8)

    # mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    s = hsv[:, :, 1]
    mask[s < 40] = 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    hz = 0.75 * h
    if num > 1:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_dist = float("inf")
        selected_contour = None

        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            dist = np.hypot(center_x - cx, center_y - cy)
            if dist < min_dist:
                min_dist = dist
                selected_contour = cnt
                hz = cy

        segment_mask = np.zeros((h, w), dtype=np.uint8)
        if selected_contour is not None:
            cv2.drawContours(segment_mask, [selected_contour], -1, 255, -1)



    else:
        segment_mask = np.zeros((h, w), dtype=np.uint8)
    segmented_region = cv2.bitwise_and(frame, frame, mask=segment_mask)


    segmented_region2 = cv2.bitwise_and(frame, frame, mask=segment_mask)
    y1 = int( h * 0.75) # use guidance instead of middle part detection hz

    cv2.line(segmented_region, (0, y1), (w, y1), (255, 255, 255), 2)
    return segment_mask, segmented_region, segmented_region2


