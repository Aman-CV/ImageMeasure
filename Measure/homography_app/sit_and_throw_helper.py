import numpy as np
import cv2
from ultralytics import YOLO
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d



def analyze_positions(positions):

    positions = np.array(positions)
    frame_numbers = positions[:, 0]
    x_raw = positions[:, 1]
    y_raw = positions[:, 2]

    total_frames = frame_numbers[-1]
    half_point = total_frames // 2


    first_half_mask  = frame_numbers < half_point
    second_half_mask = frame_numbers >= half_point

    y_first  = y_raw[first_half_mask]
    y_second = y_raw[second_half_mask].copy()
    frames_second = frame_numbers[second_half_mask]
    x_second = x_raw[second_half_mask]


    mask_missing = (y_second == 0)
    if np.any(~mask_missing):
        y_second[mask_missing] = np.interp(
            frames_second[mask_missing],
            frames_second[~mask_missing],
            y_second[~mask_missing]
        )


    y_second_med = medfilt(y_second, kernel_size=5)


    y_second_smooth = gaussian_filter1d(y_second_med, sigma=2)


    y_final = y_raw.copy()
    y_final[second_half_mask] = y_second_smooth


    y = y_second_smooth
    peaks = []

    prominence = 1.0
    window = 5

    for i in range(window, len(y) - window):

        left = y[i - window:i]
        right = y[i + 1:i + 1 + window]

        if y[i] != np.max(y[i - window:i + window + 1]):
            continue

        baseline = max(np.min(left), np.min(right))

        if y[i] - baseline >= prominence:
            peaks.append(i)

    if len(peaks) == 0:
        print("No peak detected in second half, using global max.")
        peak_idx_local = np.argmax(y)
    else:
        peak_idx_local = peaks[0]

    idx = peak_idx_local

    start = max(idx - 2, 0)
    end = min(idx + 3, len(y_second))

    local_y = y_second[start:end]

    offset = np.argmax(local_y)
    peak_idx_local = start + offset

    peak_frame = frames_second[peak_idx_local]
    peak_y = y_second[peak_idx_local]
    peak_x = x_second[peak_idx_local]




    return int(peak_frame), int(peak_x), float(peak_y)



def get_first_bounce_frame_MOG(inp, start_cutoff=0.25):
    cap = cv2.VideoCapture(inp)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("motion_output.mp4", fourcc, fps, (w, h))

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=True)

    frame_no = 0
    positions = []
    model = YOLO("yolov8x.pt")  # pre-trained on COCO dataset
    use_ai = True
    while True:
        ret, frame0 = cap.read()
        if not ret:
            break

        frame_no += 1
        frame = cv2.resize(frame0, (1280, 720))
        frame[:, :int(start_cutoff * frame.shape[1])] = 0

        fgmask = fgbg.apply(frame)

        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)


        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        curr_pos = [frame_no, 0, 0]
        if use_ai:
            results = model.predict(frame, classes=[32], conf=0.2, verbose=False)  # sports ball only
            detected = False
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy[0]  # take first detected ball
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.rectangle(frame0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame0, (cx, cy), 6, (255, 0, 255), -1)
                    cv2.putText(frame0, f"Frame {frame_no}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                    curr_pos = [frame_no, cx, cy]
                    detected = True
                    break

            if not detected:
                curr_pos = [frame_no, 0, 0]

        if not use_ai or (len(contours) > 0 and (curr_pos[1]==0 and curr_pos[2] == 0)):

            best = max(contours, key=lambda c: cv2.contourArea(c))

            if cv2.contourArea(best) > 80:
                x, y, w_box, h_box = cv2.boundingRect(best)

                cx = x + w_box // 2
                cy = y + h_box // 2

                cv2.rectangle(frame0, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
                cv2.circle(frame0, (cx, cy), 6, (0, 255, 255), -1)
                cv2.putText(frame0, f"...", (x, y-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                cv2.putText(frame0, f"Frame {frame_no}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                curr_pos = [frame_no, cx, cy]
        positions.append(curr_pos)
        out.write(frame0)


    cap.release()
    out.release()

    print("Saved output video: motion_output.mp4")

    positions = np.array(positions)

    peak_frame, peak_x, peak_y = analyze_positions(positions)
    return peak_x, peak_y, peak_frame

def get_first_bounce_frame(inp):
    cap = cv2.VideoCapture(inp)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("oo.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), (1280, 720))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half_point = total_frames // 4

    ret, frame1 = cap.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

    prev_centroids = []
    positions = []

    frame_no = 0

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        frame_no += 1
        frame2 = cv2.resize(frame2, (1280, 720))
        frame2[:, :int(0.15 * frame2.shape[1])] = 0
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

        diff = cv2.absdiff(gray1, gray2)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        curr_centroids = []
        displacements = []

        for contour in contours:
            if cv2.contourArea(contour) < 200:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            curr_centroids.append((cx, cy))

            if prev_centroids:
                distances = [np.linalg.norm(np.array([cx, cy]) - np.array(p)) for p in prev_centroids]
                displacements.append((max(distances), contour))
            else:
                displacements.append((0, contour))

        if displacements:
            largest_disp, best_contour = max(displacements, key=lambda x: x[0])
            x, y, w, h = cv2.boundingRect(best_contour)

            roi = frame2[y:y + h, x:x + w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

            circle_box_used = False
            circles = cv2.HoughCircles(
                gray_roi,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=20,
                param1=50,
                param2=15,
                minRadius=5,
                maxRadius=int(min(w, h) * 0.5)
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                cx_r, cy_r, r = circles[0]

                cx = x + cx_r
                cy = y + cy_r

                cv2.circle(frame2, (cx, cy), r, (0, 255, 255), 3)

                bx1 = cx - r
                by1 = cy - r
                bx2 = cx + r
                by2 = cy + r
                cv2.rectangle(frame2, (bx1, by1), (bx2, by2), (0, 255, 255), 2)

                circle_box_used = True
            else:
                cx = x + w // 2
                cy = y + h // 2

                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 3)

            cv2.putText(frame2, f"{largest_disp:.1f}px",
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame2, f"{largest_disp:.1f}px", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


            if frame_no >= half_point:
                if frame_no == 69:
                    print(cx, positions[-1], positions[-2])
                positions.append([frame_no, cy, cx])

        gray1 = gray2.copy()
        prev_centroids = curr_centroids.copy()
        out.write(frame2)

    cap.release()
    out.release()

    positions = np.array(positions)

    if len(positions) > 0:

        frame_nums = positions[:, 0]
        y_vals = positions[:, 1]
        x_vals = positions[:, 2]
        mean_y = np.mean(y_vals)
        std_y = np.std(y_vals)

        mask = np.abs(y_vals - mean_y) < 2 * std_y
        frame_nums_clean = frame_nums[mask]
        y_clean = y_vals[mask]

        window = 5
        smooth_y = np.convolve(y_clean, np.ones(window) / window, mode='same')


        smooth_y = np.array(smooth_y)
        frame_nums_clean = np.array(frame_nums_clean)

        y_max = np.max(smooth_y)
        y_max_idx = np.argmax(smooth_y)

        threshold = 0.90 * y_max

        candidate_peaks = []
        smooth_y = y_clean
        for i in range(1, y_max_idx):
            if smooth_y[i] > smooth_y[i - 1] and smooth_y[i] > smooth_y[i + 1]:
                if smooth_y[i] >= threshold:
                    candidate_peaks.append(i)

        if len(candidate_peaks) == 0:
            selected_idx = y_max_idx - 1
        else:
            selected_idx = candidate_peaks[0] - 1

        selected_frame = int(frame_nums_clean[selected_idx])
        selected_y = float(y_vals[selected_idx])
        selected_x = 0.5 * (x_vals[selected_idx + 1] + x_vals[selected_idx])

        print(x_vals[selected_idx], x_vals[selected_idx + 1] )
        return selected_x, selected_y, selected_frame

    else:
        return None, None, None

