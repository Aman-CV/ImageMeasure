import numpy as np
import cv2


def get_first_bounce_frame(inp):
    cap = cv2.VideoCapture(inp)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("oo.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), (1280, 720))

    # Get total frames to determine second half
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half_point = total_frames // 2

    ret, frame1 = cap.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

    prev_centroids = []
    positions = []  # Store [frame_no, y]

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

        # centroid + displacement
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

            # --- Extract ROI for circle detection ---
            roi = frame2[y:y + h, x:x + w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

            # --- Try to detect a circle inside ROI ---
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
                # Use first detected circle
                circles = np.round(circles[0, :]).astype("int")
                cx_r, cy_r, r = circles[0]

                # Convert ROI coords -> frame coords
                cx = x + cx_r
                cy = y + cy_r

                # Draw circle
                cv2.circle(frame2, (cx, cy), r, (0, 255, 255), 3)

                # Draw new tight bounding box around circle
                bx1 = cx - r
                by1 = cy - r
                bx2 = cx + r
                by2 = cy + r
                cv2.rectangle(frame2, (bx1, by1), (bx2, by2), (0, 255, 255), 2)

                circle_box_used = True
            else:
                # --- fallback: original bbox ---
                cx = x + w // 2
                cy = y + h // 2

                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 3)

            # Always draw displacement text
            cv2.putText(frame2, f"{largest_disp:.1f}px",
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

            # Draw
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame2, f"{largest_disp:.1f}px", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Store Y position ONLY for second half of video
            # IMPORTANT: cx, cy ALREADY computed above (circle or fallback)
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
        # ---- Outlier removal using standard deviation ----
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

        threshold = 0.90 * y_max  # 10% range

        candidate_peaks = []
        smooth_y = y_clean
        for i in range(1, y_max_idx):  # only before the max
            if smooth_y[i] > smooth_y[i - 1] and smooth_y[i] > smooth_y[i + 1]:
                if smooth_y[i] >= threshold:  # within 10% of max
                    candidate_peaks.append(i)

        if len(candidate_peaks) == 0:
            selected_idx = y_max_idx - 1
        else:
            selected_idx = candidate_peaks[0] - 1

        selected_frame = int(frame_nums_clean[selected_idx])
        selected_y = float(smooth_y[selected_idx])
        selected_x = 0.5 * (x_vals[selected_idx + 1] + x_vals[selected_idx])

        print(x_vals[selected_idx], x_vals[selected_idx + 1] )
        return selected_x, selected_y, selected_frame

    else:
        return None, None, None

