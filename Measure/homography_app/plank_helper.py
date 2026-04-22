import cv2
import numpy as np
from ultralytics import YOLO


import numpy as np


def draw_angle(frame, a, b, c, radius=40, color=(0, 255, 255), thickness=2):

    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    angle1 = np.degrees(np.arctan2(ba[1], ba[0]))
    angle2 = np.degrees(np.arctan2(bc[1], bc[0]))

    start_angle = int(min(angle1, angle2))
    end_angle = int(max(angle1, angle2))

    cv2.ellipse(
        frame,
        b,
        (radius, radius),
        0,
        start_angle,
        end_angle,
        color,
        thickness
    )


def core_engagement_level(hip_angle):


    if  hip_angle is None:
        return "Unavailable"

    if hip_angle >= 160:
        return "Strong"
    elif hip_angle >= 140:
        return "Developing"
    else:
        return "Needs Support"


def calculate_plank_angles(shoulder, hip, knee, ankle):


    def angle(a, b, c):

        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba == 0 or norm_bc == 0:
            return None

        cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return np.degrees(np.arccos(cos_angle))

    if any(p is None for p in [shoulder, hip, knee, ankle]):
        return None, None
    angle_shk = angle(shoulder, hip, knee)
    angle_hka = angle(hip, knee, ankle)

    return angle_shk, angle_hka


def mark_right_side_pose(
    video_path,
    output_path="motion_output.mp4",
    model_path="yolov8m-pose.pt",
    conf=0.25,
    video_obj=None
):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    tfc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    frame_no = 0
    RIGHT_CHAIN = [
        ("shoulder", 6),
        ("hip", 12),
        ("knee", 14),
        ("ankle", 16)
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)
        frame_no += 1

        # Safe Progress Update
        if video_obj and tfc > 0 and (frame_no % max(1, int(tfc / 10)) == 0):
            video_obj.progress = int((frame_no / tfc) * 100)
            video_obj.save(update_fields=["progress"])

        # Check if results and keypoints exist
        if results and len(results[0].keypoints.xy) > 0:
            # results[0].keypoints.xy is a tensor of shape [NumPersons, 17, 2]
            kpts_all = results[0].keypoints.xy.cpu().numpy()
            
            # Ensure the first person has actual data (shape [17, 2])
            if kpts_all.shape[0] > 0 and kpts_all[0].shape[0] > 16:
                person = kpts_all[0]
                points = []

                for _, idx in RIGHT_CHAIN:
                    # Double check index safety against the person array size
                    if idx < len(person):
                        x, y = person[idx]
                        if x > 0 and y > 0:
                            points.append((int(x), int(y)))
                        else:
                            points.append(None)
                    else:
                        points.append(None)

                # Draw Points
                for p in points:
                    if p is not None:
                        cv2.circle(frame, p, 6, (0, 255, 0), -1)

                # Logic only if ALL points are detected
                if all(p is not None for p in points):
                    shoulder, hip, knee, ankle = points
                    
                    # Wrap draw_angle in try-except in case of math errors (div by zero)
                    try:
                        draw_angle(frame, shoulder, hip, knee)
                        hipdelta, kneedalta = calculate_plank_angles(shoulder, hip, knee, ankle)
                        
                        cv2.putText(
                            frame,
                            f"Core Engagement: {core_engagement_level(hipdelta)}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA
                        )
                    except Exception as e:
                        print(f"Drawing/Angle error: {e}")

                # Draw skeleton lines (only between valid consecutive points)
                for p1, p2 in zip(points[:-1], points[1:]):
                    if p1 is not None and p2 is not None:
                        cv2.line(frame, p1, p2, (255, 0, 0), 2)

                # Draw long line (Shoulder to Ankle)
                if points[0] is not None and points[-1] is not None:
                    cv2.line(frame, points[0], points[-1], (0, 0, 255), 2)
            else:
                # No person detected in this specific frame, write original frame
                pass
        
        out.write(frame)

    cap.release()
    out.release()