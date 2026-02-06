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
    conf=0.25
):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Failed to open video"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    RIGHT_CHAIN = [
     #   ("head", 0),
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

        if results and results[0].keypoints is not None:
            kpts_all = results[0].keypoints.xy.cpu().numpy()  # (N,17,2)

            if len(kpts_all) > 0:
                person = kpts_all[0]  # first detected person

                points = []
                for _, idx in RIGHT_CHAIN:
                    x, y = person[idx]
                    if x > 0 and y > 0:
                        points.append((int(x), int(y)))
                    else:
                        points.append(None)

                # draw points
                for p in points:
                    if p is not None:
                        cv2.circle(frame, p, 6, (0, 255, 0), -1)
                all_points_exist = all(p is not None for p in points)
                if None not in points:
                    shoulder, hip, knee, ankle = points
                    draw_angle(
                        frame,
                        shoulder,  # a
                        hip,  # b (vertex)
                        knee  # c
                    )
                    hipdelta, kneedalta = calculate_plank_angles(shoulder , hip, knee, ankle)

                    cv2.putText(
                        frame,
                        f"Corengagement : {core_engagement_level(hipdelta)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,  # font
                        1.0,  # font scale
                        (0, 255, 255),  # color (B, G, R)
                        2,  # thickness
                        cv2.LINE_AA
                    )

                for p1, p2 in zip(points[:-1], points[1:]):
                    if p1 is not None and p2 is not None:
                        cv2.line(frame, p1, p2, (255, 0, 0), 2)

                if points[0] is not None and points[-1] is not None:
                    cv2.line(frame, points[0], points[-1], (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
