import cv2
import numpy as np
from ultralytics import YOLO

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
        ("head", 0),
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

                for p1, p2 in zip(points[:-1], points[1:]):
                    if p1 is not None and p2 is not None:
                        cv2.line(frame, p1, p2, (255, 0, 0), 2)

                if points[0] is not None and points[-1] is not None:
                    cv2.line(frame, points[0], points[-1], (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
