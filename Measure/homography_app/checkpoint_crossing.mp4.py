import cv2
from ultralytics import YOLO

def detect_crossing_rightmost_ankle(
    video_path,
    x_B,
    output_image_path="crossing_frame.jpg",
    resize_width=1280,
    resize_height=720,
    conf=0.5
):
    """
    Detect crossing of point B using ankle keypoints.
    Tracks only the right-most person detected in the first valid frame.

    Returns:
        frame_number (int)
        time_seconds (float)
        output_image_path (str)
    """

    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0

    target_id = None
    prev_x = None
    crossed = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        frame = cv2.resize(frame, (resize_width, resize_height))

        results = model.track(
            frame,
            persist=True,
            conf=conf,
            verbose=False
        )

        r = results[0]

        if r.keypoints is None or r.boxes is None:
            continue

        keypoints = r.keypoints.xy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)


        if target_id is None:
            max_x = 10000

            for kp, track_id in zip(keypoints, ids):
                left_ankle = kp[15]
                right_ankle = kp[16]

                if left_ankle[0] == 0 or right_ankle[0] == 0:
                    continue

                x_ankle = max(left_ankle[0] , right_ankle[0])

                if x_ankle < max_x:
                    max_x = x_ankle
                    target_id = track_id
                    prev_x = x_ankle

            continue

        for kp, track_id in zip(keypoints, ids):
            if track_id != target_id:
                continue

            left_ankle = kp[15]
            right_ankle = kp[16]

            if left_ankle[0] == 0 or right_ankle[0] == 0:
                continue

            x_ankle = (left_ankle[0] + right_ankle[0]) / 2

            if prev_x < x_B <= x_ankle and not crossed:
                crossed = True

                cv2.imwrite(output_image_path, frame)
                time_seconds = frame_number / fps

                cap.release()
                return frame_number, time_seconds, output_image_path

            prev_x = x_ankle

    cap.release()
    return None, None, None
