import cv2
from ultralytics import YOLO

def detect_crossing_rightmost_ankle(
    video_path,
    x_B,
    output_image_path="crossing_frame.jpg",
    resize_width=1280,
    resize_height=720,
    conf=0.5,
    show=False
):

    start_x = None
    start_time = None
    meters_per_pixel = None

    model = YOLO("yolov8m-pose.pt")
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("motion_output.mp4", fourcc, fps, (w, h))
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

        if r.keypoints is None or r.boxes is None or r.boxes.id is None:
            if show:
                cv2.imshow("Processing", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        keypoints = r.keypoints.xy.cpu().numpy()
        if len(keypoints) == 0 :
            continue

        ids = r.boxes.id.cpu().numpy().astype(int)

        # Select target (right-most person in first valid frame)
        if target_id is None:
            max_x = 10000
            for kp, track_id in zip(keypoints, ids):
                left_ankle = kp[15]
                right_ankle = kp[16]

                if left_ankle[0] == 0 or right_ankle[0] == 0:
                    continue

                x_ankle = max(left_ankle[0], right_ankle[0])
                if x_ankle < max_x:
                    max_x = x_ankle
                    target_id = track_id
                    prev_x = x_ankle
                    start_x = x_ankle
                    start_time = frame_number / fps

        if meters_per_pixel is None and start_x is not None:
            pixel_dist = abs(start_x - x_B)
            if pixel_dist > 0:
                meters_per_pixel = 15.0 / pixel_dist

        # Track target
        for kp, track_id in zip(keypoints, ids):
            if track_id != target_id:
                continue

            left_ankle = kp[15]
            right_ankle = kp[16]

            if left_ankle[0] == 0 or right_ankle[0] == 0:
                continue

            x_ankle = max(left_ankle[0] , right_ankle[0])  + 10
            current_time = frame_number / fps

            if start_time is not None and meters_per_pixel is not None:
                covered_m = abs(x_ankle - start_x) * meters_per_pixel
                delta_t = current_time - start_time - 3

                if delta_t > 0:
                    speed_mps = covered_m / delta_t

                    cv2.putText(
                        frame,
                        f"Speed: {100 / speed_mps:.2f} s/100m",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2
                    )

            cv2.circle(frame, (int(x_ankle), int(left_ankle[1])), 6, (0, 255, 0), -1)
            cv2.line(frame, (int(x_B), 0), (int(x_B), resize_height), (0, 0, 255), 2)

            if prev_x < x_B <= x_ankle and not crossed:
                crossed = True
                cv2.imwrite(output_image_path, frame)
                time_seconds = frame_number / fps

                cap.release()
                cv2.destroyAllWindows()
                return frame_number, time_seconds, output_image_path

            prev_x = x_ankle

        if show:
            cv2.imshow("Processing", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return None, None, None


def detect_crossing_person_box(
    video_path,
    x_B,
    output_image_path="crossing_frame.jpg",
    resize_width=1280,
    resize_height=720,
    conf=0.3,
    show=False
):
    model = YOLO("yolov8m.pt")
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_number = 0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("motion_output.mp4", fourcc, fps, (w, h))

    target_id = None
    prev_x = None
    crossed = False

    start_x = None
    start_time = None
    meters_per_pixel = None

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
            classes=[0],
            verbose=False
        )

        r = results[0]

        if r.boxes is None or r.boxes.id is None:
            if show:
                cv2.imshow("Processing", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        ids = r.boxes.id.cpu().numpy().astype(int)
        boxes = r.boxes.xyxy.cpu().numpy()


        if target_id is None:
            min_x = float("inf")
            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                x_pos = x1 + 0.75 * (x2 - x1)

                if x_pos < min_x:
                    min_x = x_pos
                    target_id = track_id
                    prev_x = x_pos
                    start_x = x_pos
                    start_time = frame_number / fps

        # --------------------------------
        # Track selected person
        # --------------------------------
        for box, track_id in zip(boxes, ids):
            if track_id != target_id:
                continue

            x1, y1, x2, y2 = box

            x_pos = x1 + 0.75 * (x2 - x1)
            y_pos = y2

            current_time = frame_number / fps

            if start_x is not None:
                if meters_per_pixel is None:
                    pixel_dist = abs(start_x - x_B)
                    if pixel_dist > 0:
                        meters_per_pixel = 15.0 / pixel_dist

                if meters_per_pixel is not None:
                    covered_m = abs(x_pos - start_x) * meters_per_pixel
                    delta_t = current_time - start_time - 3

                    if delta_t > 0:
                        speed_mps = covered_m / delta_t
                        cv2.putText(
                            frame,
                            f"Speed: {100 / speed_mps:.2f} s/100m",
                            (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 255),
                            2
                        )

            # # ---- visualization ----
            # cv2.rectangle(
            #     frame,
            #     (int(x1), int(y1)),
            #     (int(x2), int(y2)),
            #     (255, 0, 0),
            #     2
            # )
            cv2.circle(frame, (int(x_pos), int(y_pos)), 6, (0, 255, 0), -1)
            cv2.line(frame, (int(x_B), 0), (int(x_B), resize_height), (0, 0, 255), 2)

            if prev_x is not None and prev_x < x_B <= x_pos and not crossed:
                crossed = True
                cv2.imwrite(output_image_path, frame)
                time_seconds = frame_number / fps

                cap.release()
                cv2.destroyAllWindows()
                return frame_number, time_seconds, output_image_path

            prev_x = x_pos

        if show:
            cv2.imshow("Processing", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return None, None, None


if __name__ == "__main__":
    frame_id, time_sec, img_path = detect_crossing_person_box(
        video_path="/Users/notcamelcase/PycharmProjects/MyMacSpace/nttes/thistr.mp4",
        x_B=1172,
        show=True
    )

    print(frame_id, time_sec, img_path)

    a, b, c = detect_crossing_rightmost_ankle(
    video_path="/Users/notcamelcase/PycharmProjects/MyMacSpace/nttes/thistr.mp4",
    x_B=1129,
    show=True
    )
    print(a, b, c)
