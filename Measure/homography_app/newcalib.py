import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

def segment_object_sam(
    image_bgr,
    point=None,
    checkpoint="sam_vit_b_01ec64.pth",
    kernel_size=7,
    device=None,
    debug=False,
    debug_path="debug_segmentation.png"
):


    h, w = image_bgr.shape[:2]

    if (
        point is None or
        point[0] < 0 or point[1] < 0 or
        point[0] >= w or point[1] >= h
    ):
        point = (w // 2, h // 2)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    input_points = np.array([[point[0], point[1]]])
    input_labels = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )

    best_mask = masks[np.argmax(scores)]

    mask_uint8 = best_mask.astype(np.uint8) * 255

    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((9, 9), np.uint8)

    mask_clean = cv2.morphologyEx(
        mask_uint8,
        cv2.MORPH_OPEN,
        kernel_open
    )

    mask_filled = cv2.morphologyEx(
        mask_clean,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    if debug:
        overlay = image_bgr.copy()
        overlay[mask_filled == 255] = (0, 255, 0)

        debug_img = cv2.addWeighted(
            overlay, 0.4,
            image_bgr, 0.6,
            0
        )

        cv2.circle(debug_img, point, 6, (0, 0, 255), -1)

        cv2.imwrite(debug_path, debug_img)

    return debug_img, mask_filled