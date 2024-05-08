from typing import List, Tuple, Union

import cv2
import numpy as np
from sklearn.pipeline import Pipeline

StdImage = Tuple[np.ndarray, float, float]
StdRatios = Tuple[float, float]
Bbox = Tuple[int, int, int, int]
Prediction = Tuple[Bbox, float]


def imgs_to_std_size(images: List[np.ndarray], w: int = 500, h: int = 700) -> List[StdImage]:
    std_images = []
    for image in images:
        img_h, img_w, _ = image.shape
        w_ratio = w / img_w
        h_ratio = h / img_h
        std_image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
        std_images.append((std_image, w_ratio, h_ratio))

    return std_images


def scale_image(image: np.ndarray, scale: float) -> np.ndarray:
    h, w, _ = image.shape
    new_h, new_w = int(h * scale), int(w * scale)

    return cv2.resize(image, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)


def gaussian_pyramid(
        image: np.ndarray,
        sigma: int = 5,
        kernel_size: (int, int) = (3, 3),
) -> List[np.ndarray]:
    images = []

    for scale in [1.3, 1.0, 0.5]:
        if scale < 1.0:
            # add filter if scaling down image to correct edge sharpening
            filtered_image = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma, sigmaY=sigma)
            images.append(scale_image(filtered_image, scale))
        elif scale > 1.0:
            images.append(scale_image(image, scale))
        else:
            images.append(image)

    return images


def sliding_window(
        image: np.ndarray,
        win_size: (int, int),
        stride: (int, int) = (30, 30),
) -> (Bbox, np.ndarray):
    for x1 in range(0, image.shape[0], stride[0]):
        for y1 in range(0, image.shape[1], stride[1]):
            x2, y2 = x1 + win_size[1], y1 + win_size[0]
            bbox = (x1, y1, x2, y2)
            yield bbox, image[x1:x2, y1:y2]


def compute_hog(image: np.ndarray) -> np.ndarray:
    x = cv2.HOGDescriptor().compute(image)
    return np.array(x)


def detect(
        image: np.ndarray,
        clf: Pipeline,
        win_size: (int, int) = (64, 128),
) -> List[Prediction]:
    preds = []

    for (bbox, cropped_image) in sliding_window(image, win_size):
        cropped_w, cropped_h, _ = cropped_image.shape

        # skip if bbox overflows
        if cropped_w != win_size[1] or cropped_h != win_size[0]:
            continue

        x = compute_hog(image)
        y = clf.predict(x)[0]

        # skip bbox if no pedestrian is detected
        if y == -1:
            continue

        confidence = clf.decision_function(x)[0]

        preds.append((bbox, confidence))

    return preds


def predict(images: List[np.ndarray], clf: Pipeline) -> List[Tuple[StdRatios, List[Prediction]]]:
    result = []

    std_images = imgs_to_std_size(images)
    for (image, w_ratio, h_ratio) in std_images:
        detections = []
        for scaled_image in gaussian_pyramid(image):
            detections += detect(scaled_image, clf)

        result += [((w_ratio, h_ratio), detections)]

    return result
