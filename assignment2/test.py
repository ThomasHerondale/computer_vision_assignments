import os
import warnings

import cv2
import numpy as np

from data import read_pos_images_list, read_pos_images, build_neg_images_list, read_bboxes
from nms import non_maxima_suppression, image_score
from window import Prediction, StdRatios, Bbox
from sklearn.utils import shuffle
from typing import List, Tuple

from joblib import load, dump


def load_test_set(use_cache=True) -> (list[np.ndarray], list[list[list[int]]]):
    if use_cache:
        dir_path = os.environ['CACHE_DIR_PATH']
        test_set_images_path = os.path.join(dir_path, "test_images_data.joblib")
        test_set_annotations_path = os.path.join(dir_path, "test_annotations_data.joblib")
        if os.path.isfile(test_set_images_path) and os.path.isfile(test_set_annotations_path):
            images = load(test_set_images_path)
            annotations = load(test_set_annotations_path)
            return images, annotations
        else:
            warnings.warn('Test set cache files not found. Rebuilding dataset from scratch...')

    pos_fnames = read_pos_images_list(os.environ['TEST_ASSIGNMENT_TXT_PATH'])
    pos_images = read_pos_images(pos_fnames, os.environ['POS_IMAGES_PATH'])
    neg_fnames = build_neg_images_list(os.environ['TEST_IMAGES_PATH'])
    neg_images = read_pos_images(neg_fnames, os.environ['TEST_IMAGES_PATH'])

    images = pos_images + neg_images
    bboxes = read_bboxes(pos_fnames, os.environ['ANNOTATIONS_PATH'])
    # add empty list for each negative test image, since there are no bboxes
    bboxes += [[] for _ in range(len(neg_images))]

    images, bboxes = shuffle(images, bboxes, random_state=42)
    if use_cache:
        dir_path = os.environ['CACHE_DIR_PATH']
        test_set_images_path = os.path.join(dir_path, "test_images_data.joblib")
        test_set_annotations_path = os.path.join(dir_path, "test_annotations_data.joblib")
        dump(images, test_set_images_path)
        dump(bboxes, test_set_annotations_path)

    return images, bboxes


def test_model(
        images: List[np.ndarray],
        bboxes: List[List[Bbox]],
        std_predictions: List[Tuple[StdRatios, List[Prediction]]]
):
    total_tp, total_fp, total_fn = 0, 0, 0
    for image, (std_ratios, predictions), targets in zip(images, std_predictions, bboxes):
        scaled_preds = []
        # rescale every bbox because of the image stretch
        for bbox, confidence in predictions:
            x_ratio, y_ratio = std_ratios
            x1, y1, x2, y2 = bbox
            scaled_bbox = (
                int(x1 * x_ratio),
                int(y1 * y_ratio),
                int(x2 * x_ratio),
                int(y2 * y_ratio)
            )
            scaled_preds.append((scaled_bbox, confidence))

        filtered_bboxes = non_maxima_suppression(scaled_preds)

        for bbox in filtered_bboxes:
            cv2.rectangle(
                image,
                pt1=(bbox[0], bbox[1]),
                pt2=(bbox[2], bbox[3]),
                color=(0, 255, 0),
                thickness=2
            )
        cv2.imshow("", image)
        cv2.waitKey(0)

        tp, fp, fn = image_score(filtered_bboxes, targets)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    return total_tp, total_fp, total_fn
