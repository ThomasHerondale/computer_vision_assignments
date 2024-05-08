import os
import warnings

import numpy as np

from new import read_pos_images_list, read_pos_images, build_neg_images_list, read_bboxes
from sklearn.utils import shuffle

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
    pos_images = read_pos_images(pos_fnames, os.environ['POS_IMAGES_PATH'], preproc=None)
    neg_fnames = build_neg_images_list(os.environ['TEST_IMAGES_PATH'])
    neg_images = read_pos_images(neg_fnames, os.environ['TEST_IMAGES_PATH'], preproc=None)

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
