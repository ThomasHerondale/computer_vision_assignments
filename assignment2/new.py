import logging
import os
import random
import warnings
from typing import Literal

import cv2
import numpy as np
from sklearn.utils import shuffle

from readenv import loads


def preprocess(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def read_pos_images_list(img_list_path, limit=None):
    img_fnames = []
    counter = 0
    with open(img_list_path, 'r') as f:
        for line in f:
            if limit and counter == limit:
                break
            img_fname = f'{line.strip()}.jpg'
            img_fnames.append(img_fname)
            counter += 1
    return img_fnames


def build_neg_images_list(imgs_dir, limit=None):
    # get all available images names
    img_fnames = [
        fname
        for fname in os.listdir(imgs_dir)
        if os.path.isfile(os.path.join(imgs_dir, fname))
    ]
    limit = len(img_fnames) if limit is None else limit
    return random.sample(population=img_fnames, k=limit)


# k number of bboxes per img
def build_neg_samples(img_fnames, imgs_dir, k, bbox_size=(64, 128), preproc: Literal['BW'] = 'BW'):
    neg_samples = []
    for fname in img_fnames:
        img_path = os.path.join(imgs_dir, fname)

        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f'Could not load image {fname}')

        if preproc == 'BW':
            img = preprocess(img)

        for _ in range(k):
            bbox = random_bbox(img.shape[:2], bbox_size)
            sample = crop_on_bbox(img, bbox, bbox_size)
            neg_samples.append(sample)

    return neg_samples


def build_pos_samples(imgs, bboxes):
    pos_samples = []
    for img, bboxes in zip(imgs, bboxes):
        pos_samples += [
            crop_on_bbox(img, bbox)
            for bbox in bboxes
        ]

    return pos_samples


def random_bbox(img_shape, size=(64, 128)):
    h, w = img_shape
    while True:
        x_1 = random.randint(0, w - size[0] + 1)
        y_1 = random.randint(0, h - size[1] + 1)
        x_2 = random.randint(x_1, w - 1)
        y_2 = y_1 + 2 * x_2
        if y_2 < h and is_bbox_valid([x_1, y_1, x_2, y_2]):
            break

    return [x_1, y_1, x_2, y_2]


def read_pos_images(img_fnames, imgs_dir, preproc: Literal['BW'] = 'BW'):
    images = []
    for fname in img_fnames:
        img_path = os.path.join(imgs_dir, fname)

        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f'Could not load image {fname}')

        if preproc == 'BW':
            img = preprocess(img)
        images.append(img)

    return images


def read_bboxes(img_fnames, annotations_path):
    bboxes = []
    for img_name in img_fnames:
        img_bboxes = []
        img_annotation_path = os.path.join(annotations_path, f'{img_name}.txt')
        with open(img_annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                # if bounding box type is 'pedestrian'
                if line.startswith('1'):
                    bbox = line.split()
                    bbox = [int(n) for n in bbox[1:]]
                    if is_bbox_valid(bbox):
                        img_bboxes.append(bbox)
                    else:
                        logging.info(f'bbox {bbox} malformed: skipping')
        bboxes.append(img_bboxes)

    return bboxes


def is_bbox_valid(bbox):
    # check if bbox is empty
    if bbox:
        # check bbox size
        x1, y1, x2, y2 = bbox
        return x1 != x2 and y1 != y2
    else:
        return False


def crop_on_bbox(img, bbox: list, size=(64, 128)):
    assert len(bbox) == 4, 'Bounding box should have two vertices'
    x1, y1, x2, y2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    return cv2.resize(cropped_img, size)


def build_samples(n=None):
    fnames = read_pos_images_list(os.environ['TRAIN_ASSIGNMENT_TXT_PATH'], n)
    bboxes = read_bboxes(fnames, os.environ['ANNOTATIONS_PATH'])
    imgs = read_pos_images(fnames, os.environ['POS_IMAGES_PATH'])
    pos_samples = build_pos_samples(imgs, bboxes)

    fnames = build_neg_images_list(os.environ['NEG_IMAGES_PATH'], n)
    # compute how many bboxes to extract per negative image,
    # in order to, have an equal number of positive and negative samples
    neg_factor = len(pos_samples) // len(fnames)
    neg_samples = build_neg_samples(fnames, os.environ['NEG_IMAGES_PATH'], neg_factor)

    return pos_samples, neg_samples


def build_dataset(pos_samples, neg_samples, size=None, cache=True):
    hog = cv2.HOGDescriptor()
    data = []
    targets = []
    for sample in pos_samples:
        data.append(hog.compute(sample))
        targets.append(1)
    for sample in neg_samples:
        data.append(hog.compute(sample))
        targets.append(-1)

    assert len(data) == len(targets), "Samples and targets should have same length."
    # shuffle samples
    data, targets = shuffle(data, targets)
    # slice if desired size was specified
    if size:
        data, targets = data[:size], targets[:size]

    X, y = np.array(data, dtype=np.float32), np.array(targets, dtype=np.float32)

    if cache:
        # join features and targets
        data = np.column_stack((X, y))
        cache_ndarray(data, 'train_descriptor_data.npy')

    return X, y


def load_dataset(use_cache=True, size=None) -> (np.ndarray, np.ndarray):
    if use_cache:
        dir_path = os.environ.get('CACHE_DIR_PATH')
        file_path = os.path.join(dir_path, 'train_descriptor_data.npy')
        if os.path.isfile(file_path):
            if size is not None:
                warnings.warn("Size can't be specified when loading dataset from cache, it's being ignored.")
            data = np.load(file_path)
            return data[:, :-1], data[:, -1]
        else:
            warnings.warn('Cache file not found. Rebuilding dataset from scratch...')

    # if we didn't return, we didn't use cache -> rebuild dataset
    pos_samples, neg_samples = build_samples(size)
    return build_dataset(pos_samples, neg_samples, size, use_cache)


def cache_ndarray(arr, fname):
    dir_path = os.environ.get('CACHE_DIR_PATH')
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, fname)

    # write to file
    np.save(file_path, arr)


if __name__ == '__main__':
    X, y = load_dataset(use_cache=True)