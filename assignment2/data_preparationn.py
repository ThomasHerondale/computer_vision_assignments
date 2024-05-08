import os
import random
from readenv import loads
from typing import Literal

import cv2


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
    return random.sample(img_fnames, limit)


# k number of bboxes per img
def build_neg_samples(img_fnames, imgs_dir, k, bbox_size=(64, 128), preproc: Literal['Sobel'] = None):
    neg_samples = []
    for fname in img_fnames:
        img_path = os.path.join(imgs_dir, fname)

        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f'Could not load image {fname}')

        if preproc == 'Sobel':
            img = preprocess(img)

        for _ in range(k):
            bbox = random_bbox(img.shape[:2], bbox_size)
            sample = crop_on_bbox(img, bbox, bbox_size)
            cv2.imshow("sample", sample)
            cv2.waitKey(0)
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
    flag = True
    while (flag) :
        x_1 = random.randint(0, w - size[0] + 1)
        y_1 = random.randint(0, h - size[1] + 1)
        x_2 = random.randint(x_1, w-1)
        y_2 = y_1 + 2 * x_2
        if (y_2 < h) :
            flag = False
    return [x_1, y_1, x_2, y_2]


def read_pos_images(img_fnames, imgs_dir, preproc: Literal['Sobel'] = None):
    images = []
    for fname in img_fnames:
        img_path = os.path.join(imgs_dir, fname)

        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f'Could not load image {fname}')

        if preproc == 'Sobel':
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
                    if bbox:
                        img_bboxes.append(bbox)
        bboxes.append(img_bboxes)

    return bboxes


def crop_on_bbox(img, bbox: list, size=(64, 128)):
    assert len(bbox) == 4, 'Bounding box should have two vertices'
    x1, y1, x2, y2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    return cv2.resize(cropped_img, size)


def build_dataset():
    fnames = read_pos_images_list(os.environ['TRAIN_ASSIGNMENT_TXT_PATH'], 10)
    bboxes = read_bboxes(fnames, os.environ['ANNOTATIONS_PATH'])
    imgs = read_pos_images(fnames, os.environ['POS_IMAGES_PATH'])
    pos_samples = build_pos_samples(imgs, bboxes)

    fnames = build_neg_images_list(os.environ['NEG_IMAGES_PATH'], 10)
    neg_samples = build_neg_samples(fnames, os.environ['NEG_IMAGES_PATH'], 5)

    return pos_samples, neg_samples