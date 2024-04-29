import os
import random

import readenv.loads

import cv2


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
def build_neg_samples(img_fnames, imgs_dir, k, bbox_size=(64, 128)):
    neg_samples = []
    for fname in img_fnames:
        img_path = os.path.join(imgs_dir, fname)

        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f'Could not load image {fname}')

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
    x = random.randint(0, w - size[0] + 1)
    y = random.randint(0, h - size[1] + 1)
    return [x, y, x + size[0], y + size[1]]


def read_pos_images(img_fnames, imgs_dir):
    images = []
    for fname in img_fnames:
        img_path = os.path.join(imgs_dir, fname)

        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            raise IOError(f'Could not load image {fname}')

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
    imgs = read_pos_images(fnames, os.environ['IMAGES_PATH'])

    pos_samples = build_pos_samples(imgs, bboxes)
    neg_samples = build_neg_samples(fnames, os.environ['IMAGES_PATH'], 5)

    return pos_samples, neg_samples