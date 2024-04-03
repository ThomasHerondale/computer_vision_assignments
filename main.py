from typing import Literal, Tuple

import cv2
import numpy as np
from conversions import cartesian_to_spherical, spherical_to_img


def rotation_matrix(axis: Literal['x', 'y'], theta: np.float32):
    s = np.sin(theta)
    c = np.cos(theta)
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == 'y':
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    else:
        raise ValueError('Asse sbagliato coglione')


def load_img(fname: str):
    video = cv2.VideoCapture(fname)
    if video is None or not video.isOpened():
        raise IOError('Video could not be opened.')
    ret, frame = video.read()
    return frame, frame.shape


def to_equirectangular(img: np.ndarray, FOV, theta, phi, img_size: Tuple):
    H = np.tan(FOV / 2)
    W = np.tan(FOV / 2)

    # genera i punti del piano dell'immagine
    u = (np.linspace(
        -W,
        W,
        num=img_size[1],
        dtype=np.int32))
    v = (np.linspace(
        -H,
        H,
        num=img_size[0],
        dtype=np.int32))

    P = np.ones((*img_size, 3), dtype=np.float32)
    print(np.stack(np.meshgrid(u, -v), axis=-1).shape)
    P[:, :2] = np.stack(np.meshgrid(u, -v), axis=-1)

    R_x = rotation_matrix('x', phi)
    R_y = rotation_matrix('y', theta)
    P = P @ R_x @ R_y

    Q = cartesian_to_spherical(P)
    T = spherical_to_img(Q, (H, W))

    return cv2.remap(img, T[:, 0], T[:, 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


if __name__ == '__main__':
    img1, shape = load_img('data/video_1.MP4')
    img2 = to_equirectangular(img1, 60, 0, 0, (shape[0], shape[1]))
    cv2.imshow('img2', img2)
