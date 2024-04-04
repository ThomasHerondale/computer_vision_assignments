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
        raise ValueError('Axis provided must be x or y.')


def to_equirectangular(
        img: np.ndarray,
        FOV: np.float32,
        theta: np.float32,
        phi: np.float32,
        img_size: Tuple[int, int]
):
    assert 0 <= FOV <= 90
    H = np.tan(np.radians(FOV / 2.0))
    W = H

    # generazione dei punti del piano dell'immagine
    u = (np.linspace(
        -W,
        W,
        num=img_size[1]
    ))
    v = (np.linspace(
        -H,
        H,
        num=img_size[0]
    ))
    u, v = np.meshgrid(u, v)
    w = np.ones(u.shape)
    P = np.stack([u, v, w], axis=-1)

    # rotazione del piano sul punto di tangenza desiderato
    R_x = rotation_matrix('x', phi)
    R_y = rotation_matrix('y', theta)
    P = P @ R_x @ R_y

    # determinazione della trasformazione equirettangolare
    Q = cartesian_to_spherical(P)
    T = spherical_to_img(Q, img.shape).astype(np.float32)

    # inverse mapping e interpolazione
    return cv2.remap(
        img,
        T[..., 0],
        T[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP
    )
