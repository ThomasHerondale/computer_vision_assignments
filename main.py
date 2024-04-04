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
        raise ValueError('Invalid axis')

def load_img(fname: str):
    video = cv2.VideoCapture(fname)
    if video is None or not video.isOpened():
        raise IOError('Video could not be opened.')
    ret, frame = video.read()
    return frame, frame.shape

def to_equirectangular(img: np.ndarray, FOV, theta, phi, img_size: Tuple):
    H = np.tan(np.radians(FOV / 2.0))
    W = np.tan(np.radians(FOV / 2.0))

    #H, W = img_size[0], img_size[1]

    # genera i punti del piano dell'immagine
    u = (np.linspace(
        -W,
        W,
        num=img_size[1]))
    v = (np.linspace(
        -H,
        H,
        num=img_size[0]))
    u, v = np.meshgrid(u, v)
    w = np.ones_like(u)
    P = np.concatenate([u[..., None], v[..., None], w[..., None]], axis=-1)
    R_x = rotation_matrix('x', phi)
    R_y = rotation_matrix('y', theta)
    P = P @ R_x @ R_y
    Q = cartesian_to_spherical(P)
    T = spherical_to_img(Q, img.shape).astype(np.float32)
    return cv2.remap(img, T[..., 0], T[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

def browse_video(fname: str, FOV, height, width):
    long = 0
    lat = 0
    tick = 0.2
    video = cv2.VideoCapture(fname)
    if video is None or not video.isOpened():
        raise IOError('Video could not be opened.')
    ret, frame = video.read()
    while ret:
        ret, frame = video.read()
        frame = to_equirectangular(
            frame,
            FOV,
            theta=long,
            phi=lat,
            img_size=(height, width)
            )
        cv2.imshow('Video', frame)
        keypress = cv2.waitKey(25)
        if keypress == ord('q'):
            break
        elif keypress == ord('s'):
            lat += tick
        elif keypress == ord('w'):
            lat -= tick
        elif keypress == ord('a'):
            long += tick
        elif keypress == ord('d'):
            long -= tick
        elif keypress == ord('z') and FOV < 85:
            FOV += 2
        elif keypress == ord('x') and FOV > 13:
            FOV -= 2
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    browse_video('video_1.MP4', 60, 720, 1080)
