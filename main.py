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
    # P = np.ones((*img_size, 3), dtype=np.float32)
    # print(f'P: {P.shape}')
    # print(f'Other: {np.stack(np.meshgrid(u, -v), axis=-1).shape}')
    # P[:, :, :2] = np.stack(np.meshgrid(u, -v), axis=-1)
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
    video.release()
    cv2.destroyAllWindows()

def browse_img(file_path: str, FOV, height: int, width: int):
    long = 0
    lat = 0
    tick = 0.2
    img = cv2.imread(file_path)
    if img is None:
        raise IOError('Image could not be opened.')
    else:
        while True:
            img = cv2.imread(file_path)
            img = to_equirectangular(
                img,
                FOV,
                theta=long,
                phi=lat,
                img_size=(height, width)
            )
            cv2.imshow("Immagine", img)
            keypress = cv2.waitKey(0)
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
            cv2.destroyAllWindows()

# Sarà necessario un metodo per stabilire se l'oggetto passato è un video o un immagine
def file_path_type(file_path:str, FOV, height, width):
    img = cv2.imread(file_path)
    video = cv2.VideoCapture(file_path)
    if img is not None:
        # viene invocato il metodo che permette di operare sull'imagine
        browse_img(file_path, FOV, height, width)
    elif video.isOpened():
        browse_video(file_path, FOV, height, width)
    else:
        print("Errore")

if __name__ == '__main__':
    print(" "*10 + "\tWelcome to Tool for navigating 360° images\t" + " "*10)
    file_path = input("Inserire il percorso file dell'immagine o del video:  \n")
    FOV = int(input("Inserire il Fov: ")) #FOV deve essere float?
    file_path_type(file_path.strip(), FOV, 720, 1080)
    #browse_video(file_path.strip(), FOV, 720, 1080)
