from typing import Tuple

import numpy as np
from numpy.linalg import norm


def cartesian_to_spherical(P: np.ndarray):
    # normalizzazione
    P_norm = P / norm(P, axis=-1, keepdims=True)
    # estrai le componenti di ogni punto
    x, y, z = P_norm[..., 0:1], P_norm[..., 1:2], P_norm[..., 2:]
    # phi = arctan(x/z)
    # theta = arcsin(y)
    phi, theta = np.arctan2(x, z), np.arcsin(y)

    # rimetti insieme i punti in coordinate sferiche
    return np.concatenate([phi, theta], axis=-1)


def spherical_to_img(P: np.ndarray, img_size: Tuple):
    theta, phi = P[..., 0:1], P[..., 1:]

    # normalizzazione affinché U,V stiano in [0,1] e
    # calcolo delle coordinate dei pixel
    U = (theta / (2 * np.pi) + 0.5) * (img_size[1] - 1)     # -1 per non "uscire fuori" dall'immagine
    V = (phi / np.pi + 0.5) * (img_size[0] - 1)

    # rimetti insieme i punti in coordinate sul piano dell'immagine
    return np.concatenate([U, V], axis=-1)


