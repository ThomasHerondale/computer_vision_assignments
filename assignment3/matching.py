import math
from scipy.optimize import linear_sum_assignment
import numpy as np
from skimage.feature import hog
from skimage import exposure
import torch
from detection import get_detections


def compute_cost(detection_point, tracking_point) -> float:
    """
    Function to compute the cost of linking two points. This cost is calculated as euclidian distance
    between the two points.
    :param: detection_point: list of [x_c, y_c, area, aspect_ratio] of the frame in t-1
    :param: tracking_point: list of [x_c, y_c, area, aspect_ratio] of the frame in t
    :return: d: is cost of linking two points
    """
    d_x = tracking_point[0] - detection_point[0]
    d_y = tracking_point[1] - detection_point[1]
    d = float(math.sqrt((d_x * d_x) + (d_y * d_y)))

    return d


def calculate_cost_matrix(detection_list, track_list) -> np.ndarray:

    cost_matrix = np.zeros((len(track_list), len(detection_list)), np.float32)

    for trak_i, track in enumerate(track_list):
        for det_i, detection in enumerate(detection_list):
            cost = compute_cost(detection, track)
            cost_matrix[trak_i, det_i] = cost
    return cost_matrix


def compute_descriptor_hog(img, bbox):
    """
    Compute the HOG descriptor of the object
    """
    x1, y1, x2, y2 = bbox.astype(int)
    cropped_image = img[y1:y2, x1:x2]

    # Calcola l'HOG dell'immagine ritagliata
    features, hog_image = hog(cropped_image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True, multichannel=True)

    # Normalizzo l'HOG
    features = exposure.rescale_intensity(features, in_range=(0, 10))

    return features


def matching(img, detections, tracks, threshold: int):
    """
    :param tracks: lista di oggetti alla quale è gia stato assegnato un id
    """
    track_ind: list[int] = [i for i in range(len(tracks))]
    detect_ind: list[int] = [i for i in range(len(detections))]

    if len(detect_ind) == 0 or len(track_ind) == 0:
        return [], track_ind, detect_ind

    cost_matrix = calculate_cost_matrix(detections, tracks)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    match_list, unmatched_tracks, unmatched_detections = [], [], []

    unmatched_detections = list(set(range(len(detections))) - set(col_ind))
    unmatched_tracks = list(set(range(len(tracks))) - set(row_ind))

    for row, col in zip(row_ind, col_ind):
        track_id = track_ind[row]
        detection_id = detect_ind[col]
        if cost_matrix[row, col] > threshold:
            unmatched_tracks.append(row)
            unmatched_detections.append(col)
        else:
            match_list.append([detection_id, track_id])

    return match_list, unmatched_tracks, unmatched_detections

