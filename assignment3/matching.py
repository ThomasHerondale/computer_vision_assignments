import math

import cv2
from scipy.optimize import linear_sum_assignment
import numpy as np
from skimage.feature import hog
from skimage import exposure
import torch
from detection import get_detections


#  the first version
def _compute_euclidian_distance(detection_point, tracking_point) -> float:
    """
    Function to compute the cost of linking two points. This cost is calculated as euclidian distance
    between the two points.
    :param detection_point: list of [x_c, y_c, area, aspect_ratio] of the frame in t-1
    :param tracking_point: list of [x_c, y_c, area, aspect_ratio] of the frame in t
    :return d: is cost of linking two points
    """
    d_x = tracking_point[0] - detection_point[0]
    d_y = tracking_point[1] - detection_point[1]
    d = float(math.sqrt((d_x * d_x) + (d_y * d_y)))

    return d


def _convert_bbox(bbox) -> np.ndarray:
    """
    convert_bbox coverte a bbox from form [x_c, y_c, w, h] into numpy array [x_l, y_t, w, h]
    :param bbox: list of [x_c, y_c, w, h]
    :return numpy array [x_, y, w, h]
    """
    x_c, y_c, w, h = bbox
    x = int(x_c - w / 2) # forse è meglio convertire tutto a intero ?
    y = int(y_c - h / 2)
    return np.array([x, y, w, h], dtype=int)


def _compute_descriptor_hog(frame, bbox):
    """
    Compute the HOG descriptor of the object.
    :param frame: current frame of image
    :param bbox: a simple array in form [x, y, w, h]
    :return features: features of cropped image that include pedestrian
    """
    bbox_converted = _convert_bbox(bbox)
    x1, y1, w, h = bbox_converted  # estraggo gli elementi di una bbox
    cropped_image = frame[y1:(y1 + h), x1:(x1 + w)]  # ritaglio l'imagine
    resize = cv2.resize(cropped_image, dsize=(64, 128))
    # Calcola l'HOG dell'immagine ritagliata
    features, hog_image = hog(resize, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(3, 3), visualize=True, channel_axis=2)

    # Normalizzo l'HOG
    features = exposure.rescale_intensity(features, in_range=(0, 10))

    return features


def _compute_cosine_similarity(a, b):
    """
    _compute_cosine_similarity calculate the cosine of angle alpha of two vector, a and b, and then it subtract to one
    :param a: bbox's features of current frame
    :param b: bbox's features of previous frame
    """
    cosine = np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1 - cosine


def _calculate_cost_matrix(detections_list, tracks_list, detections_features, lambda_1, lambda_2) -> np.ndarray:
    """
    Compute the cost matrix using euclidian distance and cosine similarity.
    :param detections_list: detections list in the form of [x_c, y_c, width, height]
    :param tracks_list: list of tracks in the form of [x_c, y_c, width, height]
    :param detections_features: list of  HOG descriptors of current frame
    :param lambda_1: first hyperparameter
    :param lambda_2: second hyperparameter
    :return: the cost matrix
    """
    # creo la matrice di soli zero di len(tracks_list) * len(detections_list)
    cost_matrix = np.zeros((len(tracks_list), len(detections_list)), np.float32)

    # estraggo Hog descriptor dei trackers
    tracks_features = [track.descriptor for track in tracks_list]

    # costruisco la matrice dei costi
    for trak_i, track in enumerate(tracks_list):
        for det_i, detection in enumerate(detections_list):
            cost = (lambda_1 * _compute_euclidian_distance(detection, track) +
                    lambda_2 * _compute_cosine_similarity(detections_features, tracks_features))
            cost_matrix[trak_i, det_i] = cost
    return cost_matrix


def matching(current_frame, detections, tracks,lambda_1: float, lambda_2: float, threshold: int):
    """
    This function apply Hungarian algorithm to solve the assignment problem
    :param current_frame: current frame of video
    :param detections: detections list in the form of [x_c, y_c, width, height]
    :param tracks: list of tracks in the form of [x_c, y_c, width, height]
    :param lambda_1: first hyperparameter
    :param lambda_2: second hyperparameter
    :param threshold:
    """

    track_ind: list[int] = [i for i in range(len(tracks))]
    detect_ind: list[int] = [i for i in range(len(detections))]

    if len(detect_ind) == 0 or len(track_ind) == 0:
        return [], track_ind, detect_ind

    # creo la lista di fetures delle detection
    detections_features = [_compute_descriptor_hog(current_frame, detection) for detection in detections]

    cost_matrix = _calculate_cost_matrix(detections, tracks, detections_features, lambda_1, lambda_2)

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

