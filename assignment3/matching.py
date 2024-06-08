import math

from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from detection import get_detections

def convert_bbx(bbox):
    """
    Converts bounding box from form [x1, y1, x2, y2] to [x_c, y_c, area, aspect_ratio]
    :param bbox: list of [x1, y1, x2, y2]
    :return: list of [x_c, y_c, area, aspect_ratio]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    area = w * h
    aspect_ratio = w / h
    x_c = (bbox[0] + w) / 2
    y_c = (bbox[1] + h) / 2

    return np.array([x_c, y_c, area, aspect_ratio], dtype=np.float32)


def compute_cost(detection_point, tracking_point, threshold: int):
    """
    Function to compute the cost of linking two points. This cost is calculated as euclidian distance
    between the two points.
    :param: detection_point: list of [x_c, y_c, area, aspect_ratio] of the frame in t-1
    :param: tracking_point: list of [x_c, y_c, area, aspect_ratio] of the frame in t
    :return: d: is cost of linking two points
    """
    d_x = detection_point[0] - tracking_point[0]
    d_y = detection_point[1] - tracking_point[1]
    d = float(math.sqrt((d_x * d_x) + (d_y * d_y)))

    if d > threshold:
        d = np.inf  # float('inf')

    return d


def calculate_cost_matrix(detection_list, track_list, threshold) -> np.ndarray:

    cost_matrix = np.zeros((len(track_list), len(detection_list)), np.float32)
    print(cost_matrix)

    for trak_i, track in enumerate(track_list):
        for det_i, detection in enumerate(detection_list):
            cost = compute_cost(detection, track, threshold)
            cost_matrix[trak_i, det_i] = cost
    print("----------------------------------")
    print(cost_matrix)
    return cost_matrix


def matching(detections, tracks, threshold: int = 20):

    cost_matrix = calculate_cost_matrix(detections, tracks, threshold)

    track_ind: list[int] = [i for i in range(len(tracks))]
    detect_ind: list[int] = [i for i in range(len(detections))]

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    match_list, unmatched_tracks, unmatched_detections = [], [], []

    for col, det_index in enumerate(detect_ind):
        if col not in col_ind:
            unmatched_detections.append(det_index)

    for row, track_index in enumerate(track_ind):
        if row not in row_ind:
            unmatched_tracks.append(track_index)

    for row, col in zip(row_ind, col_ind):
        track_index = track_ind[row]
        detection_ind = detect_ind[col]
        if cost_matrix[row, col] > threshold:
            unmatched_tracks.append(track_index)
            unmatched_tracks.append(detection_ind)
        else:
            match_list.append((track_index, detection_ind))
    return match_list, unmatched_tracks, unmatched_detections


