import csv
import os
import numpy as np
from Tracker import Tracker


def _bbox_converter(bbox):
    """
    bbox_converter converts bounding box from the form [x_c, y_c, w, h] to [x_l, y_h, w, h]
    :param bbox: a list of [x_c, y_c, w, h]
    :return: bbox: a list of [x_l, y_h, w, h]
    """
    new_x = bbox[0] - bbox[2] / 2
    new_y = bbox[1] - bbox[3] / 2
    bbox[0] = new_x
    bbox[1] = new_y
    return bbox


def _write_csv_file(path: str, frame: int, tracker: np.ndarray, x, y, z, challenge):

    if challenge == '2D':
        x, y, z = -1, -1, -1

    with open(path, 'a', newline='') as file_csv:
        writer = csv.writer(file_csv)

        id = tracker[4]
        bbox = tracker[0:4]
        confidence = tracker[8]
        converted_bbox = _bbox_converter(bbox)
        bb_left = converted_bbox[0]
        bb_top = converted_bbox[1]
        bb_width = converted_bbox[2]
        bb_height = converted_bbox[3]

        writer.writerow([frame, id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z])


def save_results(video_name: str, frame: int, trackers: np.ndarray, x=None, y=None, z=None, challenge='2D'):

    directory_path = 'TrackEval/data/trackers/mot_challenge/MOT17-train/my_trackers/data'

    try:
        os.makedirs(directory_path)
    except FileExistsError:
        pass

    file_path = os.path.join(directory_path, video_name)

    for trackers in trackers:
        _write_csv_file(file_path, frame, trackers, x, y, z, challenge)
