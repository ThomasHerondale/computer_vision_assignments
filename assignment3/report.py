import csv
import os
import warnings
from typing import Dict, Any

import numpy as np
import itertools as it
import subprocess


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

        writer.writerow([frame, int(id), bb_left, bb_top, bb_width, bb_height, confidence, x, y, z])


def save_results(video_name: str, frame: int, trackers: np.ndarray, x=None, y=None, z=None, challenge='2D'):
    directory_path = 'TrackEval/data/trackers/mot_challenge/MOT17-train/my_trackers/data'

    try:
        os.makedirs(directory_path)
    except FileExistsError:
        pass

    file_path = os.path.join(directory_path, f"{video_name}.txt")

    # se il file già esiste lo elimino
    if os.path.exists(file_path):
        os.remove(file_path)

    for trackers in trackers:
        _write_csv_file(file_path, frame, trackers, x, y, z, challenge)


def _write_result(file_path, output: str):
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, 'w') as file:
        file.write(output)


def compute_report(video):
    command = ("python TrackEval/scripts/run_mot_challenge.py --USE_PARALLEL False --METRICS HOTA CLEAR "
               "--TRACKERS_TO_EVAL my_trackers ")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)

        directory_path = 'TrackEval/data/trackers/mot_challenge/MOT17-train/my_trackers/data'
        file_path = os.path.join(directory_path, "output.txt")

        _write_result(file_path, result.stdout)
        return __read_score_file(file_path, video)

    except subprocess.CalledProcessError as e:
        print(e.output)
        print(e.stderr)


def __read_score_file(fname, video) -> dict[str, float]:
    with open(fname, 'r') as f:
        # skip lines until start of the tables
        while True:
            line = f.readline()
            if line.startswith('All sequences for my_trackers finished in'):
                f.readline()
                break

        # read header row to get list of scores
        header = f.readline()
        f1_metrics = header.strip().split()[2:]

        # skip 2nd header line
        # _ = f.readline()

        while True:
            line = f.readline()

            # check if we finished reading this table
            if not line or line == '\n':
                break

            name, *scores = line.strip().split()

            # skip COMBINED score
            if name == 'COMBINED':
                continue

            # check if we are only reading the rows of our interests
            assert name.startswith('MOT17-')

            if name == video:
                f1_scores = scores

            # skip useless line
            f.readline()

        # read header row to get list of scores
        header = f.readline()
        f2_metrics = header.strip().split()[2:]

        for line in f.readlines():
            # check if we finished reading this table
            if not line or line == '\n':
                break

            name, *scores = line.strip().split()

            # skip COMBINED score
            if name == 'COMBINED':
                continue

            # check if we are only reading the rows of our interests
            assert name.startswith('MOT17-')

            if name == video:
                f2_scores = scores

            # skip useless line
            f.readline()

            # check if we finished reading this table
            if line.isspace():
                break

        score_data = it.chain.from_iterable([zip(f1_metrics, f1_scores), zip(f2_metrics, f2_scores)])

        return {k: v for k, v in score_data}
