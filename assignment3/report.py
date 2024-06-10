import csv

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


def write_csv_file(file_name, frame, tracker: Tracker, confidence, x=None, y=None, z=None, challenge='2D'):

    if challenge == '2D':
        x, y, z = -1, -1, -1

    with open(f'{file_name}.txt', 'a', newline='') as file_csv:
        writer = csv.writer(file_csv)

        id = tracker.id
        bbox = tracker.bbox
        converted_bbox = _bbox_converter(bbox)
        bb_left = converted_bbox[0]
        bb_top = converted_bbox[1]
        bb_width = converted_bbox[2]
        bb_height = converted_bbox[3]

        writer.writerow([frame, id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z])
