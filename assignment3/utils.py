import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

def iou(a: np.ndarray,b: np.ndarray) -> float:
    """
    calculate iou between two boxes
    """
    a_tl, a_br = a[:4].reshape((2, 2))
    b_tl, b_br = b[:4].reshape((2, 2))
    int_tl = np.maximum(a_tl, b_tl)
    int_br = np.minimum(a_br, b_br)
    int_area = np.product(np.maximum(0., int_br - int_tl))
    a_area = np.product(a_br - a_tl)
    b_area = np.product(b_br - b_tl)
    return int_area / (a_area + b_area - int_area)


def compare_boxes(detections,trackers,iou_thresh=0.3):

    iou_matrix = np.zeros(shape=(len(detections),len(trackers)),dtype=np.float32)

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = iou(det,trk)
    
    # calculate maximum iou for each pair through hungarian algorithm
    row_id, col_id = linear_sum_assignment(-iou_matrix)
    matched_indices = np.transpose(np.array([row_id,col_id]))
    # geting matched ious
    iou_values = np.array([iou_matrix[row_id,col_id] for row_id,col_id in matched_indices])
    best_indices = matched_indices[iou_values > iou_thresh]

    unmatched_detection_indices = np.array([d for d in range(len(detections)) if d not in best_indices[:,0]])  
    unmatched_trackers_indices = np.array([t for t in range(len(trackers)) if t not in best_indices[:,1]])

    return best_indices,unmatched_detection_indices,unmatched_trackers_indices


