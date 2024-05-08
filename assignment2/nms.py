import numpy as np
from assignment2.window import Bbox, Prediction, Tuple
from typing import List


def area(bbox: Bbox) -> int:
    x1, y1, x2, y2 = bbox
    return np.abs((x2 - x1) * (y2 - y1))


def intersection(bbox1: Bbox, bbox2: Bbox) -> int:
    a1, b1, c1, d1 = bbox1
    a2, b2, c2, d2 = bbox2

    a_min = max(a1, a2)
    a_max = min(c1, c2)
    b_min = max(b1, b2)
    b_max = min(d1, d2)

    return area((a_min, b_min, a_max, b_max))


def area_of_union(bbox1: Bbox, bbox2: Bbox) -> int:
    return area(bbox1) + area(bbox2) - intersection(bbox1, bbox2)


def iou(bbox1: Bbox, bbox2: Bbox) -> float:
    return intersection(bbox1, bbox2) / area_of_union(bbox1, bbox2)


def non_maxima_suppression(preds: List[Prediction], threshold: float = 0.5) -> List[Bbox]:
    result = []

    # sort bbox list by confidence score
    preds = sorted(preds, key=lambda pred: pred[1], reverse=True)
    bboxes = [bbox for bbox, _ in preds]
    for bbox1 in bboxes:
        result.append(bbox1)
        for bbox2 in bboxes:
            if bbox1 == bbox2:
                continue
            if iou(bbox1, bbox2) > threshold:
                bboxes.remove(bbox2)

    return result


def image_score(targets: List[Bbox], preds: List[Bbox]) -> Tuple[int, int, int]:
    tp = 0
    for pred in preds:
        max_iou = 0
        pred_match, target_match = None, None
        for target in targets:
            # look for best matching predicted bbox for target
            if iou(pred, target) > max_iou:
                max_iou = iou(pred, target)
                pred_match, target_match = pred, target

        if max_iou >= 0.5:
            tp += 1
            preds.remove(pred_match)
            targets.remove(target_match)

    fp, fn = len(preds), len(targets)

    return tp, fp, fn
