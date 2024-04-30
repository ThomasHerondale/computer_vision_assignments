import cv2
import numpy as np

def compute_area_rect(rect):
    area = (rect[2] - rect[0]) * (rect[3] - rect[1])
    return area

def compute_area_intersection_rectangles(rect1, rect2):
    # Estrai le coordinate e le dimensioni dei rettangoli
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Calcola le coordinate dei vertici dei rettangoli
    x1_min, y1_min = x1, y1
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_min, y2_min = x2, y2
    x2_max, y2_max = x2 + w2, y2 + h2

    # Calcola i vertici dell'intersezione fra i due rettangoli
    x_intersection_min = max(x1_min, x2_min)
    y_intersection_min = max(y1_min, y2_min)
    x_intersection_max = min(x1_max, x2_max)
    y_intersection_max = min(y1_max, y2_max)

    # Calcola le dimensioni del rettangolo intersecante
    intersection_w = max(0, x_intersection_max - x_intersection_min)
    intersection_h = max(0, y_intersection_max - y_intersection_min)

    return compute_area_rect([x_intersection_min, y_intersection_min, intersection_w, intersection_h])

def compute_sum_rectangles(area_rect1, area_rect2, area_intersection_rect):
    return area_rect1 + area_rect2 - area_intersection_rect

def nms (bboxes, threshold):
    # Ordino le bbox in base allo score, ossia il 5 elemento dell'array [x1,y1,x2,y2,c] rappresentate la bbox
    index_ordered = np.argsort(bboxes[:, 4])[::-1]
    bboxes_ordered = bboxes[index_ordered]
    #array di bbox corrette
    filtered = []

    while True:
        bbox_selected = bboxes_ordered.pop(0)
        filtered.append(bbox_selected)
        if len(bboxes_ordered) == 0:
            break
        #confronto la bbox "corretta" con quelle rimanenti, calcolando l'indice iou
        for bbox_compared in bboxes_ordered:
            area_intersection_rectangle = compute_area_intersection_rectangles(bbox_selected, bbox_compared)
            area_bbox_selected = compute_area_rect(bbox_selected)
            area_bbox_compared = compute_area_rect(bbox_compared)
            area_sum_rectangles = compute_sum_rectangles(area_bbox_selected, area_bbox_compared, area_intersection_rectangle)
            iou = area_intersection_rectangle / area_sum_rectangles
            if iou > threshold:
                bboxes_ordered.remove(bbox_compared)

    return filtered
