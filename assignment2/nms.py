import cv2
import numpy as np
from test import load_test_set

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
def compute_iou (bbox_selected, bbox_compared) :
    area_intersection_rectangle = compute_area_intersection_rectangles(bbox_selected, bbox_compared)
    area_bbox_selected = compute_area_rect(bbox_selected)
    area_bbox_compared = compute_area_rect(bbox_compared)
    area_sum_rectangles = compute_sum_rectangles(area_bbox_selected, area_bbox_compared, area_intersection_rectangle)
    iou = area_intersection_rectangle / area_sum_rectangles
    return iou
def nms (bboxes, threshold = 0.5):
    # Ordino le bbox in base allo score, ossia il 5 elemento dell'array [x1,y1,x2,y2,c] rappresentate la bbox
    index_ordered = np.argsort(bboxes[:])[::-1]
    bboxes_ordered = bboxes[index_ordered]
    #array di bbox corrette
    filtered_bboxes = []

    while True:
        bbox_selected = bboxes_ordered.pop(0)
        filtered_bboxes.append(bbox_selected)
        if len(bboxes_ordered) == 0:
            break
        #confronto la bbox "corretta" con quelle rimanenti, calcolando l'indice iou
        for bbox_compared in bboxes_ordered:
            iou = compute_iou(bbox_selected, bbox_compared)
            if iou > threshold:
                bboxes_ordered.remove(bbox_compared)

    return filtered_bboxes
def compute_scores (all_bboxes_filtered, all_annotated_bboxes) :
    tot_tp, tot_fp, tot_fn = 0,0,0
    for image_bboxes in all_bboxes_filtered:
        for image_annotated_bboxes in all_annotated_bboxes:
            tp, fp, fn = get_image_scores(image_annotated_bboxes, image_bboxes)
            tot_tp = tot_tp + tp
            tot_fp = tot_fp + fp
            tot_fn = tot_fn + fn
    return tot_tp, tot_fp, tot_fn
def get_image_scores(annotated_boxes, retrivied_bboxes):
    tp, fp, fn = 0, 0, 0
    for retrieved_bbox in retrivied_bboxes:
        max_iou = 0
        for annoted_box in annotated_boxes:
            iou = compute_iou(retrieved_bbox, annoted_box)
            if iou > max_iou:
                max_iou = iou
        if max_iou < 0.5 and max_iou > 0:
            fp += 1
        elif max_iou == 0:
            fn += 1
        else:
            tp +=1
    return tp, fp, fn

# [ ((float, float), [(int, int, int, int, np.ndarray)])]
def test_model(factors_with_bboxes_founded):
    images, all_annotated_bboxes = load_test_set()
    all_bboxes_filtered = np.array([])
    i = 0
    for scales, bboxes_founded in factors_with_bboxes_founded:
        h_factors, w_factors = scales
        for bbox in bboxes_founded:
            bbox[0] = int(bbox[0] * scales[1])
            bbox[1] = int(bbox[1] * scales[0])
            bbox[2] = int(bbox[2] * scales[1])
            bbox[3] = int(bbox[3] * scales[0])
        print(bboxes_founded)
        bboxes_filtered = nms(bboxes_founded)
        image = images[i]
        for bbox in bboxes_filtered:
            cv2.rectangle(image, pt1= (bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=(0, 255, 0), thickness=2)
        cv2.imshow("Pippo", image)
        cv2.waitKey(0)
        i +=1
        all_bboxes_filtered += bboxes_filtered
    tp, fp, tn = compute_scores(all_bboxes_filtered)
    return tp, fp, tn





