import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
from typing import Tuple
import matplotlib.pyplot as plt

import os

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def _convert_bbox(bbox: torch.Tensor) -> torch.Tensor:
    """
    Converts the bounding box specified by its center coordinates and its size to
    a bbox expressed by the coordinates of its vertices.
    :param bbox: bounding box as `[c_x, c_y, w, h]`
    :return: the bounding box as `[x_1, y_1, x_2, y_2]`
    """
    x_c, y_c, w, h = bbox.unbind(dim=1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def _rescale_bbox(bbox: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
    """
    Converts the bbox specified by its center coordinates in [0, 1] and its size to
    a bbox expressed by the coordinates of its vertices rescaled with respect to the image size.
    :param bbox: the bounding box as `[c_x, c_y, w, h]`, where c_x, c_y are in [0, 1]
    :param img_size: the size of the image
    :return: the bounding box as `[x_1, y_1, x_2, y_2]` in image scale
    """
    img_w, img_h = img_size
    bbox = _convert_bbox(bbox)
    bbox = bbox * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return bbox


def detect(model, img, transform=None, confidence_threshold=0.5, peopleOnly=True) -> (torch.Tensor, torch.Tensor):
    # default preprocessing pipeline
    if transform is None:
        transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            # todo: normalize?
        ])

    img_size = img.size

    # preprocess image
    img = transform(img).unsqueeze(dim=0)

    # compute prediction
    outputs = model(img)

    # compute prediction probabilities
    probs = outputs['pred_logits'].softmax(dim=-1)[0, :, :-1]

    # suppress predictions below confidence treshold
    to_keep = probs.max(dim=-1).values > confidence_threshold
    probs = probs[to_keep]

    bboxes_scaled = _rescale_bbox(outputs['pred_boxes'][0, to_keep], img_size)

    confidence_scores, bboxes = [], []

    if peopleOnly:
        for prob, bbox in zip(probs, bboxes_scaled.tolist()):
            cl = prob.argmax()
            if CLASSES[cl] != 'person':
                continue
            confidence_scores.append(prob[cl])
            bboxes.append(bbox)
    else:
        confidence_scores = probs
        bboxes = bboxes_scaled

    # return each bbox with its confidence score
    return torch.tensor(confidence_scores, dtype=torch.float32), torch.tensor(bboxes, dtype=torch.float32)


def plot_results(pil_img, prob, boxes):
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        # if conf_score is 0-dimensional, there's a single confidence score per bounding box
        # so only people are being detected
        if len(p.shape) == 0:
            text = f'{p:0.2f}'
        else:
            cl = p.argmax()
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.pause(0.1)


def show_video_detections(frames_path):
    plt.figure(figsize=(16, 10))
    for img_path in os.listdir(frames_path):
        plt.clf()
        img = Image.open(os.path.join(path, img_path))
        conf_scores, bboxes_scaled = detect(model, img, peopleOnly=True)
        plot_results(img, conf_scores, bboxes_scaled)


if __name__ == '__main__':
    path = "MOT17/train/MOT17-02-DPM/img1"
    model = torch.hub.load(
        'facebookresearch/detr:main',
        'detr_resnet50',
        pretrained=True,
    )
    show_video_detections(path)
