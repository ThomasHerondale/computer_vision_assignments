import logging
import os
import warnings
from typing import Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from alive_progress import alive_it, alive_bar

from assignment3.utils import get_dir_path

__CLASSES = [
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


__logger = logging.getLogger('alive_progress')


# suppress warnings related to our weights choice
warnings.filterwarnings('ignore')
__model = torch.hub.load(
        'facebookresearch/detr:main',
        'detr_resnet50',
        pretrained=True,
    )
warnings.filterwarnings('default')


def __convert_bbox(bbox: torch.Tensor) -> torch.Tensor:
    """
    Converts the bounding box specified by its center coordinates and its size to
    a bbox expressed by the coordinates of its vertices.
    :param bbox: bounding box as `[c_x, c_y, w, h]`
    :return: the bounding box as `[x_1, y_1, x_2, y_2]`
    """
    x_c, y_c, w, h = bbox.unbind(dim=1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def __rescale_bbox(bbox: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
    """
    Converts the bbox specified by its center coordinates in [0, 1] and its size to
    a bbox expressed by the coordinates of its vertices rescaled with respect to the image size.
    :param bbox: the bounding box as `[c_x, c_y, w, h]`, where c_x, c_y are in [0, 1]
    :param img_size: the size of the image
    :return: the bounding box as `[x_1, y_1, x_2, y_2]` in image scale
    """
    img_w, img_h = img_size
    bbox = __convert_bbox(bbox)
    bbox = bbox * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return bbox


def __detect(model, img, transform=None, confidence_threshold=0.5, peopleOnly=True) -> (torch.Tensor, torch.Tensor):
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

    bboxes_scaled = __rescale_bbox(outputs['pred_boxes'][0, to_keep], img_size)

    confidence_scores, bboxes = [], []

    if peopleOnly:
        for prob, bbox in zip(probs, bboxes_scaled.tolist()):
            cl = prob.argmax()
            if __CLASSES[cl] != 'person':
                continue
            confidence_scores.append(prob[cl])
            bboxes.append(bbox)
    else:
        confidence_scores = probs
        bboxes = bboxes_scaled

    # return each bbox with its confidence score
    return torch.tensor(confidence_scores, dtype=torch.float32), torch.tensor(bboxes, dtype=torch.float32)


def __detect_video(video_dir_path, conf_treshold=0.5, peopleOnly=True):
    seq_path = os.path.join(video_dir_path + '/', 'img1')
    fnames = os.listdir(seq_path)
    for frame_id, fname in alive_it(
            zip(range(1, len(fnames) + 1), fnames),
            total=len(fnames),
            title='Detecting video frames...'):
        img = Image.open(os.path.join(seq_path + '/', fname))
        conf_scores, bboxes = __detect(__model, img, confidence_threshold=conf_treshold, peopleOnly=peopleOnly)
        __cache_detections(video_dir_path, frame_id, conf_scores, bboxes)
        yield conf_scores, bboxes


def __load_detections(video_dir_path):
    seq_path = os.path.join(video_dir_path + '/', 'img1')
    frame_count = len(os.listdir(seq_path))

    cache_file_path = os.path.join(video_dir_path, 'detections.txt')
    assert os.path.exists(cache_file_path)
    with (open(cache_file_path, 'r') as f):
        lines = f.readlines()
        # check if cached detection is incomplete
        last_frame = int(lines[-1].strip().split(',')[0])
        if last_frame < frame_count:
            warnings.warn(f'Video frame count is {frame_count}. The detected cache file '
                          f'only contains detections for {last_frame} frames.')
        current_frame = 1
        current_conf, current_bboxes = [], []
        with alive_bar(total=len(lines), title='Reading cache file...') as bar:
            for line in lines:
                frame_id, x_1, y_1, x_2, y_2, conf_score = line.strip().split(',')
                # check if we finished reading all the detections for this frame
                if int(frame_id) != current_frame:
                    conf_scores, bboxes = (
                        torch.tensor(current_conf, dtype=torch.float32),
                        torch.tensor(current_bboxes, dtype=torch.float32)
                    )
                    current_conf = []
                    current_bboxes = []
                    current_frame += 1
                    # update progress bar
                    yield conf_scores, bboxes
                else:
                    current_conf += [float(conf_score)]
                    current_bboxes += [[float(x_1), float(y_1), float(x_2), float(y_2)]]
                bar()


def __cache_detections(video_dir_path, frame_id, conf_scores, bboxes):
    fpath = os.path.join(video_dir_path, 'detections.txt')
    with open(fpath, mode='a') as f:
        for conf, (x_1, y_1, x_2, y_2) in zip(conf_scores, bboxes.tolist()):
            f.write(f'{frame_id},{x_1},{y_1},{x_2},{y_2},{conf}\n')


def get_detections(video_name):
    """
    Outputs the detections for each frame of the specified video, along with their confidence score.
    Be aware that stopping this function from running until its end will create an incomplete detection file.

    :param video_name: the name of the video to be detected
    :return: a generator yielding two torch tensors `(conf_scores, bboxes)` for each frame.
        Note that `conf_scores` is a 1D (N, ) tensor, while `bboxes` is a 2D (N, 4) tensor, where N is the
        number of frames in the video.
        Also note that each bbox is expressed as `[x_1, y_1, x_2, y_2]`.
    """
    video_dir_path = get_dir_path(video_name)

    # check if detections cache file exists
    if os.path.exists(os.path.join(video_dir_path, 'detections.txt')):
        # if so, load detections from cache
        logging.info(f'Using detections cache file for {video_name}.')
        return __load_detections(video_dir_path)
    else:
        logging.info(f'No detections cache file found for {video_name}. '
                     f'Detector will now process the video.')
        return __detect_video(video_dir_path)


if __name__ == '__main__':
    path = "MOT17/train/MOT17-02-DPM/img1"
    model = torch.hub.load(
        'facebookresearch/detr:main',
        'detr_resnet50',
        pretrained=True,
    )
    show_video_detections(path)
