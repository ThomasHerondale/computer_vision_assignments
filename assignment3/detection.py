import logging
import os
import warnings
from typing import Tuple, Optional, Dict, Any

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


def __rescale_bbox(bbox: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
    """
    Converts the bbox specified by its center coordinates in [0, 1] and its size to
    a bbox expressed by the coordinates of its vertices rescaled with respect to the image size.
    :param bbox: the bounding box as `[c_x, c_y, w, h]`, where c_x, c_y are in [0, 1]
    :param img_size: the size of the image
    :return: the bounding box as `[x_1, y_1, x_2, y_2]` in image scale
    """
    img_w, img_h = img_size
    bbox = bbox * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return bbox


def __detect(model, img, transform=None, confidence_threshold=0.5, people_only=True) -> (torch.Tensor, torch.Tensor):
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

    if people_only:
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


def __detect_video(video_dir_path: str, people_only: bool, progress_bar_prefix, conf_threshold):
    seq_path = os.path.join(video_dir_path + '/', 'img1')
    fnames = os.listdir(seq_path)
    cache_fpath = os.path.join(video_dir_path, __get_next_cache_fname(video_dir_path))
    __write_cache_header(cache_fpath, {'conf_threshold': conf_threshold})
    for frame_id, fname in alive_it(
            zip(range(1, len(fnames) + 1), fnames),
            total=len(fnames),
            title=f'{progress_bar_prefix} Detecting video frames...'):
        img = Image.open(os.path.join(seq_path + '/', fname))
        conf_scores, bboxes = __detect(__model, img, confidence_threshold=conf_threshold, people_only=people_only)
        if people_only:
            __cache_detections(
                cache_fpath,
                frame_id,
                conf_scores,
                bboxes,
            )
        yield conf_scores, bboxes


def __load_detections(video_dir_path, cache_fname, progress_bar_prefix):
    seq_path = os.path.join(video_dir_path + '/', 'img1')
    frame_count = len(os.listdir(seq_path))

    cache_file_path = os.path.join(video_dir_path, cache_fname)
    assert os.path.exists(cache_file_path)
    with (open(cache_file_path, 'r') as f):
        # skip first line containing hyperparameters
        lines = f.readlines()[1:]
        # check if cached detection is incomplete
        last_frame = int(lines[-1].strip().split(',')[0])
        if last_frame < frame_count:
            warnings.warn(f'Video frame count is {frame_count}. The detected cache file '
                          f'only contains detections for {last_frame} frames.')
        current_frame = 1
        current_conf, current_bboxes = [], []
        with alive_bar(total=len(lines), title=f'{progress_bar_prefix} Reading detections cache file...') as bar:
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


def __write_cache_header(cache_fpath, hyperparameters):
    with open(cache_fpath, mode='w') as f:
        # write header with hyperparameter values
        param_str = ''
        for k, v in hyperparameters.items():
            param_str = f'{k}={str(v)},'
        f.write(param_str + '\n')


def __cache_detections(cache_fpath, frame_id, conf_scores, bboxes):
    with open(cache_fpath, mode='a') as f:
        for conf, (x_1, y_1, x_2, y_2) in zip(conf_scores, bboxes.tolist()):
            f.write(f'{frame_id},{x_1},{y_1},{x_2},{y_2},{conf}\n')


def __check_cache_hyperparameter_match(video_dir_path: str, hyperparameters: Dict[str, Any]) -> Optional[str]:
    def match(params, hyperparameters):
        for param in params:
            if param:
                k, v = param.split('=')
                # if parameter doesn't match, skip to the next file
                if hyperparameters[k] != v:
                    return False
        return True

    for cache_fname in __get_cache_fnames(video_dir_path):
        with (open(os.path.join(video_dir_path, cache_fname), 'r') as f):
            # first line contains parameter list
            line = f.readline()
            # get all (param_name, value) pairs
            params = line.strip().split(',')
            if match(params, hyperparameters):
                return cache_fname
    return None


def __get_next_cache_fname(video_dir_path: str) -> str:
    max_n = -1
    for fname in __get_cache_fnames(video_dir_path):
        _, s = fname.split('_')
        n, _ = s.split('.')
        if int(n) > max_n:
            max_n = int(n)

    idx = max_n + 1 if max_n != -1 else '1'
    return f'detections_{idx}.txt'


def __get_cache_fnames(video_dir_path):
    return [fname for fname in os.listdir(video_dir_path) if fname.startswith('detections_')]


def get_detections(
        video_name: str,
        people_only: bool,
        progress_bar_prefix: str = '',
        conf_threshold: float = 0.5):
    """
    Outputs the detections for each frame of the specified video, along with their confidence score.
    Be aware that stopping this function from running until its end might create an incomplete detection cache file.

    :param video_name: the name of the video to be detected
    :param people_only: whether to instruct the tracker to detect only people in the video or not. Note that
     this parameter will be ignored if a cache file exists, since caching is only allowed for people detections
    :param progress_bar_prefix: a prefix to display before the title of the progress bar
    :param conf_threshold: the confidence threshold that will be used to suppress detections for which
     the detector is too unsure. Note that this parameter will be ignored if a cache file exists
    :return: a generator yielding two torch tensors `(conf_scores, bboxes)` for each frame.
     Note that `conf_scores` is a 1D (N, ) tensor, while `bboxes` is a 2D (N, 4) tensor, where N is the
     number of frames in the video.
     Also note that each bbox is expressed as `[x_1, y_1, x_2, y_2]`.
    """
    video_dir_path = get_dir_path(video_name)

    # check for existing cache file with this hyperparameters combination
    hyperparameters = {'conf_threshold': str(conf_threshold)}
    cache_fname = __check_cache_hyperparameter_match(
        video_dir_path,
        hyperparameters
    )
    if cache_fname:
        logging.info(f'Using detections cache file {cache_fname} for {video_name}.')
        if not people_only:
            warnings.warn("Loading of non-people detections from cache is not supported. "
                          "Parameter will be ignored")
        return __load_detections(
            video_dir_path,
            cache_fname,
            progress_bar_prefix=progress_bar_prefix,
        )
    else:
        logging.info(f'No detections cache file found for {video_name} and this hyperparameter combination '
                     f'{hyperparameters}. Detector will now process the video.')
        return __detect_video(
            video_dir_path,
            people_only=people_only,
            progress_bar_prefix=progress_bar_prefix,
            conf_threshold=conf_threshold
        )


if __name__ == '__main__':
    for _ in get_detections('MOT17-05-SDP', people_only=True, conf_threshold=0.5):
        pass