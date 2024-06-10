import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from Tracking_Algorithm import TrackingAlgorithm
from assignment3.utils import get_dir_path
from detection import get_detections, CLASSES

# colors for visualization
__COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def show_video_tracking(video_name: str, waitKeys: bool = False, bufferize: bool = True):
    """
    Shows the tracking algorithm predicted trajectories for the specified video.
    :param video_name: the name of the video
    :param waitKeys: whether the algorithm has to wait for a key-press to go to the next frame or not
    :param bufferize: whether to track all frames before showing results or to track each frame
            on the fly before showing it. This will be ignored if results are already cached.
    """
    video_dir_path = get_dir_path(video_name)
    seq_path = os.path.join(video_dir_path + '/', 'img1')
    fnames = os.listdir(seq_path)

    # work generator until end of the video
    if bufferize:
        _ = [_ for _ in get_detections(video_name, people_only=True)]

    tracker = TrackingAlgorithm()  # Inizializza il tracker

    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', lambda event: exit())
    for img_path, (conf_scores, bboxes_scaled) in zip(fnames, get_detections(video_name, people_only=True)):
        img = Image.open(os.path.join(seq_path, img_path))

        # Converto il tensore in un array numpy per comodità
        detections = bboxes_scaled.numpy()

        tracked_people = tracker.update(detections, np.array(img))

        # Visualizza i risultati del tracking
        __plot_results(img, waitKeys, tracked_people=tracked_people)


def show_video_detections(video_name: str,
                          waitKeys: bool = False,
                          bufferize: bool = True,
                          people_only: bool = False):
    """
    Shows the detection algorithm predictions for the specified video.
    :param video_name: the name of the video
    :param waitKeys: whether the algorithm has to wait for a key-press to go to the next frame or not
    :param bufferize: whether to detect all frames before showing results or to detect each frame
     on the fly before showing it. This will be ignored if results are already cached.
    :param people_only: whether to instruct the tracker to detect only people in the video or not
    """
    video_dir_path = get_dir_path(video_name)
    seq_path = os.path.join(video_dir_path + '/', 'img1')
    fnames = os.listdir(seq_path)

    # work generator until end of the video
    if bufferize:
        _ = [_ for _ in get_detections(video_name, people_only=people_only)]

    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', lambda event: exit())
    for (conf, bboxes), fname in zip(get_detections(video_name, people_only=people_only), fnames):
        img = Image.open(os.path.join(seq_path, fname))
        __plot_results(img, waitKeys, prob=conf, bboxes=bboxes)


def __convert_bbox(bbox) -> torch.Tensor:
    """
    Converts the bounding box specified by its center coordinates and its size to
    a bbox expressed by the coordinates of its vertices.
    :param bbox: bounding box as `[c_x, c_y, w, h]`
    :return: the bounding box as `[x_1, y_1, x_2, y_2]`
    """
    x_c, y_c, w, h = bbox.unbind(dim=-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def __plot_results(pil_img, waitKeys, prob=None, bboxes: torch.Tensor = None, tracked_people=None):
    if (prob is not None and bboxes is None) or (prob is None and bboxes is not None):
        raise ValueError("Only one among confidence scores and bounding boxes was provided. "
                         "Please provide both or use another argument combination.")
    elif bboxes is None and prob is None:
        if tracked_people is None:
            raise ValueError("No data was provided for visualization.")
        else:
            mode = 'trackers'
    else:
        # if conf_score is 1-dimensional, there's a single confidence score per bounding box
        # so only people were detected
        if len(prob.shape) == 1:
            mode = 'detections_people_only'
        else:
            mode = 'detections'

    plt.clf()
    plt.imshow(pil_img)
    ax = plt.gca()

    if mode == 'trackers':
        it = zip(tracked_people, __COLORS * 100)
    else:
        assert bboxes is not None
        it = zip(prob, enumerate(bboxes), __COLORS * 100)

    for e in it:
        if mode == 'trackers':
            tracked_person, c = e
            # vi odio per avere usato gli ndarray dentro il tracker
            # devo convertire e riconvertire le cose ottocento volte T.T
            bbox, bbox_id = torch.tensor(tracked_person[:4]), tracked_person[-1]
            text = f'ID: {bbox_id:.0f}'
        elif mode == 'detections':
            p, bbox, c = e
            cl = p.argmax()
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        elif mode == 'detections_people_only':
            # _ is needed to ignore indices of enumerate()
            p, (_, bbox), c = e
            text = f'{p:0.2f}'
        else:
            raise ValueError("Could not detect display mode based on arguments provided.")

        (x_1, y_1, x_2, y_2) = __convert_bbox(bbox)
        ax.add_patch(plt.Rectangle((x_1, y_1), x_2 - x_1, y_2 - y_1,
                                   fill=False, color=c, linewidth=2))
        ax.text(x_1, y_1, text, fontsize=8, color="white").set_bbox(dict(facecolor=c, linewidth=0, alpha=0.8))
    plt.axis('off')
    plt.pause(0.1)
    if waitKeys:
        plt.waitforbuttonpress()


if __name__ == '__main__':
    video_name = 'MOT17-10-SDP'
    show_video_tracking(video_name, waitKeys=True)
    #show_video_detections(video_name, waitKeys=True, people_only=True)
