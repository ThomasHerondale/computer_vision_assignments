from detection import get_detections, __get_dir_path
from Tracking_Algorithm import TrackingAlgorithm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


# colors for visualization
__COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


# waitKeys = true -> il video non va avanti di frame se non premi un pulsante
def show_video_detections(video_name, waitKeys=False):
    video_dir_path = __get_dir_path(video_name)
    seq_path = os.path.join(video_dir_path + '/', 'img1')
    fnames = os.listdir(seq_path)

    tracker = TrackingAlgorithm()  # Inizializza il tracker

    plt.figure(figsize=(16, 10))
    for frame_id, img_path, (conf_scores, bboxes_scaled) in (
            zip(range(1, len(fnames) + 1), fnames, get_detections(video_name))):
        plt.clf()
        img = Image.open(os.path.join(seq_path, img_path))

        # Converto il tensore in un array numpy per comodità
        detections = bboxes_scaled.numpy()

        tracked_people = tracker.update(detections, np.array(img))

        # Visualizza i risultati del tracking
        __plot_results(img, tracked_people, waitKeys=waitKeys)


def __plot_results(pil_img, tracked_people, waitKeys):
    plt.imshow(pil_img)
    ax = plt.gca()
    for (x_1, y_1, x_2, y_2, id), c in zip(tracked_people, __COLORS * 100):
        text = f'ID: {int(id)}'
        ax.add_patch(plt.Rectangle((x_1, y_1), x_2 - x_1, y_2 - y_1,
                                   fill=False, color=c, linewidth=1))
        ax.text(x_1, y_1, text, fontsize=8, color="white")
    plt.axis('off')
    plt.pause(0.1)
    if waitKeys:
        plt.waitforbuttonpress()


if __name__ == '__main__':
    video_name = 'MOT17-05-SDP'
    show_video_detections(video_name)
