import os
import time
import random
from itertools import product
from typing import List, Any

import PIL.Image
import numpy as np

from Tracking_Algorithm import TrackingAlgorithm
from detection import get_detections
from utils import get_dir_path
from alive_progress import alive_bar


class TrackerTuner:
    def __init__(self, multi_metric_criteria='mean'):
        self.__param_grid = {}
        self.__current_detector = None
        self.__curent_tracker = None
        self.__criteria = multi_metric_criteria

        self.best_aggregated_score = None
        self.best_scores = None
        self.best_params = None

        self.__tuning_results = []

    def register_hyperparameter(self,
                                name: str,
                                values: List[Any],
                                overwrite: bool = False):
        # check parameter name format
        strs = name.split('__')
        if len(strs) != 2:
            raise ValueError(f"Parameter {name} must match format 'prefix__name' to correctly determine"
                             f"its location.")

        if name in self.__param_grid:
            if overwrite:
                self.__param_grid[name] = values
            else:
                self.__param_grid[name].extend(values)
        else:
            self.__param_grid[name] = values

    def param_combinations(self):
        keys, values = zip(*self.__param_grid.items())
        return [dict(zip(keys, vals)) for vals in product(*values)]

    def tune(self, videos: List[str]):
        if not videos:
            raise ValueError("No videos were provided for hyperparameter tuning.")

        video_counter = 1

        combinations_count = len(self.param_combinations())

        for video in videos:
            # reset parameter combinations counter
            combination_counter = 1
            print(f'Beginning detection of video {video_counter}/{len(videos)}')

            # get frame list
            video_dir_path = get_dir_path(video)
            seq_path = os.path.join(video_dir_path + '/', 'img1')
            fnames = os.listdir(seq_path)

            for comb in self.param_combinations():
                # keep track of detection time
                start = time.perf_counter()

                # setup hyperparameters for detector and tracker
                self._setup(video_counter, comb)

                # work detections generator until its end
                detections = [d for d in
                              get_detections(video,
                                             people_only=True,
                                             progress_bar_prefix=f'[{video_counter}, '
                                                                 f'{combination_counter}/{combinations_count}]'
                                             )]

                detection_time = time.perf_counter() - start
                # keep track of tracking time
                start = time.perf_counter()

                with alive_bar(
                        total=len(detections),
                        title=f'[{video_counter}, {combination_counter}/{combinations_count}] '
                              f'Tracking video frames...        '       # spaces to gracefully align bars
                ) as bar:
                    for fname, (conf_scores, bboxes) in zip(fnames, detections):
                        img_path = os.path.join(seq_path, fname)
                        with PIL.Image.open(img_path) as img:
                            res = self.__current_tracker.update(bboxes.numpy(), conf_scores, np.array(img))
                        bar()

                tracking_time = time.perf_counter() - start

                scores = FOR_TEST_get_scores(res)
                score = self._aggregate_scores(scores)
                self._save_scores(comb, detection_time, tracking_time, scores)

                print(f'\t[{video_counter}, {combination_counter}/{combinations_count}] Hyperparameters: {comb}\n'
                      f'\t[{video_counter}, {combination_counter}/{combinations_count}] Score: {score:.4f}\n'
                      f'\t[{video_counter}, {combination_counter}/{combinations_count}] '
                      f'Elapsed time: {detection_time + tracking_time:.2f}s')

                combination_counter += 1

            video_counter += 1

        self._cleanup()

    @property
    def results(self):
        return [*self.__tuning_results]

    def _setup(self, video_number, param_comb):
        # setup detector parameters
        detector_params = self.__get_params_by_target(param_comb, 'detector')
        self.__current_detector = lambda video_name: get_detections(
            video_name, people_only=True, progress_bar_prefix=f'[{video_number}]', **detector_params
        )

        # setup tracking parameters
        tracking_alg_params = self.__get_params_by_target(param_comb, 'tracking')
        self.__current_tracker = TrackingAlgorithm(**tracking_alg_params)

    def _cleanup(self):
        # invalidate last detector and tracker
        self.__current_detector = None
        self.__curent_tracker = None

        # sort results by aggregate score
        self.__tuning_results.sort(key=lambda res: self._aggregate_scores(res[3]), reverse=True)

    def _save_scores(self, params, detection_time, tracking_time, scores):
        # aggregate scores and check if they're currently the best we have
        aggregated_score = self._aggregate_scores(scores)
        if self.best_aggregated_score is None or aggregated_score > self.best_aggregated_score:
            self.best_aggregated_score = aggregated_score
            self.best_scores = scores
            self.best_params = params

        self.__tuning_results += [(detection_time, tracking_time, params, scores)]

    def _aggregate_scores(self, scores):
        if self.__criteria == 'mean':
            return np.mean(scores)
        elif self.__criteria == 'max':
            return np.max(scores)

    @staticmethod
    def __get_params_by_target(param_comb, target):
        return {k.split('__')[1]: v for (k, v) in param_comb.items() if k.startswith(target)}


def FOR_TEST_get_scores(res):
    return random.random() * random.randint(1, 1500), random.random() * random.randint(1, 1500)


if __name__ == '__main__':
    tuner = TrackerTuner()
    tuner.register_hyperparameter('detector__conf_threshold', [2, 5])
    tuner.register_hyperparameter('detector__iou_threshold', [3, 9])
    tuner.tune(['MOT17-05-SDP', 'MOT17-09-DPM', 'MOT17-04-DPM'])
    for params in tuner.results:
        print(params)
    print(tuner.best_aggregated_score)
    print(tuner.best_scores)
    print(tuner.best_params)
