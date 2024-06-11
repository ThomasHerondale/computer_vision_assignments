import json
import os
import time
from itertools import product
from typing import List, Any, Union

import PIL.Image
import numpy as np
from alive_progress import alive_bar
from tabulate import tabulate

from Tracking_Algorithm import TrackingAlgorithm
from detection import get_detections
from report import save_results, compute_report, clear_video_results
from utils import get_dir_path


class TrackerTuner:
    def __init__(self, multi_metric_criteria='mean', results_fnames: Union[List[str], str] = None):
        if isinstance(results_fnames, str):
            results_fnames = [results_fnames]
        if results_fnames is None:
            results_fnames = []

        self.__param_grid = {}
        self.__current_detector = None
        self.__curent_tracker = None
        self.__criteria = multi_metric_criteria

        self.best_aggregated_score = None
        self.best_scores = None
        self.best_params = None

        self.__tuning_results = []
        for results_fname in results_fnames:
            with open(results_fname, 'r') as f:
                self.__tuning_results += json.load(f)

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

    def tune(self, videos: Union[List[str], str], print_scores: List[str] = None):
        if not print_scores:
            print_scores = ['HOTA', 'MOTA']
        if isinstance(videos, str):
            videos = [videos]
        if not videos:
            raise ValueError("No videos were provided for hyperparameter tuning.")

        video_counter = 1

        combinations_count = len(self.param_combinations())

        for (video_ctr, video), (comb_ctr, comb) in product(enumerate(videos),
                                                            enumerate(self.param_combinations())):
            # reset parameter combinations counter
            combination_counter = 1
            # make counters start from 1
            comb_ctr += 1
            video_ctr += 1

            # get frame list
            video_dir_path = get_dir_path(video)
            seq_path = os.path.join(video_dir_path + '/', 'img1')
            fnames = os.listdir(seq_path)

            # keep track of detection time
            start = time.perf_counter()

            # setup hyperparameters for detector and tracker
            self._setup(video_counter, comb)

            # work detections generator until its end
            detections = [d for d in
                          get_detections(video,
                                         people_only=True,
                                         progress_bar_prefix=f'[{video_ctr}, {comb_ctr}/{combinations_count}]')]

            detection_time = time.perf_counter() - start
            # keep track of tracking time
            start = time.perf_counter()

            # clear results files
            clear_video_results(video)

            with alive_bar(
                    total=len(detections),
                    title=f'[{video_ctr}, {comb_ctr}/{combinations_count}] '
                          f'Tracking video frames...        '       # spaces to gracefully align bars
            ) as bar:
                for frame_id, fname, (conf_scores, bboxes) in zip(range(1, len(fnames) + 1), fnames, detections):
                    img_path = os.path.join(seq_path, fname)
                    with PIL.Image.open(img_path) as img:
                        trackers = self.__current_tracker.update(bboxes.numpy(), conf_scores, np.array(img))
                        save_results(video, frame_id, trackers)
                    bar()

            tracking_time = time.perf_counter() - start

            scores = compute_report(video)
            # get scores to print
            scores_to_print = {s: v for s, v in scores.items() if s in print_scores}
            # save scores
            self.__tuning_results += [(video, detection_time, tracking_time, comb, scores)]

            print(f'\t[{video_ctr}, {comb_ctr}/{combinations_count}] Hyperparameters: {comb}\n'
                  f'\t[{video_ctr}, {comb_ctr}/{combinations_count}] Scores: {scores_to_print}\n'
                  f'\t[{video_ctr}, {comb_ctr}/{combinations_count}] '
                  f'Elapsed time: {detection_time + tracking_time:.2f}s')

            combination_counter += 1
            video_counter += 1

        self._cleanup()

    @property
    def results(self):
        return [*self.__tuning_results]

    def display_results(self):
        headers = ['#',
                   'VIDEO',
                   'DETECTION TIME',
                   'TRACKING TIME',
                   'HOTA',
                   'MOTA']
        results = [
            (f'({idx})', t[0], f'{t[1]:.2f}s', f'{t[2]:.2f}s', f'{t[4]["HOTA"]:.4f}', f'{t[4]["MOTA"]:.4f}')
            for idx, t in enumerate(self.__tuning_results)
        ]
        print(tabulate(results, headers=headers))

    def __save_results(self, fname, overWrite: bool = False):
        mode = 'w' if overWrite else 'a'
        with open(fname, mode) as f:
            json.dump(self.results, f, indent=4)

    def _setup(self, video_number, param_comb):
        # setup detector parameters
        detector_params = self.__get_params_by_target(param_comb, 'detector')
        self.__current_detector = lambda video_name: get_detections(
            video_name, people_only=True, progress_bar_prefix=f'[{video_number}]', **detector_params
        )

        # setup tracking parameters
        tracking_alg_params = self.__get_params_by_target(param_comb, 'tracker')
        self.__current_tracker = TrackingAlgorithm(**tracking_alg_params)

    def _cleanup(self):
        # invalidate last detector and tracker
        self.__current_detector = None
        self.__curent_tracker = None
        self.__save_results('tuning_results.json', overWrite=True)

    @staticmethod
    def __get_params_by_target(param_comb, target):
        return {k.split('__')[1]: v for (k, v) in param_comb.items() if k.startswith(target)}


if __name__ == '__main__':
    tuner = TrackerTuner(results_fnames='tuning_results.json')
    tuner.register_hyperparameter('tracker__spatial_metric_threshold', [90, 120, 140])
    tuner.register_hyperparameter('tracker__max_age', [3, 5, 10])
    tuner.register_hyperparameter('tracker__initialize_age', [3, 5, 10])
    video = 'MOT17-05-SDP'
    #tuner.tune([video])
    tuner.display_results()