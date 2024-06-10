import numpy as np
from utils import compare_boxes
from Tracker import Tracker
from matching import matching


class TrackingAlgorithm:
    """Implementation of the tracking algorithm."""
    def __init__(self,
                 max_age=5,
                 initialize_age=5,
                 spatial_metric='euclidean',
                 spatial_metric_threshold=30,
                 appearance_metric=None,
                 **kwargs
                 ):
        """
        Initialize the tracking algorithm.
        :param max_age: Maximum number of frames for which a tracker can survive without being detected
        :param initialize_age: Number of frames after which a tracker can actually be considered a new target
        :param spatial_metric:
        :param spatial_metric_threshold:
        :param appearance_metric:
        :param kwargs:
        """
        self.max_age = max_age
        self.initialize_age = initialize_age
        self.trackers = []
        self.count = 0
        self.metric_dist_threshold = spatial_metric_threshold
        self.spatial_metric = spatial_metric
        self.appearance_metric = appearance_metric

    def new_id(self) -> int:
        """
        Generate a new tracker id.
        """
        self.count += 1
        return self.count

    def update(self, detections, conf_scores, img) -> np.ndarray:
        """
        Update all the information about the new detections and the corrispondece with the trackers,
        the unmatched detections and the unmatched trackers.
        """
        # Creo una lista contenente i miei target ancora "in vita"
        self.trackers = [tracker for tracker in self.trackers if tracker.counter_last_update <= self.max_age]

        # Creo una lista contenente le bounding box dei miei target precedentemente selezionati
        bboxes_trackers = np.array([tracker.current_position for tracker in self.trackers])

        # Il metodo compare compare_boxes l'ho preso paro paro online, ma sarebbe quello che cipo deve implementare
        #matched, unmatched_detections, unmatched_trackers = compare_boxes(detections, bboxes_trackers)
        matched, unmatched_trackers, unmatched_detections = matching(
            detections,
            bboxes_trackers,
            threshold=self.metric_dist_threshold
        )

        for detection_num, tracker_num in matched:
            self.trackers[tracker_num].update(detections[detection_num], img)

        for i in unmatched_detections:
            self.trackers.append(Tracker(detections[i, :], self.new_id(), conf_scores[i], img))

        result = []
        for tracker in self.trackers:
            if tracker.counter_last_update == 0 and tracker.counter_updates >= self.initialize_age:
                result.append(np.concatenate((tracker.current_position, [tracker.id], tracker.color)))

        return np.array(result)
