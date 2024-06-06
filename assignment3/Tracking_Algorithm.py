
import numpy as np
from utils import compare_boxes
from Tracker import Tracker


class TrackingAlgorithm:
    def __init__(self, max_age=5, initialize_age=5):
        self.max_age = max_age
        self.initialize_age = initialize_age
        self.trackers = []
        self.count = 0

    def new_id(self):
        """
        Genera l'ID da associare al nuovo target
        """
        self.count += 1
        return self.count

    def update(self, detections, img):
        """
        Aggiorna le informazioni sui trackers
        """
        #Creo una lista contenente i miei target ancora "in vita"
        self.trackers = [tracker for tracker in self.trackers if tracker.counter_last_update <= self.max_age]

        #Creo una lista contenente le bounding box dei miei target precedentemente selezionati
        bboxes_trackers = np.array([tracker.current_position for tracker in self.trackers])

        #Il metodo compare compare_boxes l'ho preso paro paro online, ma sarebbe quello che cipo deve implementare praticamente
        matched, unmatched_detections, unmatched_trackers = compare_boxes(detections, bboxes_trackers)

        for detection_num, tracker_num in matched:
            self.trackers[tracker_num].update(detections[detection_num], img)

        for i in unmatched_detections:
            self.trackers.append(Tracker(detections[i, :], self.new_id(), img))

        result = []
        for tracker in self.trackers:
            if tracker.counter_last_update == 0 and tracker.counter_updates >= self.initialize_age:
                result.append(np.concatenate((tracker.current_position, [tracker.id])))

        return np.array(result)