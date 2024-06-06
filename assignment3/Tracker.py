import numpy as np

class Tracker:
    def __init__(self, bbox, label, img):
        self.counter_updates = 0
        self.counter_last_update = 0
        self.id = label
        self.bbox = bbox
        self.descriptor = self.compute_descriptor(img, bbox)


    def predict(self):
        self.bbox += self.bbox_difference
        self.counter_last_update += 1
        return self.bbox

    def update(self, bbox, img):
        """
        Aggiorna le informazioni del Tracker
        """
        self.bbox = bbox
        self.descriptor = self.compute_descriptor(img, bbox)
        self.counter_updates += 1
        self.counter_last_update = 0

    @staticmethod
    def compute_descriptor(img, bbox):
        """
        Calcola il colore medio della bounding box e lo usa come descrittore di apperance
        """
        x_1, y_1, x_2, y_2 = bbox.astype(int)
        cropped_image = img[y_1:y_2, x_1:x_2]
        return np.mean(cropped_image, axis=(0, 1))

    @property
    def current_position(self):
        return self.bbox