import numpy as np
import random as rnd


class Tracker:
    def __init__(self, bbox, label, img):
        """
        Initialize a tracker object
        :param bbox: np.ndarry containing bounding box coordinates [x1, y1, x2, y2]
        :param label: a string label of the object
        :param img: the image to be tracked
        """
        self.counter_updates = 0
        self.counter_last_update = 0
        self.id = label
        self.bbox = bbox
        self.bbox_difference = np.array([0, 0, 0, 0], dtype=np.float32)
        self.descriptor = self.compute_descriptor(img, bbox)
        self.color = rnd.choice(self.__COLORS)

    @classmethod
    @property
    def __COLORS(cls):
        # colors for visualization
        return [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    def predict(self):
        """
        This method returns the predicted bounding box coordinates and increase the counter_last_updates variable
        :return: bbox
        """
        self.bbox += self.bbox_difference
        self.counter_last_update += 1
        return self.bbox

    def update(self, bbox, img):
        """
        Update the tracked bounding box coordinates, increase the counter_updates variable and
         inizialize the counter_last_update variable to zero
        """
        self.bbox_difference = bbox - self.bbox
        self.bbox = bbox
        self.descriptor = self.compute_descriptor(img, bbox)
        self.counter_updates += 1
        self.counter_last_update = 0

    @staticmethod
    def compute_descriptor(img, bbox):
        """
        Compute the descriptor of the object by a simple mean of color
        """
        x_1, y_1, x_2, y_2 = bbox.astype(int)
        cropped_image = img[y_1:y_2, x_1:x_2]
        return np.mean(cropped_image, axis=(0, 1))

    @property
    def current_position(self):
        """
        Return the current position of the object
        :return: bbox
        """
        return self.bbox
