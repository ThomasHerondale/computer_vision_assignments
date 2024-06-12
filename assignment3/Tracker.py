import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
import random as rnd
from matching import _convert_bbox


class Tracker:
    def __init__(self, bbox, conf_score, label, img):
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
        self.conf_score = conf_score
        self.bbox_difference = np.array([0, 0, 0, 0], dtype=np.float32)
        self.descriptor = self.compute_descriptor_hog(img, bbox)
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

    def update(self, bbox, conf_score, img):
        """
        Update the tracked bounding box coordinates, increase the counter_updates variable and
         inizialize the counter_last_update variable to zero
        """
        self.bbox_difference = bbox - self.bbox
        self.bbox = bbox
        self.conf_score = conf_score
        self.descriptor = self.compute_descriptor_hog(img, bbox)
        self.counter_updates += 1
        self.counter_last_update = 0

    @staticmethod
    def compute_descriptor_hog(frame, bbox):
        """
        Compute the HOG descriptor of the object.
        :param frame: current frame of image
        :param bbox_converted: a simple np array in form [x_l, y_t, w, h]
        :return features: features of cropped image that include pedestrian
        """
        bbox_converted = _convert_bbox(bbox=bbox)
        x_l, y_t, w, h = bbox_converted  #estraggo gli elementi di una bbox
        cropped_image = frame[y_t:(y_t + h), x_l:(x_l + w)]  # ritaglio l'imagine
        resize = cv2.resize(cropped_image, dsize=(64, 128))
        # Calcola l'HOG dell'immagine ritagliata
        features, hog_image = hog(resize, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True, channel_axis=2)

        # Normalizzo l'HOG
        features = exposure.rescale_intensity(features, in_range=(0, 10))

        return features

    @property
    def current_position(self):
        """
        Return the current position of the object
        :return: bbox
        """
        return self.bbox
