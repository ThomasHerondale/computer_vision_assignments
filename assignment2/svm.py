from typing import Sequence

import cv2
import numpy as np
from sklearn.svm import LinearSVC


def svm_classifier(x: list[Sequence[float]], y: list[int], c=0.1):
    """
        Funzione per l'addestramento di Linear SVM
        :param x: una array di Hog Features
        :param y: l'etichetta associata -1 per i negativi e 1 per i positivi
        :param c: Iperparametro di regolazzizazione.
        :return: ritorna il classificatore
    """
    clf = LinearSVC(C=c, dual=True)
    clf.fit(x, y)
    clf.score(x, y)
    return clf
