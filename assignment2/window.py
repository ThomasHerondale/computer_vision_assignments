import os
import time
import cv2
import numpy as np
from sklearn.pipeline import Pipeline
from validation import tune_hyperparameters
from new import load_dataset
"""
    implement a suitable multi-scale sliding window procedure, use your classifier to detect pedestrians in the test 
    images at 3 different scales. Include a stride parameters to speed up
    computation. You can have an idea about the scales to use by analyzing the bounding boxes in the training set.
"""


def sliding_window(image, window_size=(64, 128), stide=(10, 10)) -> (int, int, np.array):
    """
        Il generatore sliding window consente alla finestra di scorrimento di scorrere localmente sull'immagine
        :param image: immagine da analizzare
        :param window_size: dimensione della finestra scorevole
        :param stide: il passo lungo le x e lungo le y
    """
    for j in range(0, image.shape[0], stide[1]):
        for i in range(0, image.shape[1], stide[0]):
            yield i, j, image[j:j + window_size[1], i:i + window_size[0]]


def resize_img(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    h = image.shape[0]
    w = image.shape[1]
    if scale > 1.0:
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        return cv2.resize(image, (int(w * scale), int(h * scale)))


# implementare la funzione che ridimensiona l'immagine in base al fattore di scala.
# prima di effettuare il ridimensionamento devo eseguire il giltro di gauss
# immagine ridimensionata la invoco
def gaussian_pyramid(image: np.ndarray, scale: float = 1.0, sigma: int = 5,
                     kernel_size: (int, int) = (3, 3)) -> np.ndarray:

    if scale < 1.0:
        filtered_image = cv2.GaussianBlur(image, ksize=kernel_size, sigmaX=sigma, sigmaY=sigma)
        resized_image = resize_img(filtered_image, scale)
        return resized_image
    elif scale > 1.0:
        return resize_img(image, scale=scale)
    else:
        return image


def show_window(image: np.ndarray, hog: cv2.HOGDescriptor,
                clf: Pipeline, scale: float = 1.0, window_size: (int, int) = (64, 128)):
    save: [] = []
    image_scaled = gaussian_pyramid(image, scale=scale)
    for (x, y, window) in sliding_window(image_scaled, window_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            continue
        clone_image = image_scaled.copy()
        cv2.rectangle(clone_image, (x, y), (x + window_size[0], y + window_size[1]), (255, 0, 0), 2)
        # Calcolo tutti gli istogrammi di orientazione del gradiente
        descriptor = hog.compute(window)
        # Passo al classificatore
        f = clf.predict(descriptor)
        cv2.imshow("Sliding Window", clone_image)
        cv2.waitKey(1)
        time.sleep(0.20)

def multiscale_function(images: [np.ndarray]):
    x, y = load_dataset(size=500)
    clf, _, _ = tune_hyperparameters(x, y)
    hog = cv2.HOGDescriptor()
    scales = (1.2, 1.0, 0.5)
    for image in images:
        for scale in scales:
            show_window(image, hog, clf, scale)


def read(cartella: str, numero_immagini=500):
    immagini = []
    contatore_immagini = 0

    # Verifica se il percorso della cartella esiste
    if not os.path.isdir(cartella):
        print("La cartella specificata non esiste.")
        return immagini

    # Scandisce i file nella cartella
    for filename in os.listdir(cartella):
        # Verifica se il file è un'immagine JPG
        if filename.lower().endswith(('.jpg')):
            # Costruisce il percorso completo del file
            percorso_immagine = os.path.join(cartella, filename)
            try:
                # Carica l'immagine utilizzando OpenCV
                img = cv2.imread(percorso_immagine)
                # Verifica se l'immagine è stata caricata correttamente
                if img is not None:
                    immagini.append(img)
                    contatore_immagini += 1
            except Exception as e:
                print(f"Errore durante il caricamento dell'immagine {percorso_immagine}: {e}")

            # Verifica se è stato caricato il numero desiderato di immagini
            if contatore_immagini >= numero_immagini:
                break
    return immagini


image_path = "/Users/fede/Desktop/Unipa/anno4/Visione artificiale/Assignment/2/WiderPerson/Images"
immagini = read(image_path)
multiscale_function(immagini)
