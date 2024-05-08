import os
import time
import cv2
import numpy as np
from sklearn.pipeline import Pipeline
from validation import tune_hyperparameters
from new import load_dataset


def standardize_size(images: [np.ndarray], h: int = 700, w: int = 500) -> [(np.ndarray, float, float)]:
    standardized_images: [np.ndarray] = []
    if images is not None:
        for image in images:
            if image is not None:
                # Se l'immagine originaria è più grande rispetto alla dimensione (h, w) stabilite,
                # applico il filotro di Gauss
                image_h, image_w, _ = image.shape
                scale_w: float = w / image_w
                scale_h: float = h / image_h
                if image_h > h or image_w > w:
                    filtered_image = cv2.GaussianBlur(image, (3, 3), 5)
                    resized_image = cv2.resize(filtered_image, (w, h), interpolation=cv2.INTER_CUBIC)
                    standardized_images.append((resized_image, scale_h, scale_w))
                else:
                    # se l'immagine è piccola rispetto a (h, w) allora faccio semplicemente la resize
                    resized_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
                    standardized_images.append((resized_image, scale_h, scale_w))
            else:
                raise TypeError("Image cannot be None")
    else:
        raise TypeError("Images list cannot be None")
    return standardized_images


def sliding_window(image, window_size=(64, 128), stride: (int, int) = (20, 20)) -> (int, int, np.array):
    """
        Il generatore sliding window consente alla finestra di scorrimento di scorrere localmente sull'immagine
        :param image: immagine da analizzare
        :param window_size: dimensione della finestra scorevole
        :param stride: il passo lungo le x e lungo le y
    """
    for j in range(0, image.shape[0], stride[1]):
        for i in range(0, image.shape[1], stride[0]):
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


def show_window(image: np.ndarray,
                hog: cv2.HOGDescriptor,
                clf: Pipeline,
                scale: float = 1.0,
                window_size: (int, int) = (64, 128)
                ) -> [(int, int, int, int, np.ndarray)]:
    decisions: [()] = []
    image_scaled = gaussian_pyramid(image, scale=scale)
    for (x, y, window) in sliding_window(image_scaled, window_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            continue
        clone_image = image_scaled.copy()
        cv2.rectangle(clone_image, (x, y), (x + window_size[0], y + window_size[1]), (255, 0, 0), 2)
        # Calcolo tutti gli istogrammi di orientazione del gradiente
        descriptor = hog.compute(window)
        # Passo al classificatore
        f = clf.predict([descriptor])
        #print(f"Classificatore: {f}\n-----------------")
        if f[0] == 1:
            # memorizzo la tupla di cordinate della finestra (x, y, x2, y2)
            # Chiamo la funzione decision_function per determinare il parametro
            c = clf.decision_function([descriptor])
            #print(f"Decision: {c}\n-----------------------")
            #print(type(c))
            decisions.append((x, y, x + window_size[0], y + window_size[1], c))
            #print(f"Array: {decisions}\n")
        cv2.imshow("Sliding Window", clone_image)
        cv2.waitKey(1)
        #time.sleep(0.50)
    return decisions


"""x = [
1.    (1.2, 0.6),
2.    [
   0.  (0, 0, 64, 128, [0.78]),
   1.  (0, 0, 64, 128, [0.78]),
   2.  (0, 0, 64, 128, [0.78]),
   3.  (0, 0, 64, 128, [0.78])
     ]
]
"""


def multiscale_function(images: [(np.ndarray, float, float)]) -> [(float, float), [(int, int, int, int, np.ndarray)]]:
    x, y = load_dataset(size=1_000)
    clf, _, _ = tune_hyperparameters(x, y)
    hog = cv2.HOGDescriptor()
    list_of_return: [(), [()]] = []
    for (image, scale_x, scale_y) in images:
        plausibile_rectangular_regions: [()] = []
        for scale in (1.3, 1.0, 0.5):
            plausibile_rectangular_regions += show_window(image, hog, clf, scale)
        list_of_return += [(scale_x, scale_y), plausibile_rectangular_regions]
        print(f"list_of_return: {list_of_return}\n")
    return list_of_return


# ------------------------------------------------------------------------------------------------------------------


def read(cartella: str, numero_immagini=1_000):
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
immagini_2 = standardize_size(immagini)
"""for (image, i, j) in immagini_2:
    cv2.imshow("Prova", image)
    cv2.waitKey(0)
    print(f"Scale_h: {i}\nScale_w: {j}\n-------------------")
#"""
multiscale_function(immagini_2)
