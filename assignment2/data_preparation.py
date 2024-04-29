import cv2
import os
import subprocess
import random
import numpy as np
def filter(image) :
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    Mag = ((np.absolute(gx) + np.absolute(gy)) / 4).astype(np.uint8)
    return Mag
def data_preparation(path_positive_txt, path_positive_imgs, path_positive_annotations,
                     path_negative_imgs):
    positive_images=[]
    name_positive_images=[]
    with open(path_positive_txt, 'r') as file:
        counter = 0
        for riga in file:
            if (counter > 10):
                break
            nome_immagine = riga.strip()
            nome_immagine +=".jpg"
            name_positive_images.append(nome_immagine + ".txt")
            percorso_immagine = os.path.join(path_positive_imgs, nome_immagine)
            immagine = cv2.imread(percorso_immagine)
            if immagine is not None:
                immagine = filter(immagine)
                positive_images.append(immagine)
                counter += 1
            else:
                print(f"Impossibile caricare l'immagine: {nome_immagine}")
    file.close()
    bbox_positive = []
    for name_positive_image in name_positive_images:
        img_bbox = []
        counter = 0
        if (counter > 10):
            break
        with open(path_positive_annotations + "/" + name_positive_image, "r") as file:
            for riga in file:
                bbox = riga.strip("\n")
                if bbox[0:2] == "1 " :
                    bbox = bbox.split()
                    bbox = [int(num) for num in bbox[1:]]
                    img_bbox.append(bbox)
        counter += 1
        bbox_positive.append(img_bbox)
    file.close()
    crop_positive_imgs = []
    #effettuo il crop dell'immagine
    for i in range(len(positive_images)):
        for j in range(len(bbox_positive[i])):
            x1,y1,x2,y2 = bbox_positive[i][j][0], bbox_positive[i][j][1], bbox_positive[i][j][2], bbox_positive[i][j][3]
            image_crop = positive_images[i][y1:y2, x1:x2]
            resized = cv2.resize(image_crop, (64, 128))
            crop_positive_imgs.append(resized)
    negative_images = []
    name_negative_images = []
    bbox_negative = []
    #Ottengo i nomi delle immagini della mia cartella
    process = subprocess.Popen(["ls", path_negative_imgs], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, error = process.communicate()
    # Controlla se ci sono errori
    if error:
        print("Si è verificato un errore:", error)
    else:
        # L'outuput lo inserisco in una lista, ma preso solo i primi n elementi
        name_negative_images = output.strip().split("\n")[:50]
    for i in name_negative_images:
        counter = 0
        if (counter > 10):
            break
        immagine = cv2.imread(path_negative_imgs + "/" + i)
        img_bbox = []
        if immagine is not None:
            #prendo l'immagine
            negative_images.append(immagine)
            for i in range(5):
                #genero casualmente una bounding box
                h, w = immagine.shape[:2]
                x = random.randint(0, w-65)
                y = random.randint(0, h-129)
                window = [x, y, x + 64, y + 128]
                img_bbox.append(window)
            counter +=1
            bbox_negative.append(img_bbox)
        else:
            print(f"Impossibile caricare l'immagine: {nome_immagine}")
    # effettuo il crop dell'immagine
    crop_negative_imgs = []
    for i in range(len(negative_images)):
        for j in range(len(bbox_negative[i])):
            x1, y1, x2, y2 = bbox_negative[i][j][0], bbox_negative[i][j][1], bbox_negative[i][j][2], bbox_negative[i][j][3]
            image_crop = negative_images[i][y1:y2, x1:x2]
            resized = cv2.resize(image_crop, (64, 128))
            crop_negative_imgs.append(resized)
    return crop_positive_imgs, crop_negative_imgs

path_positive_txt = "/Users/antoninocentonze/Desktop/assigment_2/train_assignment.txt"
path_positive_imgs = "/Users/antoninocentonze/Desktop/assigment_2/WiderPerson/Images"
path_positive_annotations = "/Users/antoninocentonze/Desktop/assigment_2/WiderPerson/Annotations"
path_negative_imgs ="/Users/antoninocentonze/Desktop/assigment_2/train_neg"
# Carica il training set
positive_images, negative_images = data_preparation(path_positive_txt, path_positive_imgs, path_positive_annotations, path_negative_imgs)
#estrazione delle features
hog = cv2.HOGDescriptor()
positive_descriptors = []
for img in positive_images:
    positive_descriptors.append(hog.compute(img))
negative_descriptors = []
for img in negative_images:
    negative_descriptors.append(hog.compute(img))
