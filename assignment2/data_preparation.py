import cv2
import os
import numpy as np
def data_preparation(path_positive_txt, path_positive_imgs, path_positive_annotations
                     , path_negative_txt, path_negative_imgs, path_negative_annotations):
    positive_images= []
    name_positive_images=[]
    with open(path_positive_txt, 'r') as file:
        counter = 0
        for riga in file:
            if (counter > 10):
                break
            nome_immagine = riga.strip()
            nome_immagine +=".jpg"
            name_positive_images.append(nome_immagine + ".txt")
            print(nome_immagine)
            percorso_immagine = os.path.join(path_positive_imgs, nome_immagine)
            immagine = cv2.imread(percorso_immagine)
            if immagine is not None:
                positive_images.append(immagine)
                counter += 1
            else:
                print(f"Impossibile caricare l'immagine: {nome_immagine}")
    file.close()
    bbox_positive = []
    for name_positive_image in name_positive_images:
        counter = 0;
        if (counter > 10):
            break
        with open(path_positive_annotations + "/" + name_positive_image, "r") as file:
            for riga in file:
                bbox = riga.strip("\n")
                if bbox[0:2] == "1 " :
                    bbox_positive.append(bbox)
                    print(bbox)
                    counter += 1
        counter += 1
        print("\n")
    file.close()


path_positive_txt = "/Users/antoninocentonze/Desktop/assigment_2/train_assignment.txt"
path_positive_imgs = "/Users/antoninocentonze/Desktop/assigment_2/WiderPerson/Images"
path_positive_annotations = ("/Users/antoninocentonze/Desktop/assigment_2/WiderPerson/Annotations")
# Carica il training set
data_preparation(path_positive_txt, path_positive_imgs, path_positive_annotations, "", "", "")
