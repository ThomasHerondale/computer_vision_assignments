import new, test, nms, svm, validation, window, cv2

if __name__ == '__main__':
    x, y = new.load_dataset(size=0.5)
    model,_,_ = validation.tune_hyperparameters(x,y)
    immagini, bboxes = test.load_test_set()
    hog = cv2.HOGDescriptor()
    lista_immagini = window.standardize_size(immagini)
    data = window.multiscale_function(lista_immagini, model, hog)
    nms.test_model(data)
