import data, test, nms, svm, validation, window, cv2

if __name__ == '__main__':
    x, y = data.load_dataset(size=0.5)
    model,_,_ = validation.tune_hyperparameters(x,y)
    immagini, bboxes = test.load_test_set()
    i = 1
    immagini, bboxes = immagini[:i], bboxes[:i]
    data = window.predict(images=immagini, clf=model)
    test.test_model(immagini, bboxes, data)
