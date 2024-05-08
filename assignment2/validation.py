import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

from joblib import dump, load

import cv2

from new import *


def _build_samples(n=None, neg_factor=5):
    fnames = read_pos_images_list(os.environ['VAL_ASSIGNMENT_TXT_PATH'], n)
    bboxes = read_bboxes(fnames, os.environ['ANNOTATIONS_PATH'])
    imgs = read_pos_images(fnames, os.environ['POS_IMAGES_PATH'])
    pos_samples = build_pos_samples(imgs, bboxes)

    fnames = build_neg_images_list(os.environ['VAL_IMAGES_PATH'], n)
    neg_samples = build_neg_samples(fnames, os.environ['VAL_IMAGES_PATH'], neg_factor)

    return pos_samples, neg_samples


def _build_validation_set(pos_samples, neg_samples, size=None, cache=True):
    hog = cv2.HOGDescriptor()
    data = []
    targets = []
    for sample in pos_samples:
        data.append(hog.compute(sample))
        targets.append(1)
    for sample in neg_samples:
        data.append(hog.compute(sample))
        targets.append(-1)

    assert len(data) == len(targets), "Samples and targets should have same length."
    # shuffle samples
    data, targets = shuffle(data, targets)
    # slice if desired size was specified
    if size:
        data, targets = data[:size], targets[:size]

    X, y = np.array(data, dtype=np.float32), np.array(targets, dtype=np.float32)

    if cache:
        # join features and targets
        data = np.column_stack((X, y))
        cache_ndarray(data, 'val_descriptor_data.npy')

    return X, y


def _load_validation_set(use_cache=True, size=None) -> (np.ndarray, np.ndarray):
    if use_cache:
        dir_path = os.environ.get('CACHE_DIR_PATH')
        file_path = os.path.join(dir_path, 'val_descriptor_data.npy')
        if os.path.isfile(file_path):
            if size is not None:
                warnings.warn("Size can't be specified when loading dataset from cache, it's being ignored.")
            data = np.load(file_path)
            return data[:, :-1], data[:, -1]
        else:
            warnings.warn('Cache file not found. Rebuilding dataset from scratch...')

    # if we didn't return, we didn't use cache -> rebuild dataset
    pos_samples, neg_samples = _build_samples(size)
    return _build_validation_set(pos_samples, neg_samples, size, use_cache)


def tune_hyperparameters(X_train, y_train, use_cache=True) -> (object, dict, float):
    if use_cache:
        dir_path = os.environ.get('CACHE_DIR_PATH')
        model_file_path = os.path.join(dir_path, 'fit_best_model.joblib')
        model_info_file_path = os.path.join(dir_path, 'fit_best_model_info.joblib')
        if os.path.isfile(model_file_path) and os.path.isfile(model_info_file_path):
            model = load(model_file_path)
            model_info = load(model_info_file_path)
            return model, model_info['best_params'], model_info['best_score']
        else:
            warnings.warn('Validated model cache file not found. Revalidating model from scratch...')

    X_val, y_val = _load_validation_set(use_cache)
    logging.info('BUILT VALIDATION SET')
    X = np.concatenate((X_train, X_val))
    y = np.concatenate((y_train, y_val))

    # this will tell GridSearchCV where to split data between train and val set
    train_idxs = np.full(shape=(X_train.shape[0],), fill_value=-1)
    val_idxs = np.full(shape=(X_val.shape[0],), fill_value=0)
    split_idxs = np.append(train_idxs, val_idxs)
    split = PredefinedSplit(split_idxs)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('dim_reducer', PCA(svd_solver='full')),
        ('classifier', LinearSVC()),
    ])
    param_grid = {
        'scaler': ['passthrough', StandardScaler()],
        'dim_reducer': [PCA(svd_solver='full')],
        'dim_reducer__n_components': [0.95, 0.9, 0.85],
        'classifier__C': [1e-5, 1e-3, 1e-1, 1, 10, 100],
    }
    gs = GridSearchCV(pipe, param_grid, scoring='f1', cv=split, verbose=3, n_jobs=4)
    gs.fit(X, y)
    best_model = gs.best_estimator_
    if use_cache:
        dir_path = os.environ.get('CACHE_DIR_PATH')
        model_file_path = os.path.join(dir_path, 'fit_best_model.joblib')
        model_info_file_path = os.path.join(dir_path, 'fit_best_model_info.joblib')

        model_info = {'best_params': gs.best_params_, 'best_score': gs.best_score_}
        dump(best_model, model_file_path)
        dump(model_info, model_info_file_path)
    return best_model, gs.best_params_, gs.best_score_


if __name__ == '__main__':
    X, y = load_dataset(use_cache=True, size=0.5)
    print(X.shape)
    logging.info('BUILT TRAINING SET')
    model, params, score = tune_hyperparameters(X, y)
    print(model.named_steps['classifier'], params, score)