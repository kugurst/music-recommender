import logging
import multiprocessing
import os
import pickle

import numpy as np
import pyarrow
import sklearn

from sklearn import neighbors, model_selection

# import tensorflow as tf

# from main import *

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]%(processName)s:%(name)s:%(levelname)s - %(message)s')

__logger__ = None

_NN_INPUT_PYARROW_DIR_ENV = "NN_INPUT_PYARROW_DIR"


def get_logger():
    global __logger__
    if __logger__ is None:
        __logger__ = logging.getLogger(__name__)
    return __logger__


def flatten_inputs():
    all_loaded = True
    all_collections = []
    for collection_name in ["train_data", "train_target_data", "validate_data", "validate_target_data", "test_set",
                            "test_target"]:
        collection_fn = os.path.join(os.environ.get(_NN_INPUT_PYARROW_DIR_ENV, ""), collection_name + ".pyarrow")
        if not os.path.exists(collection_fn):
            print("Collection not persisted: [{}]. Recomputing...".format(collection_fn))
            all_loaded = False
            break
        with open(collection_fn, 'rb') as f:
            all_collections.append(pyarrow.deserialize(f.read()))

    if not all_loaded:
        raise ValueError("Must pickle features before calling this function")

    train_data, train_target_data, validate_data, validate_target_data, test_set, test_target = all_collections
    train_num_samples = len(train_data[list(train_data.keys())[0]])
    validate_num_samples = len(validate_data[list(validate_data.keys())[0]])
    # test_num_samples = len(test_set[list(test_set.keys())[0]])

    keys = []
    flat_train_feature_shape = [train_num_samples, 0]
    flat_validate_feature_shape = [validate_num_samples, 0]
    # flat_test_feature_shape = [test_num_samples, 0]
    for key in train_data:
        keys.append(key)
        feature = train_data[key][0]
        flat_train_feature_shape[1] += feature.flatten().shape[0]
        flat_validate_feature_shape[1] += feature.flatten().shape[0]
        # flat_test_feature_shape[1] += feature.flatten().shape[0]

    flat_train_features = np.zeros(shape=flat_train_feature_shape, dtype=np.float32)
    flat_validate_features = np.zeros(shape=flat_validate_feature_shape, dtype=np.float32)
    # flat_test_features = np.zeros(shape=flat_test_feature_shape, dtype=np.float32)

    for num_samples, data_features, data_set in [(train_num_samples, flat_train_features, train_data),
                                                 (validate_num_samples, flat_validate_features, validate_data)]:
        for sample_idx in range(num_samples):
            sample_flat_features = []
            for key in keys:
                sample_flat_features.append(data_set[key][sample_idx].flatten())

            data_features[sample_idx] = np.concatenate(sample_flat_features, axis=None)

    return flat_train_features, train_target_data, flat_validate_features, validate_target_data, test_set, \
           test_target


def main(flat_train_features, train_target_data, flat_validate_features, validate_target_data, test_set,
         test_target):
    with sklearn.utils.parallel_backend('multiprocessing'):
        knn = sklearn.neighbors.KNeighborsClassifier(weights='distance', n_jobs=multiprocessing.cpu_count())
        # n_neighbors

        # clf_train_features = np.concatenate((flat_train_features, flat_validate_features), axis=None)
        # clf_target_data = np.concatenate((train_target_data, validate_target_data), axis=None)

        # train_indexes = [[idx for idx in range(len(clf_train_features))]]
        # test_indexes = [[idx for idx in range(len(clf_target_data))]]

        clf = sklearn.model_selection.GridSearchCV(
            estimator=knn, param_grid={'n_neighbors': [5, 10, 20, 50, 100]}, scoring=make_knn_scorer(), n_jobs=-1,
            verbose=2, error_score=0, return_train_score=True
        )

        clf.fit(flat_train_features, train_target_data)
        with open('saved_models/clf.weights.from_scratch.pickle', 'wb+') as f:
            pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)

        return clf.best_estimator_


def load_knn():
    with open('saved_models/knn.weights.from_scratch.pyarrow', 'rb') as f:
        knn = pickle.load(f)
    return knn


# # https://ensemblearner.github.io/blog/2017/04/01/knn
# def predict(X_t, y_t, x_t, k_t):
#     neg_one = tf.constant(-1.0, dtype=tf.float64)
#     # we compute the L-1 distance
#     distances = tf.reduce_sum(tf.abs(tf.subtract(X_t, x_t)), 1)
#     # to find the nearest points, we find the farthest points based on negative distances
#     # we need this trick because tensorflow has top_k api and no closest_k or reverse=True api
#     neg_distances = tf.multiply(distances, neg_one)
#     # get the indices
#     vals, indx = tf.nn.top_k(neg_distances, k_t)
#     # slice the labels of these points
#     y_s = tf.gather(y_t, indx)
#     return y_s
#
#
# # https://ensemblearner.github.io/blog/2017/04/01/knn
# def get_label(preds):
#     counts = np.bincount(preds.astype('int64'))
#     return np.argmax(counts)
#
#
# def main_tf():
#     pass


def compute_precision_recall(knn, flat_feature_shape, target_data):
    y_pred = knn.predict(flat_feature_shape)
    y_pred = y_pred.flatten()
    # tn, fp, fn, tp = sklearn.metrics.confusion_matrix(self.validate_target, y_pred_label)

    val_precision, val_recall, val_f1, _ = sklearn.metrics.precision_recall_fscore_support(
        target_data, y_pred, beta=0.5, labels=[0, 1], average="binary")
    get_logger().info("— val_f1: % f — val_precision: % f — val_recall % f" % (val_f1, val_precision, val_recall))


def knn_less_compute_precision_recall(y_true, y_pred, **kwargs):
    val_precision, val_recall, val_f1, _ = sklearn.metrics.precision_recall_fscore_support(
        y_true.flatten(), y_pred.flatten(), beta=0.5, labels=[0, 1], average="binary")
    get_logger().info("— val_f1: % f — val_precision: % f — val_recall % f" % (val_f1, val_precision, val_recall))
    return val_precision


def make_knn_scorer():
    return sklearn.metrics.make_scorer(knn_less_compute_precision_recall)


if __name__ == '__main__':
    get_logger().info("Retrieving features")
    flat_train_features, train_target_data, flat_validate_features, validate_target_data, test_set, \
    test_target = flatten_inputs()
    get_logger().info("Fitting kNN")
    knn = main(flat_train_features, train_target_data, flat_validate_features, validate_target_data, test_set,
               test_target)
    # knn = load_knn()
    get_logger().info("Predicting on validation data")
    compute_precision_recall(knn, flat_validate_features, validate_target_data)
    # get_logger().info("Training TensorFlow k-NN")
    # train_and_validate_tf_knn(flat_train_features, train_target_data, flat_validate_features, validate_target_data,
    #                           test_set, test_target)
    pass
