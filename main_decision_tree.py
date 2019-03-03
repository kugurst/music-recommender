import logging
import math
import multiprocessing
import os
import pickle

import numpy as np
import pyarrow
import sklearn

from sklearn import neighbors, model_selection

# import tensorflow as tf

# from main import *
from pipeline import sequencer

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
        tree = sklearn.tree.DecisionTreeClassifier(
            random_state=sequencer._RANDOM_SEED, min_samples_split=int(math.sqrt(30))
        )
        # n_neighbors

        # clf_train_features = np.concatenate((flat_train_features, flat_validate_features), axis=None)
        # clf_target_data = np.concatenate((train_target_data, validate_target_data), axis=None)

        # train_indexes = [[idx for idx in range(len(clf_train_features))]]
        # test_indexes = [[idx for idx in range(len(clf_target_data))]]

        clf = sklearn.model_selection.GridSearchCV(
            estimator=tree, param_grid={
                'max_depth': [15, 30],
                'class_weight': [{0: 0.125, 1: 1}, {0: 0.25, 1: 1}, {0: 1, 1: 1}],
                'max_features': ["auto", "log2"]
            },
            cv=2, scoring=make_model_scorer(), n_jobs=-1, verbose=2, error_score=0, return_train_score=True
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


def compute_precision_recall(model, flat_feature_shape, target_data):
    y_pred = model.predict(flat_feature_shape)
    y_pred = y_pred.flatten()
    return modelless_compute_precision_recall(target_data, y_pred)


def modelless_compute_precision_recall(y_true, y_pred, **kwargs):
    val_precision, val_recall, val_f1, _ = sklearn.metrics.precision_recall_fscore_support(
        y_true.flatten(), y_pred.flatten(), beta=0.5, labels=[0, 1], average="binary")
    combined = val_precision * val_f1
    get_logger().info("— val_f1: % f — val_precision: % f — val_recall % f — val_combined % f" % (
        val_f1, val_precision, val_recall, combined))
    return combined


def make_model_scorer():
    return sklearn.metrics.make_scorer(modelless_compute_precision_recall)


def train_and_validate_tf_knn(flat_train_features, train_target_data, flat_validate_features, validate_target_data,
                              test_set, test_target):
    feature_number = flat_train_features.shape[1]
    train_target_data = train_target_data.reshape((train_target_data.shape[0], 1))

    k = 5

    x_data_train = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)
    y_data_train = tf.placeholder(shape=[None, train_target_data.shape[1]], dtype=tf.float32)
    x_data_test = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)

    # manhattan distance
    distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))), axis=2)

    # nearest k points
    _, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
    top_k_label = tf.gather(y_data_train, top_k_indices)

    sum_up_predictions = tf.reduce_sum(top_k_label, axis=1)
    prediction = tf.argmax(sum_up_predictions, axis=1)

    sess = tf.Session()
    prediction_outcome = sess.run(prediction, feed_dict={x_data_train: flat_train_features,
                                                         x_data_test: flat_validate_features,
                                                         y_data_train: train_target_data})

    # evaluation
    accuracy = 0
    for pred, actual in zip(prediction_outcome, validate_target_data):
        if pred == np.argmax(actual):
            accuracy += 1

    print(accuracy / len(prediction_outcome))


if __name__ == '__main__':
    get_logger().info("Retrieving features")
    flat_train_features, train_target_data, flat_validate_features, validate_target_data, test_set, \
        test_target = flatten_inputs()
    # get_logger().info("Fitting Tree")
    # best_tree = main(flat_train_features, train_target_data, flat_validate_features, validate_target_data, test_set,
    #                  test_target)
    # knn = load_knn()
    with open("tree.pickle", "rb") as f:
        best_tree = pickle.load(f)
    # get_logger().info("Predicting on validation data")
    compute_precision_recall(best_tree, flat_validate_features, validate_target_data)
    # with open("tree.pickle", "wb+") as f:
    #     pickle.dump(best_tree, f, protocol=pickle.HIGHEST_PROTOCOL)
    # get_logger().info("Training TensorFlow k-NN")
    # train_and_validate_tf_knn(flat_train_features, train_target_data, flat_validate_features, validate_target_data,
    #                           test_set, test_target)
    pass
