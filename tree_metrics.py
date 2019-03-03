import logging
import os
import pickle

import numpy as np
import pyarrow
import sklearn
from numpy import ComplexWarning

from main_decision_tree import flatten_inputs
from pipeline.flat_feature_sequence import *

import warnings

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]%(processName)s:%(name)s:%(levelname)s - %(message)s')

_NN_INPUT_PYARROW_DIR_ENV = "NN_INPUT_PYARROW_DIR"

__all__ = ["_NN_INPUT_PYARROW_DIR_ENV"]


__logger__ = None


def get_logger():
    global __logger__
    if __logger__ is None:
        __logger__ = logging.getLogger(__name__)
    return __logger__


def main():
    get_logger().info("Retrieving features")
    flat_train_features, train_target_data, flat_validate_features, validate_target_data, test_set, \
        test_target = flatten_inputs()

    with open("tree.pickle", "rb") as f:
        tree = pickle.load(f)

    y_pred = tree.predict_proba(flat_validate_features)
    compute_roc(validate_target_data, y_pred[:, 1])


def compute_roc(y_true, y_pred):
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true, y_pred)
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == "__main__":
    main()
