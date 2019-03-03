import logging
import os

import numpy as np
import pyarrow
import sklearn
from numpy import ComplexWarning

from pipeline.flat_feature_sequence import *

import warnings

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]%(processName)s:%(name)s:%(levelname)s - %(message)s')


_NN_INPUT_PYARROW_DIR_ENV = "NN_INPUT_PYARROW_DIR"


__all__ = ["_NN_INPUT_PYARROW_DIR_ENV"]


def main():
    warnings.filterwarnings("ignore", category=ComplexWarning)

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

    if all_loaded:
        train_data, train_target_data, validate_data, validate_target_data, test_set, test_target = all_collections
    else:
        validate_split = 0.2
        train_set, train_target, validate_set, validate_target, test_set, test_target = \
            generate_train_validate_and_test(validation_split=validate_split)

        train_data = convert_feature_list_of_dicts_to_dict_of_np_array(train_set)
        train_target_data = np.array(train_target, dtype=np.int16)
        validate_data = convert_feature_list_of_dicts_to_dict_of_np_array(validate_set)
        validate_target_data = np.array(validate_target, dtype=np.int16)

        for collection_name in ["train_data", "train_target_data", "validate_data", "validate_target_data", "test_set",
                                "test_target"]:
            collection_fn = os.path.join(os.environ.get(_NN_INPUT_PYARROW_DIR_ENV, ""), collection_name + ".pyarrow")
            with open(collection_fn, 'wb+') as f:
                f.write(pyarrow.serialize(locals()[collection_name]).to_buffer())

    with open("validation_prediction.arrow", "rb") as f:
        y_pred = pyarrow.deserialize(f.read())

    compute_roc(validate_target_data, y_pred.flatten())


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
