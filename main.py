import logging
import os

import numpy as np
import pyarrow
from numpy import ComplexWarning

from pipeline.flat_feature_sequence import *

import warnings

from model import nn

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]%(processName)s:%(name)s:%(levelname)s - %(message)s')


_NN_INPUT_PYARROW_DIR_ENV = "NN_INPUT_PYARROW_DIR"


def main():
    warnings.filterwarnings("ignore", category=ComplexWarning)

    model = nn.gen_model()
    print(model.summary())
    # exit(0)
    # sequencer = Sequencer()
    # sequencer.initialize_dataset()
    nn.compile_model(model)


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

    nn.train_model_flat(model, train_data, train_target_data, validate_data, validate_target_data)
    # nn.train_model(model, sequencer)


if __name__ == "__main__":
    main()
