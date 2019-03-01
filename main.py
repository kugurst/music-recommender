import logging

import numpy as np
from numpy import ComplexWarning

from pipeline.flat_feature_sequence import *

import warnings

from model import nn

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]%(processName)s:%(name)s:%(levelname)s - %(message)s')


def main():
    warnings.filterwarnings("ignore", category=ComplexWarning)

    model = nn.gen_model()
    print(model.summary())
    # exit(0)
    # sequencer = Sequencer()
    # sequencer.initialize_dataset()
    nn.compile_model(model)

    validate_split = 0.2
    train_set, train_target, validate_set, validate_target, test_set, test_target = generate_train_validate_and_test(
        validation_split=validate_split)

    train_data = convert_feature_list_of_dicts_to_dict_of_np_array(train_set)
    train_target_data = np.array(train_target, dtype=np.int16)
    validate_data = convert_feature_list_of_dicts_to_dict_of_np_array(validate_set)
    validate_target_data = np.array(validate_target, dtype=np.int16)

    nn.train_model_flat(model, train_data, train_target_data, validate_data, validate_target_data) #, validation_split=validate_split)
    # nn.train_model(model, sequencer)


if __name__ == "__main__":
    main()
