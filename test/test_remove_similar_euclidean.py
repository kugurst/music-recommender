import os

import numpy as np
import pyarrow
import sklearn

from main import *


def test_remove_euclidean_similar():
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
    num_samples = len(train_data[list(train_data.keys())[0]])

    keys = []
    key_list_flat_shapes = []
    for key in train_data:
        keys.append(key)
        feature = train_data[key][0]
        key_list_flat_shapes.append(feature.flatten().shape[0])

    flat_features = [np.zeros(shape=(num_samples, key_list_flat_shape), dtype=np.float32) for key_list_flat_shape in
                     key_list_flat_shapes]
    for key_idx, key in enumerate(keys):
        for sample_idx in range(num_samples):
            flat_features[key_idx][sample_idx] = train_data[key][sample_idx].flatten()

    euclidean_distances = []
    for flat_feature in flat_features:
        # ret = scipy.spatial.distance.pdist(flat_feature)
        euclidean_distances.append(sklearn.metrics.euclidean_distances(flat_feature[:10000], flat_feature[:10000]))
    euclidean_distances = np.sum(euclidean_distances, axis=0)
    euclidean_distances = euclidean_distances/np.max(euclidean_distances)

    similar_elements = np.argwhere(euclidean_distances < 0.1)
    filtered_similar_elements = []
    for index in similar_elements:
        if index[0] != index[1]:
            filtered_similar_elements.append(index)
    pass
