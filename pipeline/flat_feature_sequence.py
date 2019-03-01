import numpy as np

from file_store import database
from file_store.database import *
from model.nn import *
from params import in_use_features
from pipeline import features

_RANDOM_SEED = 92

__all__ = ["generate_train_validate_and_test", "convert_feature_list_of_dicts_to_dict_of_np_array"]


def convert_feature_list_of_dicts_to_dict_of_np_array(feature_set):
    total_dict = {}
    num_samples = len(feature_set)
    for idx, sample_features in enumerate(feature_set):
        for feature_name in sample_features:
            feature_value = sample_features[feature_name]

            dtype = feature_value.dtype
            feature_shape = feature_value.shape

            np_arr = total_dict.setdefault(feature_name, np.zeros((num_samples,) + feature_shape, dtype=dtype))
            np_arr[idx] = feature_value

    return total_dict


def generate_train_validate_and_test(validation_split=0.2, test_split=0.1, shuffle_seed=_RANDOM_SEED):
    train_set_indexes, validate_set_indexes, test_set_indexes = initialize_dataset(
        validation_split=validation_split, test_split=test_split, shuffle_seed=shuffle_seed
    )
    train_set, train_target = _load_features_from_indexes(train_set_indexes)
    validate_set, validate_target = _load_features_from_indexes(validate_set_indexes)
    test_set, test_target = _load_features_from_indexes(test_set_indexes)

    if shuffle_seed is not None:
        train_data = list(zip(train_set, train_target))
        validate_data = list(zip(validate_set, validate_target))
        test_data = list(zip(test_set, test_target))

        np.random.seed(shuffle_seed)
        np.random.shuffle(train_data)
        np.random.shuffle(validate_data)
        np.random.shuffle(test_data)

        train_set, train_target = list(zip(*train_data))
        validate_set, validate_target = list(zip(*validate_data))
        test_set, test_target = list(zip(*test_data))

    return train_set, train_target, validate_set, validate_target, test_set, test_target


def _load_features_from_indexes(indexes):
    data, targets = [], []

    for index, song_class in indexes:
        features_list = SongSamplesFeatureDB.get_db()[index]
        for feature in features_list:
            model_input, model_target = _compute_model_input_and_target(feature)
            data.append(model_input)
            targets.append(model_target)

    return data, targets


def _compute_model_input_and_target(song_features):
    index_features = {}

    for feature, should_train, feature_name, model_feature_shape in [
        (song_features.normalize_tempo, in_use_features.USE_TEMPO, InputNames.TEMPO,
         features.TEMPO_SHAPE),
        (song_features.normalize_flux, in_use_features.USE_FLUX, InputNames.FLUX, features.FLUX_SHAPE),
        (song_features.normalize_rolloff, in_use_features.USE_ROLLOFF, InputNames.ROLLOFF,
         features.ROLLOFF_SHAPE),
        (song_features.normalize_mel, in_use_features.USE_MEL, InputNames.MEL, features.MEL_SHAPE),
        (song_features.normalize_contrast, in_use_features.USE_CONTRAST, InputNames.CONTRAST,
         features.CONTRAST_SHAPE),
        (song_features.normalize_tonnetz, in_use_features.USE_TONNETZ, InputNames.TONNETZ,
         features.TONNETZ_SHAPE),
        (song_features.normalize_chroma, in_use_features.USE_CHROMA, InputNames.CHROMA,
         features.CHROMA_SHAPE),
        (song_features.normalize_hpss, in_use_features.USE_HPSS, InputNames.HPSS, features.HPSS_SHAPE),
        (song_features.get_fraction_rms_energy, in_use_features.USE_RMS_FRACTIONAL,
         InputNames.RMS_FRACTIONAL, features.RMS_SHAPE),
    ]:
        if not should_train:
            continue

        result = feature()

        if not hasattr(result, "shape"):
            result = np.array([result])
            # if feature_name == InputNames.FLUX:
            #     pass
            index_features[feature_name.get_nn_input_name()] = result
        elif result.shape != model_feature_shape:
            model_feature_input = np.zeros(model_feature_shape, dtype=np.float32)
            model_feature_input[tuple([slice(0, n) for n in result.shape])] = result
            index_features[feature_name.get_nn_input_name()] = model_feature_input
        else:
            try:
                index_features[feature_name.get_nn_input_name()] = result
            except np.ComplexWarning:
                index_features[feature_name.get_nn_input_name()] = np.abs(result)

    return index_features, song_features.is_good_song


def initialize_dataset(validation_split=0.2, test_split=0.1, shuffle_seed=_RANDOM_SEED):
    good_songs, bad_songs = [], []
    good_collection = SongInfoDatabase.db.collection(database.DB_GOOD_SONGS)
    bad_collection = SongInfoDatabase.db.collection(database.DB_BAD_SONGS)

    total_len = len(good_collection) + len(bad_collection)

    for collection, song_bin, is_good_song in [(good_collection, good_songs, True),
                                               (bad_collection, bad_songs, False)]:
        for idx in range(len(collection)):
            # if idx > 64:
            #     break
            record = collection.fetch(idx)
            song_sample_id = int(SongInfoDatabase.SongInfoODBC.SONG_SAMPLES_ZODB_ID.get_value(record))

            if song_sample_id >= total_len - 1:
                continue

            if is_good_song:
                good_songs.append((song_sample_id, 1))
            else:
                bad_songs.append((song_sample_id, 0))

    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)
        np.random.shuffle(good_songs)
        np.random.shuffle(bad_songs)

    good_song_validate_len = int(round(len(good_songs) * validation_split))
    bad_song_validate_len = int(round(len(bad_songs) * validation_split))
    good_song_test_len = int(round(len(good_songs) * test_split))
    bad_song_test_len = int(round(len(bad_songs) * test_split))

    good_songs_train = good_songs[:len(good_songs) - good_song_validate_len - good_song_test_len]
    bad_songs_train = bad_songs[:len(bad_songs) - bad_song_validate_len - bad_song_test_len]

    good_songs_validate = good_songs[len(good_songs_train):len(good_songs_train) + good_song_validate_len]
    bad_songs_validate = bad_songs[len(bad_songs_train):len(bad_songs_train) + bad_song_validate_len]

    good_songs_test = good_songs[len(good_songs_train) + len(good_songs_validate):]
    bad_songs_test = good_songs[len(bad_songs_train) + len(bad_songs_validate):]

    train_set = good_songs_train + bad_songs_train
    validate_set = good_songs_validate + bad_songs_validate
    test_set = good_songs_test + bad_songs_test

    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)
        np.random.shuffle(train_set)
        np.random.shuffle(validate_set)
        np.random.shuffle(test_set)

    return train_set, validate_set, test_set
