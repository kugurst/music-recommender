import logging
import multiprocessing
import pickle
import queue
import threading

import numpy as np
import transaction
from keras.utils import Sequence

from file_store import database
from file_store.database import *
from model.nn import InputNames
from params import in_use_features
from pipeline import features

__all__ = ["Sequencer"]

__logger__ = logging.getLogger(__name__)

_RANDOM_SEED = 92


class Sequencer(object):
    def __init__(self, validation_split=0.2, batch_size=64):
        self.train_sequence = None
        self.validate_sequence = None

        self.train_set = None
        self.validate_set = None

        self.validation_split = validation_split
        self.batch_size = batch_size

        self.processes_train = None
        self.thread_train = None
        self.manager_train = None
        self.input_queue_train = None
        self.result_queue_train = None
        self.done_queue_train = None

        self.processes_validate = None
        self.thread_validate = None
        self.manager_validate = None
        self.input_queue_validate = None
        self.result_queue_validate = None
        self.done_queue_validate = None

        self.validate_samples = None

    def initialize_dataset(self):
        good_songs, bad_songs = [], []
        good_collection = SongInfoDatabase.db.collection(database.DB_GOOD_SONGS)
        bad_collection = SongInfoDatabase.db.collection(database.DB_BAD_SONGS)

        total_len = len(good_collection) + len(bad_collection)

        for collection, song_bin, is_good_song in [(good_collection, good_songs, True),
                                                   (bad_collection, bad_songs, False)]:
            for idx in range(len(collection)):
                record = collection.fetch(idx)
                song_sample_id = int(SongInfoDatabase.SongInfoODBC.SONG_SAMPLES_ZODB_ID.get_value(record))

                if song_sample_id >= total_len - 1:
                    continue

                if is_good_song:
                    good_songs.append((song_sample_id, 1))
                else:
                    bad_songs.append((song_sample_id, 0))

        np.random.seed(_RANDOM_SEED)
        np.random.shuffle(good_songs)
        np.random.shuffle(bad_songs)

        good_song_test_len = int(round(len(good_songs) * (1.0 - self.validation_split)))
        bad_song_test_len = int(round(len(bad_songs) * (1.0 - self.validation_split)))

        good_songs_test, good_song_validate = good_songs[:good_song_test_len], good_songs[good_song_test_len:]
        bad_songs_test, bad_song_validate = bad_songs[:bad_song_test_len], bad_songs[bad_song_test_len:]

        train_set = good_songs_test + bad_songs_test
        validate_set = good_song_validate + bad_song_validate

        np.random.shuffle(train_set)
        np.random.shuffle(validate_set)

        self.train_sequence = SampleFeatureGenerator(train_set)
        self.validate_sequence = SampleFeatureGenerator(validate_set, is_validation=True)
        # self.train_sequence, self.processes_train, self.thread_train, self.manager_train, self.input_queue_train, \
        #     self.result_queue_train, self.done_queue_train = data_generator(train_set, self.batch_size)
        #
        # self.validate_sequence, self.processes_validate, self.thread_validate, self.manager_validate, \
        #     self.input_queue_validate, self.result_queue_validate, self.done_queue_validate = \
        #     data_generator(validate_set, self.batch_size, validation_set_len=self.validate_samples)

        self.train_set = train_set
        self.validate_set = validate_set

    def calculate_total_samples(self, data_set):
        ret = 0
        songs, conn = SongSampleZODBDatabase.get_songs(True)
        try:
            idx = 0
            for song_record_id, is_good_song in data_set:
                song_record = songs[song_record_id]
                ret += len(song_record.samples_indexes)

                idx += 1
                if idx % 15 == 14:
                    transaction.abort()
        finally:
            transaction.abort()
            conn.close()
            SongSampleZODBDatabase.close_db()

        return ret


def data_generator(data_set, batch_size, validation_set_len=None):
    manager = multiprocessing.Manager()
    input_queue = manager.Queue(batch_size * 2)
    result_queue = manager.Queue(batch_size * 2)
    done_queue = manager.Queue()
    selected_sample_indexes = dict()

    processes = []

    for _ in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(target=get_features, args=(input_queue, result_queue, data_set))
        processes.append(p)
        p.start()

    thread = threading.Thread(target=_feed_feature_generators,
                              args=(batch_size, data_set, input_queue, done_queue, selected_sample_indexes,
                                    validation_set_len))
    thread.start()

    def generator():
        base_index = 0
        if validation_set_len:
            generator.__len__ = validation_set_len
        else:
            generator.__len__ = len(data_set)
        while True:
            # batch_features = np.zeros((batch_size, 1), dtype=np.float32)
            batch_features = {}
            batch_labels = np.zeros((batch_size, 1), dtype=np.float32)

            for idx in range(min(batch_size, generator.__len__ - base_index)):
                computed_features, chosen_samples = result_queue.get()

                #: :type: pipeline.features.Feature
                computed_features = pickle.loads(computed_features)
                if not validation_set_len:
                    selected_sample_indexes[computed_features.song_record_id] = chosen_samples

                for feature, should_train, feature_name, feature_shape in [
                    (computed_features.normalize_tempo, in_use_features.USE_TEMPO, InputNames.TEMPO,
                     features.TEMPO_SHAPE),
                    (computed_features.normalize_flux, in_use_features.USE_FLUX, InputNames.FLUX, features.FLUX_SHAPE),
                    (computed_features.normalize_rolloff, in_use_features.USE_ROLLOFF, InputNames.ROLLOFF,
                     features.ROLLOFF_SHAPE),
                    (computed_features.normalize_mel, in_use_features.USE_MEL, InputNames.MEL, features.MEL_SHAPE),
                    (computed_features.normalize_contrast, in_use_features.USE_CONTRAST, InputNames.CONTRAST,
                     features.CONTRAST_SHAPE),
                    (computed_features.normalize_tonnetz, in_use_features.USE_TONNETZ, InputNames.TONNETZ,
                     features.TONNETZ_SHAPE),
                    (computed_features.normalize_chroma, in_use_features.USE_CHROMA, InputNames.CHROMA,
                     features.CHROMA_SHAPE),
                    (computed_features.normalize_hpss, in_use_features.USE_HPSS, InputNames.HPSS, features.HPSS_SHAPE),
                    (computed_features.compute_fractional_rms_energy, in_use_features.USE_RMS_FRACTIONAL,
                     InputNames.RMS_FRACTIONAL, features.RMS_SHAPE),
                ]:
                    if not should_train:
                        continue
                    result = feature()
                    shape = (batch_size,) + feature_shape
                    feature_results = batch_features.setdefault(
                        feature_name.get_nn_input_name(), np.zeros(shape, dtype=np.float32)
                    )
                    if result.shape != shape[1:]:
                        feature_results[idx][tuple([slice(0, n) for n in result.shape])] = result
                    else:
                        try:
                            feature_results[idx] = result
                        except np.ComplexWarning:
                            feature_results[idx] = np.abs(result)

                batch_labels[idx] = int(computed_features.is_good_song)

            base_index += batch_size
            if base_index >= generator.__len__:
                base_index = 0

            yield batch_features, batch_labels

    return generator, processes, thread, manager, input_queue, result_queue, done_queue


def _feed_feature_generators(batch_size, data_set, input_queue, done_queue, selected_sample_indexes,
                             validation_set_len=None):
    base_index = 0

    while True:
        try:
            length = len(data_set)
            if validation_set_len:
                length = validation_set_len
            samples_in_batch = min(batch_size, length - base_index)
            for idx in range(samples_in_batch):
                if validation_set_len:
                    input_queue.put((None, None, None, validation_set_len), timeout=1)
                else:
                    song_record_id, song_class = data_set[idx + base_index]
                    __logger__.debug(song_record_id)

                    input_queue.put(
                        (song_record_id, song_class, selected_sample_indexes.setdefault(song_record_id, set()), None),
                        timeout=1
                    )

            base_index += batch_size
            if base_index >= len(data_set):
                base_index = 0
        except queue.Full:
            if done_queue.qsize():
                break


def get_features(input_queue, result_queue, data_set):
    while True:
        try:
            song_record_id, song_class, chosen_samples, validation_set_len = input_queue.get()
            if validation_set_len:
                computed_features = features.compute_features(None, data_set_to_iterate=data_set)
            else:
                computed_features = features.compute_features(song_record_id, try_exclude_samples=chosen_samples)

            if validation_set_len:
                chosen_samples = None
            elif computed_features.sample_index in chosen_samples:
                chosen_samples = {computed_features.sample_index}
            else:
                chosen_samples.add(computed_features.sample_index)

            result_queue.put((
                pickle.dumps(computed_features, pickle.HIGHEST_PROTOCOL), chosen_samples
            ))
        except queue.Empty:
            # break
            pass


class SampleFeatureGenerator(Sequence):
    def __init__(self, song_record_ids, is_validation=False):
        self.song_record_ids = song_record_ids
        self.is_validation = is_validation
        self.__selected_sample_indexes = dict()  # type: dict[int, set[int]]

        self.__len = len(song_record_ids)
        self.__val_map = dict()  # type: dict[int, tuple[int, int, features.Feature]]

        if is_validation:
            idx = 0
            for song_record_id in song_record_ids:
                song_features = SongSamplesFeatureDB.get_db()[song_record_id[0]]
                for sample_idx, feature in enumerate(song_features):
                    self.__val_map[idx] = (song_record_id, sample_idx, feature)
                    idx += 1
            self.__len = len(self.__val_map)

    def __len__(self):
        return self.__len

    def __getitem__(self, index):
        song_record_id = self.song_record_ids[index][0]

        if self.is_validation:
            song_index, sample_index, song_features = self.__val_map[index]
            __logger__.info("Validating song id [{}] and sample id [{}]".format(song_index, sample_index))
        else:
            song_index, sample_index, song_features = SongSamplesFeatureDB.get_feature(
                song_record_id, chosen_samples=self.__selected_sample_indexes.setdefault(song_record_id, set())
            )
            __logger__.info("Training song id [{}] and sample id [{}]".format(song_index, sample_index))

        chosen_samples = self.__selected_sample_indexes.setdefault(song_record_id, set())
        if sample_index in chosen_samples:
            self.__selected_sample_indexes[song_record_id] = {sample_index}
        else:
            chosen_samples.add(sample_index)

        index_features = {}
        index_labels = np.zeros((1,), dtype=np.float32)

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
                if feature_name == InputNames.FLUX:
                    pass
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

        index_labels[0] = song_features.is_good_song

        return index_features, index_labels


class DataSet(Sequence):
    def __init__(self, song_record_ids):
        self.song_record_ids = song_record_ids
        self.__selected_sample_indexes = dict()  # type: dict[int, set[int]]

        self.manager = multiprocessing.Manager()
        self.input_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()

    def __len__(self):
        # return len(self.song_record_ids)
        return 10

    def __getitem__(self, index):
        song_record_id, song_class = self.song_record_ids[index]
        __logger__.debug(song_record_id)

        self.input_queue.put((song_record_id, self.__selected_sample_indexes.setdefault(song_record_id, set())))
        get_features(self.input_queue, self.result_queue)
        computed_features, _, chosen_samples = self.result_queue.get()
        computed_features = pickle.loads(computed_features)
        self.__selected_sample_indexes[song_record_id] = chosen_samples

        # if bool(song_class):
        #     song_record = self.gssc.fetch(song_record_id)
        # else:
        #     song_record = self.bssc.fetch(song_record_id)
        #
        # del song_record

        return {"actor_in_input": np.array([self.song_record_ids[index][0]])}, \
               {"output": np.array([song_class])}
