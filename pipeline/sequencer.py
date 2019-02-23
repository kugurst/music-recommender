import logging

import numpy as np
from keras.utils import Sequence

from file_store import database
from file_store.database import *

__all__ = ["Sequencer"]

__logger__ = logging.getLogger(__name__)


class Sequencer(object):
    def __init__(self, validation_split=0.2):
        self.train_set = None
        self.validate_set = None
        self.validation_split = validation_split

    def initialize_dataset(self):
        good_songs, bad_songs = [], []
        for collection, song_bin, is_good_song in [
            (SongInfoDatabase.db.collection(database.DB_GOOD_SONGS), good_songs, True),
            (SongInfoDatabase.db.collection(database.DB_BAD_SONGS), bad_songs, False)
        ]:
            for idx in range(len(collection)):
                record = collection.fetch(idx)
                song_sample_id = SongInfoDatabase.SongInfoODBC.SONG_SAMPLES_ID.get_value(record)
                if is_good_song:
                    good_songs.append((song_sample_id, 1))
                else:
                    bad_songs.append((song_sample_id, 0))

        np.random.shuffle(good_songs)
        np.random.shuffle(bad_songs)

        good_song_test_len = int(round(len(good_songs) * (1.0 - self.validation_split)))
        bad_song_test_len = int(round(len(bad_songs) * (1.0 - self.validation_split)))

        good_songs_test, good_song_validate = good_songs[:good_song_test_len], good_songs[good_song_test_len:]
        bad_songs_test, bad_song_validate = bad_songs[:bad_song_test_len], bad_songs[bad_song_test_len:]

        test_set = good_songs_test + bad_songs_test
        validate_set = good_song_validate + bad_song_validate

        np.random.shuffle(test_set)
        np.random.shuffle(validate_set)

        self.train_set = DataSet(test_set)
        self.validate_set = DataSet(validate_set)


class DataSet(Sequence):
    def __init__(self, song_record_ids):
        self.song_record_ids = song_record_ids
        self.__selected_sample_indexes = dict()  # type: dict[int, set[int]]

    def __len__(self):
        return len(self.song_record_ids)

    def __getitem__(self, index):
        song_record_id, song_class = self.song_record_ids[index]
        __logger__.debug(song_record_id)

        song_record = SongSamplesLVLDatabase.SongSamplesIndex.fetch(song_record_id)

        return {"actor_in_input": np.array([self.song_record_ids[index][0]])},\
               {"output": np.array([song_class])}
