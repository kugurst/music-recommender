import random
import tempfile
import time

from file_store import database
from file_store.database import *
from pipeline.features import *


def test_blah():
    songs, _ = SongSampleZODBDatabase.get_songs(True)
    print(len(songs))


def test_iterate_over_all_features():
        good_songs, bad_songs = [], []
        good_collection = SongInfoDatabase.db.collection(database.DB_GOOD_SONGS)
        bad_collection = SongInfoDatabase.db.collection(database.DB_BAD_SONGS)
        total_len = len(good_collection) + len(bad_collection)
        for collection, song_bin, is_good_song in [(good_collection, good_songs, True),
                                                   (bad_collection, bad_songs, False)]:
            for idx in range(len(collection)):
                record = collection.fetch(idx)
                song_sample_id = SongInfoDatabase.SongInfoODBC.SONG_SAMPLES_UNQLITE_ID.get_value(record)
                if song_sample_id >= total_len - 1:
                    continue

                if is_good_song:
                    good_songs.append((song_sample_id, 1))
                else:
                    bad_songs.append((song_sample_id, 0))

        songs, conn = SongSampleZODBDatabase.get_songs(True)
        for song_id, is_good_song in good_songs + bad_songs:
            song = songs.get(song_id)
            print(song.get_samples_left()[0].shape)


def test_compute_features():
    gsrc = SongInfoDatabase.db.collection(database.DB_GOOD_SONGS)
    song_index = random.randint(0, len(gsrc) - 1)

    features = compute_features(123)
    print(features.normalize_mel().shape)
    print(features.normalize_contrast().shape)
    print(features.normalize_chroma().shape)
    print(features.normalize_flux().shape)
    print(features)


def test_generate_audio_sample():
    gsrc = SongInfoDatabase.db.collection(database.DB_GOOD_SONGS)
    generate_audio_sample(random.randint(0, len(gsrc) - 1), delete_on_exit=False)
