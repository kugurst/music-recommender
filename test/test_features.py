import random
import tempfile
import time

from file_store import database
from file_store.database import *
from pipeline.features import *


def test_blah():
    songs, _ = SongSampleZODBDatabase.get_songs(True)
    print(len(songs))


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
