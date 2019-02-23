import random
import tempfile
import time

from file_store import database
from file_store.database import *
from pipeline.features import *


def test_compute_features():
    gsrc = SongInfoDatabase.db.collection(database.DB_GOOD_SONGS)
    song_index = random.randint(0, len(gsrc) - 1)

    features = compute_features(789, 1)
    print(features)


def test_generate_audio_sample():
    gsrc = SongInfoDatabase.db.collection(database.DB_GOOD_SONGS)
    generate_audio_sample(random.randint(0, len(gsrc) - 1), delete_on_exit=False)
