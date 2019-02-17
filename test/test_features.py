import random

from file_store import database
from file_store.database import *


def test_compute_features():
    #: :type: unqlite.Collection
    good_song_representation_collection = SongInfoDatabase.db.collection(database.DB_GOOD_SONG_REPRESENTATIONS)
    #: :type: unqlite.Collection
    bad_song_representation_collection = SongInfoDatabase.db.collection(database.DB_BAD_SONG_REPRESENTATIONS)

    rand_idx = random.choice(range(len(good_song_representation_collection)))
    rand_song = good_song_representation_collection.fetch(rand_idx)


def test_generate_audio_sample():

