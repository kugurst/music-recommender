import logging
import time

from file_store import database
from file_store.database import SongSamplesDatabase
from pre_process.audio_representation import *


def test_build_song_indexes():
    build_song_indexes()


def test_build_song_samples():
    build_song_samples()


def test_build_song_lvl_samples():
    build_song_samples(backend=DatabaseBackend.LVL)


def test_build_song_zodb_samples():
    build_song_samples(backend=DatabaseBackend.ZODB)


def test_unqlite_memory_leak():
    for collection in [SongSamplesDatabase.db.collection(database.DB_GOOD_SONGS),
                       SongSamplesDatabase.db.collection(database.DB_BAD_SONGS)]:
        for idx in range(len(collection)):
            record = SongSamplesDatabase.db.collection(database.DB_GOOD_SONGS).fetch(idx)
            print(record["hash"])
            del record
