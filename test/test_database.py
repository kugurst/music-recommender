from file_store import database

from file_store.database import *


def test_update_song_paths_with_fft_status():
    #: :type: unqlite.Collection
    good_song_paths_collection = SongInfoDatabase.db.collection(database.DB_GOOD_SONGS)
    #: :type: unqlite.Collection
    bad_song_paths_collection = SongInfoDatabase.db.collection(database.DB_BAD_SONGS)

    update_collection_with_field(good_song_paths_collection, SongInfoDatabase.SongInfoODBC.FFTS_BUILT.value, False)
    update_collection_with_field(bad_song_paths_collection, SongInfoDatabase.SongInfoODBC.FFTS_BUILT.value, False)


def test_update_song_paths_with_sample_rate():
    #: :type: unqlite.Collection
    good_song_reprs_collection = SongInfoDatabase.db.collection(database.DB_GOOD_SONG_REPRESENTATIONS)
    #: :type: unqlite.Collection
    bad_song_reprs_collection = SongInfoDatabase.db.collection(database.DB_BAD_SONG_REPRESENTATIONS)

    update_collection_with_field(good_song_reprs_collection, SongInfoDatabase.SongSamplesODBC.SONG_SAMPLE_RATE, 44100)
    update_collection_with_field(bad_song_reprs_collection, SongInfoDatabase.SongSamplesODBC.SONG_SAMPLE_RATE, 44100)


def test_remove_ffts_built_enum():
    #: :type: unqlite.Collection
    good_song_paths_collection = SongInfoDatabase.db.collection(database.DB_GOOD_SONGS)
    #: :type: unqlite.Collection
    bad_song_paths_collection = SongInfoDatabase.db.collection(database.DB_BAD_SONGS)

    remove_field_from_collection(good_song_paths_collection, str(SongInfoDatabase.SongInfoODBC.FFTS_BUILT))
    remove_field_from_collection(bad_song_paths_collection, str(SongInfoDatabase.SongInfoODBC.FFTS_BUILT))
