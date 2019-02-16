from file_store import database

from file_store.database import *


def test_update_song_paths_with_fft_status():
    #: :type: unqlite.Collection
    good_song_paths_collection = Database.db.collection(database.DB_GOOD_SONG_PATHS)
    #: :type: unqlite.Collection
    bad_song_paths_collection = Database.db.collection(database.DB_BAD_SONG_PATHS)

    update_collection_with_field(good_song_paths_collection, Database.SongPathODBC.FFTS_BUILT.value, False)
    update_collection_with_field(bad_song_paths_collection, Database.SongPathODBC.FFTS_BUILT.value, False)


def test_remove_ffts_built_enum():
    #: :type: unqlite.Collection
    good_song_paths_collection = Database.db.collection(database.DB_GOOD_SONG_PATHS)
    #: :type: unqlite.Collection
    bad_song_paths_collection = Database.db.collection(database.DB_BAD_SONG_PATHS)

    remove_field_from_collection(good_song_paths_collection, str(Database.SongPathODBC.FFTS_BUILT))
    remove_field_from_collection(bad_song_paths_collection, str(Database.SongPathODBC.FFTS_BUILT))
