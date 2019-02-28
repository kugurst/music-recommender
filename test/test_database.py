import pickle

from file_store import database
from file_store.database import *
from file_store.database import SongSamplesFeatureDB


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


def test_get_sample_song():
    #: :type: unqlite.Collection
    good_song_samples_collection = SongSamplesDatabase.db.collection(database.DB_GOOD_SONGS)

    record = good_song_samples_collection.fetch(0)
    with open("sample_song_record.pickle", 'wb') as pickle_file:
        pickle.dump(record, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def test_SongSamplesLVLDatabase():
    with open("sample_song_record.pickle", 'rb') as pickle_file:
        record = pickle.load(pickle_file)

    del record[database.DB_RECORD_FIELD]

    print(SongSamplesLVLDatabase.SongSamplesIndex.len())
    # print(SongSamplesLVLDatabase.SongSamplesIndex.all())

    print(SongSamplesLVLDatabase.SongSamplesIndex.fetch(0))

    # SongSamplesLVLDatabase.SongSamplesIndex.begin()
    # SongSamplesLVLDatabase.SongSamplesIndex.store(record)
    # SongSamplesLVLDatabase.SongSamplesIndex.commit()


def test_load_sample_features_database():
    SongSamplesFeatureDB.get_db()
    print(SongSamplesFeatureDB.get_feature(0, chosen_samples=set(range(30))))
