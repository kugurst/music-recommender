from file_store.cleaner import *
from file_store.store import FileStore


def test_clean_file_names():
    clean_file_names(FileStore.bad_songs_dir)
    clean_file_names(FileStore.good_songs_dir)
