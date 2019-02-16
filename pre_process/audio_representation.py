import base64
import hashlib
import logging
import pickle
import queue

import numpy as np
import os
import multiprocessing
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from file_store import database
from file_store.database import Database
from file_store.store import FileStore
from music.song import Song

__logger__ = logging.getLogger(__name__)

_store_batch_size = 10
_max_result_queue_len = 300

__all__ = ["build_song_indexes", "build_song_representation", "NUMBER_OF_RANDOM_SAMPLES", "SECONDS_PER_RANDOM_SAMPLES"]

NUMBER_OF_RANDOM_SAMPLES = 30
SECONDS_PER_RANDOM_SAMPLES = 5


def build_song_indexes():
    #: :type: unqlite.Collection
    good_songs_collection = Database.db.collection(database.DB_GOOD_SONG_PATHS)
    #: :type: unqlite.Collection
    bad_songs_collection = Database.db.collection(database.DB_BAD_SONG_PATHS)

    for collection in [good_songs_collection, bad_songs_collection]:
        if not collection.exists():
            collection.create()

    every_hash = set()
    with Database.db.transaction():
        for collection, root_dir in [(good_songs_collection, FileStore.good_songs_dir),
                                     (bad_songs_collection, FileStore.bad_songs_dir)]:
            all_elements = collection.all()
            every_hash.update({elem[Database.SongPathODBC.SONG_HASH.value].decode('ascii') for elem in all_elements})
            for dir_name, subdir_list, file_list in os.walk(root_dir, topdown=False):
                for fn in file_list:
                    song_path = os.path.join(root_dir, dir_name, fn)
                    song_path_hash = Song.compute_path_hash(song_path.encode())
                    if not song_path_hash:
                        raise ValueError("Failed to hash path: [{}]".format(song_path))

                    if song_path_hash not in every_hash:
                        collection.store({Database.SongPathODBC.SONG_HASH.value: song_path_hash,
                                          Database.SongPathODBC.SONG_PATH.value: song_path,
                                          Database.SongPathODBC.REPRESENTATION_BUILT.value: False})
                        __logger__.debug("Adding song: [{}]".format(song_path))
                        every_hash.add(song_path_hash)

    __logger__.debug("[{}] good songs".format(len(good_songs_collection)))
    __logger__.debug("[{}] bad songs".format(len(bad_songs_collection)))


# <editor-fold desc="build spectrograms and random ranges">
def build_song_representation():
    #: :type: unqlite.Collection
    good_song_paths_collection = Database.db.collection(database.DB_GOOD_SONG_PATHS)
    #: :type: unqlite.Collection
    bad_song_paths_collection = Database.db.collection(database.DB_BAD_SONG_PATHS)

    #: :type: unqlite.Collection
    good_song_spc_collection = Database.db.collection(database.DB_GOOD_SONG_REPRESENTATIONS)
    #: :type: unqlite.Collection
    bad_song_spc_collection = Database.db.collection(database.DB_BAD_SONG_REPRESENTATIONS)

    for collection in [good_song_spc_collection, bad_song_spc_collection]:
        if not collection.exists():
            collection.create()

    with multiprocessing.Manager() as manager:
        input_queue = manager.Queue()
        result_queue = manager.Queue(_max_result_queue_len)

        for song_paths_collection, is_good_song in [(good_song_paths_collection, True),
                                                    (bad_song_paths_collection, False)]:
            all_elements = song_paths_collection.all()
            for elem in all_elements:
                if not elem[Database.SongPathODBC.REPRESENTATION_BUILT.value]:
                    input_queue.put((elem[database.DB_RECORD_FIELD], is_good_song,
                                     elem[Database.SongPathODBC.SONG_HASH.value].decode('utf-8'),
                                     elem[Database.SongPathODBC.SONG_PATH.value].decode('utf-8')))

        del good_song_paths_collection
        del bad_song_paths_collection
        del good_song_spc_collection
        del bad_song_spc_collection

        # Start the song representers
        representers = []
        for _ in range(multiprocessing.cpu_count()):
            p = multiprocessing.Process(target=_represent_songs, args=(input_queue, result_queue))
            representers.append(p)
            p.start()

        # Store results
        _store_represented_songs(result_queue, multiprocessing.cpu_count())

        # Wait for everything to terminate
        for p in representers:
            p.join()


def _represent_songs(input_queue, result_queue):
    while True:
        try:
            id, is_good_song, song_hash, song_path = input_queue.get_nowait()
        except queue.Empty:
            break
        else:
            song = Song(song_path)

            print("Computing spectrogram for: [{}]".format(song.path))

            samples = song.song.get_array_of_samples()
            frame_rate = song.song.frame_rate
            left_samples = np.array(samples[0::2])
            right_samples = np.array(samples[1::2])

            left_samples_sets, right_samples_sets = _select_random_samples_sets(
                NUMBER_OF_RANDOM_SAMPLES, SECONDS_PER_RANDOM_SAMPLES, left_samples, right_samples, frame_rate
            )

            left_samples_storage_format, right_samples_storage_format = [], []
            left_spectrogram_sets, right_spectrogram_sets = [], []
            for left_samples_set, right_samples_set in zip(left_samples_sets, right_samples_sets):
                f_right, t_right, Sxx_right = signal.spectrogram(right_samples_set, frame_rate, return_onesided=False)
                f_left, t_left, Sxx_left = signal.spectrogram(left_samples_set, frame_rate, return_onesided=False)

                left_spectrogram_sets.append(Sxx_left)
                right_spectrogram_sets.append(Sxx_right)
                left_samples_storage_format.append(left_samples_set)
                right_samples_storage_format.append(right_samples_set)

            result_queue.put((id,
                              song_hash,
                              is_good_song,
                              pickle.dumps(left_spectrogram_sets, pickle.HIGHEST_PROTOCOL),
                              pickle.dumps(right_spectrogram_sets, pickle.HIGHEST_PROTOCOL),
                              pickle.dumps(left_samples_storage_format, pickle.HIGHEST_PROTOCOL),
                              pickle.dumps(right_samples_storage_format, pickle.HIGHEST_PROTOCOL)))
    result_queue.put(1)


def _select_random_samples_sets(num_selection, seconds_per_selection, left_samples, right_samples, frame_rate):
    assert len(left_samples) == len(right_samples)

    samples_per_selection = seconds_per_selection * frame_rate
    starting_sample_indexes = set(np.random.randint(0, max(0, len(left_samples) - samples_per_selection) + 1,
                                                    size=num_selection))

    left_samples_sets, right_samples_sets = [], []

    for starting_sample_index in starting_sample_indexes:
        ending_index = min(len(left_samples), starting_sample_index + samples_per_selection)
        left_samples_sets.append(left_samples[starting_sample_index:ending_index])
        right_samples_sets.append(right_samples[starting_sample_index:ending_index])
    return left_samples_sets, right_samples_sets


def _store_represented_songs(result_queue, worker_count):
    #: :type: unqlite.Collection
    good_song_paths_collection = Database.db.collection(database.DB_GOOD_SONG_PATHS)
    #: :type: unqlite.Collection
    bad_song_paths_collection = Database.db.collection(database.DB_BAD_SONG_PATHS)

    #: :type: unqlite.Collection
    good_song_representation_collection = Database.db.collection(database.DB_GOOD_SONG_REPRESENTATIONS)
    #: :type: unqlite.Collection
    bad_song_representation_collection = Database.db.collection(database.DB_BAD_SONG_REPRESENTATIONS)

    done_workers = 0
    songs_written = 0

    while True:
        try:
            print("Storer: retrieveing next song")
            id, song_hash, is_good_song, left_spectrogram_sets, right_spectrogram_sets, left_samples_storage_format, \
            right_samples_storage_format = \
                result_queue.get()
            if songs_written == 0:
                print("Storer: new database batch set".format(id))
                Database.db.begin()
            print("Storer: got song [{}]".format(id))
        except TypeError:
            done_workers += 1
            if done_workers == worker_count:
                break
        except queue.Empty:
            continue
        else:
            song_representation_collection = good_song_representation_collection if is_good_song else \
                bad_song_representation_collection
            song_paths_collection = good_song_paths_collection if is_good_song else bad_song_paths_collection

            db_song = song_paths_collection.fetch(id)

            print("Storer: Song [{}] is named [{}]".format(
                id, os.path.basename(db_song[Database.SongPathODBC.SONG_PATH.value].decode('utf-8'))
            ))

            song_representation_collection.store({
                Database.SongRepresentationODBC.SONG_HASH.value: song_hash,
                Database.SongRepresentationODBC.SONG_SPCS_LEFT.value: left_spectrogram_sets,
                Database.SongRepresentationODBC.SONG_SPCS_RIGHT.value: right_spectrogram_sets,
                Database.SongRepresentationODBC.SONG_SAMPLES_LEFT.value: left_samples_storage_format,
                Database.SongRepresentationODBC.SONG_SAMPLES_RIGHT.value: right_samples_storage_format,
            })
            db_song[Database.SongPathODBC.REPRESENTATION_BUILT.value] = True
            song_paths_collection.update(db_song[database.DB_RECORD_FIELD], db_song)

            songs_written += 1

            if songs_written == _store_batch_size:
                songs_written = 0
                print("Storer: close database batch set".format(id))
                Database.db.commit()

            print("Storer: songs waiting to be written: [{}]".format(result_queue.qsize()))

    Database.db.commit()
    # plt.pcolormesh(t, f, Sxx)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
# </editor-fold>


# <editor-fold desc="build song ffts">
def build_song_ffts():
    #: :type: unqlite.Collection
    good_song_paths_collection = Database.db.collection(database.DB_GOOD_SONG_PATHS)
    #: :type: unqlite.Collection
    bad_song_paths_collection = Database.db.collection(database.DB_BAD_SONG_PATHS)

    #: :type: unqlite.Collection
    good_song_spc_collection = Database.db.collection(database.DB_GOOD_SONG_REPRESENTATIONS)
    #: :type: unqlite.Collection
    bad_song_spc_collection = Database.db.collection(database.DB_BAD_SONG_REPRESENTATIONS)

    for collection in [good_song_spc_collection, bad_song_spc_collection]:
        if not collection.exists():
            collection.create()

    with multiprocessing.Manager() as manager:
        input_queue = manager.Queue()
        result_queue = manager.Queue(_max_result_queue_len)

        for song_paths_collection, is_good_song in [(good_song_paths_collection, True),
                                                    (bad_song_paths_collection, False)]:
            all_elements = song_paths_collection.all()
            for elem in all_elements:
                if not elem[Database.SongPathODBC.REPRESENTATION_BUILT.value]:
                    input_queue.put((elem[database.DB_RECORD_FIELD], is_good_song,
                                     elem[Database.SongPathODBC.SONG_HASH.value].decode('utf-8'),
                                     elem[Database.SongPathODBC.SONG_PATH.value].decode('utf-8')))

        del good_song_paths_collection
        del bad_song_paths_collection
        del good_song_spc_collection
        del bad_song_spc_collection

        # Start the song representers
        representers = []
        for _ in range(multiprocessing.cpu_count()):
            p = multiprocessing.Process(target=_represent_songs, args=(input_queue, result_queue))
            representers.append(p)
            p.start()

        # Store results
        _store_represented_songs(result_queue, multiprocessing.cpu_count())

        # Wait for everything to terminate
        for p in representers:
            p.join()


def _represent_songs(input_queue, result_queue):
    while True:
        try:
            id, is_good_song, song_hash, song_path = input_queue.get_nowait()
        except queue.Empty:
            break
        else:
            song = Song(song_path)

            print("Computing spectrogram for: [{}]".format(song.path))

            samples = song.song.get_array_of_samples()
            frame_rate = song.song.frame_rate
            left_samples = np.array(samples[0::2])
            right_samples = np.array(samples[1::2])

            left_samples_sets, right_samples_sets = _select_random_samples_sets(
                NUMBER_OF_RANDOM_SAMPLES, SECONDS_PER_RANDOM_SAMPLES, left_samples, right_samples, frame_rate
            )

            left_samples_storage_format, right_samples_storage_format = [], []
            left_spectrogram_sets, right_spectrogram_sets = [], []
            for left_samples_set, right_samples_set in zip(left_samples_sets, right_samples_sets):
                f_right, t_right, Sxx_right = signal.spectrogram(right_samples_set, frame_rate, return_onesided=False)
                f_left, t_left, Sxx_left = signal.spectrogram(left_samples_set, frame_rate, return_onesided=False)

                left_spectrogram_sets.append(Sxx_left)
                right_spectrogram_sets.append(Sxx_right)
                left_samples_storage_format.append(left_samples_set)
                right_samples_storage_format.append(right_samples_set)

            result_queue.put((id,
                              song_hash,
                              is_good_song,
                              pickle.dumps(left_spectrogram_sets, pickle.HIGHEST_PROTOCOL),
                              pickle.dumps(right_spectrogram_sets, pickle.HIGHEST_PROTOCOL),
                              pickle.dumps(left_samples_storage_format, pickle.HIGHEST_PROTOCOL),
                              pickle.dumps(right_samples_storage_format, pickle.HIGHEST_PROTOCOL)))
    result_queue.put(1)


def _select_random_samples_sets(num_selection, seconds_per_selection, left_samples, right_samples, frame_rate):
    assert len(left_samples) == len(right_samples)

    samples_per_selection = seconds_per_selection * frame_rate
    starting_sample_indexes = set(np.random.randint(0, max(0, len(left_samples) - samples_per_selection) + 1,
                                                    size=num_selection))

    left_samples_sets, right_samples_sets = [], []

    for starting_sample_index in starting_sample_indexes:
        ending_index = min(len(left_samples), starting_sample_index + samples_per_selection)
        left_samples_sets.append(left_samples[starting_sample_index:ending_index])
        right_samples_sets.append(right_samples[starting_sample_index:ending_index])
    return left_samples_sets, right_samples_sets


def _store_represented_songs(result_queue, worker_count):
    #: :type: unqlite.Collection
    good_song_paths_collection = Database.db.collection(database.DB_GOOD_SONG_PATHS)
    #: :type: unqlite.Collection
    bad_song_paths_collection = Database.db.collection(database.DB_BAD_SONG_PATHS)

    #: :type: unqlite.Collection
    good_song_representation_collection = Database.db.collection(database.DB_GOOD_SONG_REPRESENTATIONS)
    #: :type: unqlite.Collection
    bad_song_representation_collection = Database.db.collection(database.DB_BAD_SONG_REPRESENTATIONS)

    done_workers = 0
    songs_written = 0

    while True:
        try:
            print("Storer: retrieveing next song")
            id, song_hash, is_good_song, left_spectrogram_sets, right_spectrogram_sets, left_samples_storage_format, \
            right_samples_storage_format = \
                result_queue.get()
            if songs_written == 0:
                print("Storer: new database batch set".format(id))
                Database.db.begin()
            print("Storer: got song [{}]".format(id))
        except TypeError:
            done_workers += 1
            if done_workers == worker_count:
                break
        except queue.Empty:
            continue
        else:
            song_representation_collection = good_song_representation_collection if is_good_song else \
                bad_song_representation_collection
            song_paths_collection = good_song_paths_collection if is_good_song else bad_song_paths_collection

            db_song = song_paths_collection.fetch(id)

            print("Storer: Song [{}] is named [{}]".format(
                id, os.path.basename(db_song[Database.SongPathODBC.SONG_PATH.value].decode('utf-8'))
            ))

            song_representation_collection.store({
                Database.SongRepresentationODBC.SONG_HASH.value: song_hash,
                Database.SongRepresentationODBC.SONG_SPCS_LEFT.value: left_spectrogram_sets,
                Database.SongRepresentationODBC.SONG_SPCS_RIGHT.value: right_spectrogram_sets,
                Database.SongRepresentationODBC.SONG_SAMPLES_LEFT.value: left_samples_storage_format,
                Database.SongRepresentationODBC.SONG_SAMPLES_RIGHT.value: right_samples_storage_format,
            })
            db_song[Database.SongPathODBC.REPRESENTATION_BUILT.value] = True
            song_paths_collection.update(db_song[database.DB_RECORD_FIELD], db_song)

            songs_written += 1

            if songs_written == _store_batch_size:
                songs_written = 0
                print("Storer: close database batch set".format(id))
                Database.db.commit()

            print("Storer: songs waiting to be written: [{}]".format(result_queue.qsize()))

    Database.db.commit()
    # plt.pcolormesh(t, f, Sxx)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
# </editor-fold>
