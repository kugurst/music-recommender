import logging
import multiprocessing
import os
import queue
import time

import numpy as np
import scipy
from scipy import signal
from scipy.fftpack import fft

from file_store import database
from file_store.database import Database
from file_store.store import FileStore
from music.song import Song

__logger__ = logging.getLogger(__name__)

_store_batch_size = 10
_max_result_queue_len = 300

__all__ = ["build_song_indexes", "build_song_samples", "NUMBER_OF_RANDOM_SAMPLES", "SECONDS_PER_RANDOM_SAMPLES"]

NUMBER_OF_RANDOM_SAMPLES = 30
SECONDS_PER_RANDOM_SAMPLES = 5
FRAME_RATE = 44100


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
            every_hash.update({Database.SongPathODBC.SONG_HASH.get_value(elem) for elem in all_elements})
            for dir_name, subdir_list, file_list in os.walk(root_dir, topdown=False):
                for fn in file_list:
                    song_path = os.path.join(root_dir, dir_name, fn)
                    song_path_hash = Song.compute_path_hash(song_path.encode())
                    if not song_path_hash:
                        raise ValueError("Failed to hash path: [{}]".format(song_path))

                    if song_path_hash not in every_hash:
                        record = Database.SongPathODBC.initialize_record(song_path_hash, song_path)
                        collection.store(record)
                        __logger__.debug("Adding song: [{}]".format(song_path))
                        every_hash.add(song_path_hash)

    __logger__.debug("[{}] good songs".format(len(good_songs_collection)))
    __logger__.debug("[{}] bad songs".format(len(bad_songs_collection)))


# <editor-fold desc="build samples from the songs">
def build_song_samples():
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
                if not Database.SongPathODBC.SAMPLES_BUILT.get_value(elem):
                    input_queue.put((elem[database.DB_RECORD_FIELD], is_good_song,
                                     Database.SongPathODBC.SONG_HASH.get_value(elem),
                                     Database.SongPathODBC.SONG_PATH.get_value(elem)
                                     ))

        del good_song_paths_collection
        del bad_song_paths_collection
        del good_song_spc_collection
        del bad_song_spc_collection

        # Start the song samplers
        samplers = []
        for _ in range(multiprocessing.cpu_count()):
            p = multiprocessing.Process(target=_sample_songs, args=(input_queue, result_queue))
            samplers.append(p)
            p.start()

        # Store results
        _store_sampled_songs(result_queue, multiprocessing.cpu_count())

        # Wait for everything to terminate
        for p in samplers:
            p.join()


def _sample_songs(input_queue, result_queue):
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s]%(processName)s:%(name)s:%(levelname)s - %(message)s')

    while True:
        try:
            id, is_good_song, song_hash, song_path = input_queue.get_nowait()
        except queue.Empty:
            break
        else:
            song = Song(song_path)

            __logger__.debug("Sampling sequences for song: [{}]".format(song.path))

            samples = song.song.get_array_of_samples()
            frame_rate = song.song.frame_rate
            left_samples = np.array(samples[0::2])
            right_samples = np.array(samples[1::2])

            left_samples_sets, right_samples_sets = _select_random_samples_sets(
                NUMBER_OF_RANDOM_SAMPLES, SECONDS_PER_RANDOM_SAMPLES, left_samples, right_samples, frame_rate
            )

            result_queue.put((id,
                              song_hash,
                              is_good_song,
                              Database.SongRepresentationODBC.serialize_object(left_samples_sets),
                              Database.SongRepresentationODBC.serialize_object(right_samples_sets),
                              ))

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


def _store_sampled_songs(result_queue, worker_count):
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
            __logger__.debug("Storer: retrieveing next song")
            db_song_id, song_hash, is_good_song, left_samples_sets, right_samples_sets = result_queue.get()
            if songs_written == 0:
                __logger__.debug("Storer: new database batch set".format(db_song_id))
                Database.db.begin()
            __logger__.debug("Storer: got song [{}]".format(db_song_id))
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

            db_song = song_paths_collection.fetch(db_song_id)

            __logger__.debug("Storer: Song [{}] is named [{}]".format(
                db_song_id, os.path.basename(Database.SongPathODBC.SONG_PATH.get_value(db_song))
            ))

            record = Database.SongRepresentationODBC.initialize_record(
                song_hash, left_samples_sets, right_samples_sets
            )
            song_representation_collection.store(record)
            Database.SongPathODBC.SAMPLES_BUILT.set_value(db_song, True)
            song_paths_collection.update(db_song[database.DB_RECORD_FIELD], db_song)

            songs_written += 1

            if songs_written == _store_batch_size:
                songs_written = 0
                __logger__.debug("Storer: close database batch set".format(db_song_id))
                Database.db.commit()

            __logger__.debug("Storer: songs waiting to be written: [{}]".format(result_queue.qsize()))

    Database.db.commit()
# </editor-fold>
