import logging
import multiprocessing
import os
import queue
import threading

import matplotlib.pyplot as plt
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

__all__ = ["build_song_indexes", "build_song_representation", "NUMBER_OF_RANDOM_SAMPLES", "SECONDS_PER_RANDOM_SAMPLES",
           "build_song_ffts"]

NUMBER_OF_RANDOM_SAMPLES = 15
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
                if not Database.SongPathODBC.REPRESENTATION_BUILT.get_value(elem):
                    input_queue.put((elem[database.DB_RECORD_FIELD], is_good_song,
                                     Database.SongPathODBC.SONG_HASH.get_value(elem),
                                     Database.SongPathODBC.SONG_PATH.get_value(elem)
                                     ))

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

            __logger__.debug("Computing spectrogram for: [{}]".format(song.path))

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

                left_spectrogram_sets.append((f_left, t_left, Sxx_left))
                right_spectrogram_sets.append((f_right, t_right, Sxx_right))
                left_samples_storage_format.append(left_samples_set)
                right_samples_storage_format.append(right_samples_set)

            result_queue.put((id,
                              song_hash,
                              is_good_song,
                              Database.SongRepresentationODBC.serialize_object(left_spectrogram_sets),
                              Database.SongRepresentationODBC.serialize_object(right_spectrogram_sets),
                              Database.SongRepresentationODBC.serialize_object(left_samples_storage_format),
                              Database.SongRepresentationODBC.serialize_object(right_samples_storage_format),
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
            __logger__.debug("Storer: retrieveing next song")
            db_song_id, song_hash, is_good_song, left_spectrogram_sets, right_spectrogram_sets, left_samples_storage_format, \
            right_samples_storage_format = \
                result_queue.get()
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
                song_hash, left_spectrogram_sets, right_spectrogram_sets, left_samples_storage_format,
                right_samples_storage_format
            )
            song_representation_collection.store(record)
            Database.SongPathODBC.REPRESENTATION_BUILT.set_value(db_song, True)
            song_paths_collection.update(db_song[database.DB_RECORD_FIELD], db_song)

            songs_written += 1

            if songs_written == _store_batch_size:
                songs_written = 0
                __logger__.debug("Storer: close database batch set".format(db_song_id))
                Database.db.commit()

            __logger__.debug("Storer: songs waiting to be written: [{}]".format(result_queue.qsize()))

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
    good_song_repr_collection = Database.db.collection(database.DB_GOOD_SONG_REPRESENTATIONS)
    #: :type: unqlite.Collection
    bad_song_repr_collection = Database.db.collection(database.DB_BAD_SONG_REPRESENTATIONS)

    for collection in [good_song_repr_collection, bad_song_repr_collection]:
        if not collection.exists():
            collection.create()

    with multiprocessing.Manager() as manager:
        input_queue = manager.Queue(multiprocessing.cpu_count() * 3)
        result_queue = manager.Queue(_max_result_queue_len)

        songs_to_compute_fft_for = dict()
        for song_paths_collection, is_good_song in [(good_song_paths_collection, True),
                                                    (bad_song_paths_collection, False)]:
            all_elements = song_paths_collection.all()
            for elem in all_elements:
                if not Database.SongPathODBC.FFTS_BUILT.get_value(elem):
                    songs_to_compute_fft_for[Database.SongPathODBC.SONG_HASH.get_value(elem)] = [
                        elem[database.DB_RECORD_FIELD], is_good_song,
                        Database.SongPathODBC.SONG_HASH.get_value(elem),
                        Database.SongPathODBC.SONG_PATH.get_value(elem)
                    ]

        del good_song_paths_collection
        del bad_song_paths_collection

        # Setup the FFT computers
        fft_processors = []
        # _compute_ffts(input_queue, result_queue)
        for _ in range(multiprocessing.cpu_count()):
            p = multiprocessing.Process(target=_compute_ffts, args=(input_queue, result_queue))
            fft_processors.append(p)

        _fft_feeder = threading.Thread(target=_feed_ffts, args=(input_queue, songs_to_compute_fft_for,
                                                                good_song_repr_collection, bad_song_repr_collection,
                                                                multiprocessing.cpu_count()))
        _fft_feeder.start()
        # Start the FFT computers
        for p in fft_processors:
            p.start()

        # Store results
        _store_song_ffts(result_queue, multiprocessing.cpu_count())

        # Wait for everything to terminate
        _fft_feeder.join()
        for p in fft_processors:
            p.join()


def _feed_ffts(input_queue, songs_to_compute_fft_for, good_song_collection, bad_song_collection, worker_count):
    for song_representation_collection in [good_song_collection, bad_song_collection]:
        for idx in range(len(song_representation_collection)):
            song_representation = song_representation_collection.fetch(idx)
            song_hash = Database.SongRepresentationODBC.SONG_HASH.get_value(song_representation)
            if song_hash not in songs_to_compute_fft_for:
                continue

            left_samples_sets = Database.SongRepresentationODBC.SONG_SAMPLES_LEFT.get_value(
                song_representation, True)
            right_samples_sets = Database.SongRepresentationODBC.SONG_SAMPLES_RIGHT.get_value(
                song_representation, True)
            song_path_id, is_good_song, song_hash, song_path = songs_to_compute_fft_for[song_hash]
            song_repr_id = song_representation[database.DB_RECORD_FIELD]
            __logger__.debug("_feed_ffts: feeding song [{}]".format(song_path))
            input_queue.put((song_path_id, song_repr_id, is_good_song, song_hash, song_path, left_samples_sets,
                             right_samples_sets))

    for _ in range(worker_count):
        input_queue.put(1)


def _compute_ffts(input_queue, result_queue):
    while True:
        try:
            song_path_id, song_repr_id, is_good_song, song_hash, song_path, left_samples_sets, right_samples_sets = \
                input_queue.get()
        except TypeError:
            break
        else:
            __logger__.debug("Computing FFTs for: [{}]".format(song_path))

            # Database.SongRepresentationODBC.deserialize_object(left_samples_sets)
            # Database.SongRepresentationODBC.deserialize_object(right_samples_sets)
            left_samples_sets = Database.SongRepresentationODBC.deserialize_object(left_samples_sets)
            right_samples_sets = Database.SongRepresentationODBC.deserialize_object(right_samples_sets)

            left_ffts = []
            right_ffts = []
            for left_samples_set, right_samples_set in zip(left_samples_sets, right_samples_sets):
                # https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch04.html
                X_left = scipy.fftpack.fft(left_samples_set)
                X_right = scipy.fftpack.fft(right_samples_set)
                freqs_left = scipy.fftpack.fftfreq(len(left_samples_set))
                freqs_right = scipy.fftpack.fftfreq(len(right_samples_set))

                X_left = np.abs(X_left[:int(len(X_left) / 2)])
                X_right = np.abs(X_right[:int(len(X_right) / 2)])
                freqs_left = freqs_left[:int(len(freqs_left) / 2)]
                freqs_right = freqs_right[:int(len(freqs_right) / 2)]
                freqs_left = freqs_left * FRAME_RATE
                freqs_right = freqs_right * FRAME_RATE

                left_ffts.append((X_left, freqs_left))
                right_ffts.append((X_right, freqs_right))

            result_queue.put((song_path_id,
                              song_repr_id,
                              song_hash,
                              is_good_song,
                              Database.SongRepresentationODBC.serialize_object(left_ffts),
                              Database.SongRepresentationODBC.serialize_object(right_ffts)
                              ))
    result_queue.put(1)


def _store_song_ffts(result_queue, worker_count):
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
            __logger__.debug("FFT Storer: retrieveing next song")
            song_path_id, song_repr_id, song_hash, is_good_song, left_ffts, right_ffts = result_queue.get()
            if songs_written == 0:
                __logger__.debug("FFT Storer: new database batch set".format(song_path_id))
                Database.db.begin()
            __logger__.debug("FFT Storer: got song [{}]".format(song_path_id))
        except TypeError:
            done_workers += 1
            if done_workers == worker_count:
                break
        except queue.Empty:
            continue
        else:
            song_repr_collection = good_song_representation_collection if is_good_song else \
                bad_song_representation_collection
            song_paths_collection = good_song_paths_collection if is_good_song else bad_song_paths_collection

            db_song_path = song_paths_collection.fetch(song_path_id)
            db_song_repr = song_repr_collection.fetch(song_repr_id)

            __logger__.debug("FFT Storer: Song [{}] is named [{}]".format(
                song_path_id, os.path.basename(Database.SongPathODBC.SONG_PATH.get_value(db_song_path))
            ))

            Database.SongPathODBC.FFTS_BUILT.set_value(db_song_path, True)
            song_paths_collection.update(db_song_path[database.DB_RECORD_FIELD], db_song_path)

            Database.SongRepresentationODBC.SONG_FFT_LEFT.set_value(db_song_repr, left_ffts)
            Database.SongRepresentationODBC.SONG_FFT_RIGHT.set_value(db_song_repr, right_ffts)
            song_repr_collection.update(db_song_repr[database.DB_RECORD_FIELD], db_song_repr)

            songs_written += 1

            if songs_written == _store_batch_size:
                __logger__.debug("FFT Storer: close database batch set".format(song_path_id))
                songs_written = 0
                Database.db.commit()

            __logger__.debug("FFT Storer: songs waiting to be written: [{}]".format(result_queue.qsize()))

    Database.db.commit()
    # plt.pcolormesh(t, f, Sxx)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
# </editor-fold>
