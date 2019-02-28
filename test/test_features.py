import copy
import multiprocessing
import os
import regex
import random

import numpy as np
import psutil
import pyarrow
import transaction

from file_store import database
from file_store.database import *
from pipeline.features import *


def test_blah():
    songs, _ = SongSampleZODBDatabase.get_songs(True)
    print(len(songs))


def test_iterate_over_all_features():
        good_songs, bad_songs = [], []
        good_collection = SongInfoDatabase.db.collection(database.DB_GOOD_SONGS)
        bad_collection = SongInfoDatabase.db.collection(database.DB_BAD_SONGS)
        total_len = len(good_collection) + len(bad_collection)
        for collection, song_bin, is_good_song in [(good_collection, good_songs, True),
                                                   (bad_collection, bad_songs, False)]:
            for idx in range(len(collection)):
                record = collection.fetch(idx)
                song_sample_id = SongInfoDatabase.SongInfoODBC.SONG_SAMPLES_UNQLITE_ID.get_value(record)
                if song_sample_id >= total_len - 1:
                    continue

                if is_good_song:
                    good_songs.append((song_sample_id, 1))
                else:
                    bad_songs.append((song_sample_id, 0))

        songs, conn = SongSampleZODBDatabase.get_songs(True)
        for song_id, is_good_song in good_songs + bad_songs:
            song = songs.get(song_id)
            print(song.get_samples_left()[0].shape)


def test_compute_features():
    gsrc = SongInfoDatabase.db.collection(database.DB_GOOD_SONGS)
    song_index = random.randint(0, len(gsrc) - 1)

    features = compute_features(123)
    print(features.normalize_mel().shape)
    print(features.normalize_contrast().shape)
    print(features.normalize_chroma().shape)
    print(features.normalize_flux().shape)
    print(features)


def test_generate_audio_sample():
    gsrc = SongInfoDatabase.db.collection(database.DB_GOOD_SONGS)
    generate_audio_sample(random.randint(0, len(gsrc) - 1), delete_on_exit=False)


def test_load_samples(should_compute_features=True):
    songs, _ = SongSampleZODBDatabase.get_songs(True)
    queues = np.array_split(list(songs.keys()), multiprocessing.cpu_count())
    transaction.abort()
    SongSampleZODBDatabase.close_db()
    del songs

    with multiprocessing.Manager() as manager:
        processes = []
        for idx in range(multiprocessing.cpu_count()):
            p = multiprocessing.Process(target=_load_samples, args=(set(queues[idx]), should_compute_features))
            processes.append(p)
            p.start()

        for idx in range(multiprocessing.cpu_count()):
            p = processes[idx]
            p.join()


def _load_samples(song_queue, should_compute_features=False):
    songs, _ = SongSampleZODBDatabase.get_songs(True)
    for song_idx in songs:
        if song_idx not in song_queue:
            continue

        song_record = songs[song_idx]
        if not should_compute_features:
            song_record_copy = SongSamplesPickled(song_hash=song_record.song_hash, info_id=song_record.info_id,
                                                  is_good_song=song_record.is_good_song,
                                                  samples_left=song_record.get_samples_left(),
                                                  samples_right=song_record.get_samples_right(),
                                                  samples_indexes=song_record.samples_indexes)
            song_record_copy = copy.deepcopy(song_record_copy)
            with open("/capstone/tmp/sample_records/{}.pyarrow".format(song_idx), 'wb+') as f:
                f.write(pyarrow.serialize(song_record_copy.tolist()).to_buffer())
                # json.dump(song_record_copy.todict(), f)  # , protocol=pickle.HIGHEST_PROTOCOL)
            print("Loaded song index [{}]".format(song_idx))
        else:
            to_pickle = []
            to_pickle_small = []
            for sample_idx in range(len(song_record.samples_indexes)):
                features = compute_features(song_record, sample_idx, song_index_is_record=True)
                features.song_record_id = song_idx
                to_pickle.append(features.todict())
                to_pickle_small.append(features.tosmalldict())

            with open("/capstone/tmp_ramfs/features/{}.pyarrow".format(song_idx), 'wb+') as f:
                f.write(pyarrow.serialize(to_pickle).to_buffer())
                # json.dump(song_record_copy.todict(), f)  # , protocol=pickle.HIGHEST_PROTOCOL)
            with open("/capstone/tmp_ramfs/features/{}_small.pyarrow".format(song_idx), 'wb+') as f:
                f.write(pyarrow.serialize(to_pickle_small).to_buffer())
                # json.dump(song_record_copy.todict(), f)  # , protocol=pickle.HIGHEST_PROTOCOL)
            print("Computed features for song index [{}]".format(song_idx))

        if song_idx % 15 == 14:
            transaction.abort()


def test_load_all_features():
    db_dir = os.environ.get(database._SONG_SAMPLES_PYARROW_DATABASE_FN_ENV)
    contents = os.listdir(db_dir)

    db = dict()
    idx_regex = regex.compile(r"(\d+).*?\.pyarrow", flags=regex.IGNORECASE)

    for content in contents:
        idx = int(idx_regex.match(content).group(1))
        with open(os.path.join(db_dir, content), 'rb') as f:
            feature_list = pyarrow.deserialize(f.read())
            feature_list = [Feature.fromdict(feature) for feature in feature_list]
            db[idx] = feature_list
        pass

    process = psutil.Process(os.getpid())
    print(process.memory_info().rss)
