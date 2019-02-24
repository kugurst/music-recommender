import atexit
import logging
import os
import pickle
import random
import tempfile

import librosa
import numpy as np
import scipy
import transaction
from scipy import io
from scipy.io import wavfile

from file_store import database
from file_store.database import *
from pre_process.audio_representation import DatabaseBackend

__logger__ = logging.getLogger(__name__)

__all__ = ["generate_audio_sample", "compute_features", "Feature"]

_audio_amplitude_max = 32767
hop_length = 2048


class Feature(object):
    def __init__(self, sample_index, flux=None, tempo=None, rolloff=None, fft=None, contrast=None, mel=None,
                 y_harmonic=None, y_percussive=None, rms=None, chroma=None, tonnetz=None):
        self.sample_index = sample_index
        self.flux = flux
        self.tempo = tempo
        self.rolloff = rolloff
        self.fft = fft
        self.contrast = contrast
        self.mel = mel
        self.y_harmonic = y_harmonic
        self.y_percussive = y_percussive
        self.rms = rms
        self.chroma = chroma
        self.tonnetz = tonnetz


def compute_features(song_index, sample_index=None, song_file_or_path=None, try_exclude_samples=None,
                     backend=DatabaseBackend.ZODB):
    if backend == DatabaseBackend.ZODB:
        samples_func = get_samples_zodb
    else:
        samples_func = get_samples

    left_samples_set, right_samples_set, song_info_record, sample_index = samples_func(
        song_index, sample_index, try_exclude_samples=try_exclude_samples)
    # generate_audio_sample(song_index, sample_index, song_file_or_path, delete_on_exit=False)

    # Normalize the audio
    mono_mix = np.average([[left_samples_set, right_samples_set]], axis=1)[0]
    max_sample = max(abs(mono_mix))
    if max_sample != 0:
        mono_mix *= _audio_amplitude_max / max_sample
    mono_mix = mono_mix.astype(np.float32)

    sample_rate = SongInfoDatabase.SongInfoODBC.SONG_SAMPLE_RATE.get_value(song_info_record)

    # tempo
    flux = librosa.onset.onset_strength(y=mono_mix, sr=sample_rate, hop_length=hop_length)
    tempo = librosa.beat.tempo(onset_envelope=flux, sr=sample_rate, hop_length=hop_length)[0]

    # rolloff
    rolloff = librosa.feature.spectral_rolloff(y=mono_mix, sr=sample_rate, hop_length=hop_length)

    # Chroma
    n_fft = 2 ** 12
    S = librosa.stft(y=mono_mix, n_fft=n_fft, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, fmin=31,
                                                 n_bands=9)

    mel = librosa.feature.melspectrogram(y=mono_mix, sr=sample_rate, S=S, n_fft=n_fft, hop_length=hop_length)

    y_harmonic, y_percussive = librosa.decompose.hpss(S=S)

    rms = librosa.feature.rms(S=S, hop_length=hop_length, frame_length=hop_length)

    chroma = librosa.feature.chroma_stft(S=S, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)

    tonnetz = librosa.feature.tonnetz(y=mono_mix, sr=sample_rate, chroma=chroma)

    feature = Feature(flux=flux, tempo=tempo, rolloff=rolloff, fft=np.abs(S), contrast=contrast, mel=mel,
                      y_harmonic=y_harmonic, y_percussive=y_percussive, rms=rms, chroma=chroma, tonnetz=tonnetz,
                      sample_index=sample_index)
    return feature


def delete_if_present(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)


def get_samples_zodb(song_index, sample_index=None, try_exclude_samples=None):
    try:
        get_samples.gsic
        get_samples.bsic
        get_samples.songs
    except AttributeError:
        #: :type: unqlite.Collection
        get_samples.gsic = SongInfoDatabase.db.collection(database.DB_GOOD_SONGS)
        #: :type: unqlite.Collection
        get_samples.bsic = SongInfoDatabase.db.collection(database.DB_BAD_SONGS)
        get_samples.songs, _ = SongSampleZODBDatabase.get_songs(True)

    #: :type: file_store.database.SongSamplesZODBPersist
    song_sample_record = get_samples.songs[song_index]

    song_info_collection = get_samples.gsic if song_sample_record.is_good_song else get_samples.bsic
    song_info_record = song_info_collection.fetch(song_sample_record.info_id)

    left_samples_sets, right_samples_sets = \
        song_sample_record.get_samples_left(), song_sample_record.get_samples_right()
    transaction.abort()

    if sample_index is None:
        if try_exclude_samples:
            unselected = set(range(len(left_samples_sets))) - try_exclude_samples
            if unselected:
                sample_index = np.random.choice(list(unselected))
        # Still none? Either we've used all samples, or we haven't tried any
        if sample_index is None:
            sample_index = random.randint(0, len(left_samples_sets) - 1)

    left_samples_set, right_samples_set = left_samples_sets[sample_index], right_samples_sets[sample_index]

    return left_samples_set, right_samples_set, song_info_record, sample_index


def get_samples(song_index, sample_index=None, try_exclude_samples=None):
    try:
        get_samples.gsic
        get_samples.gssc
        get_samples.bsic
        get_samples.bssc
        get_samples.total_samples
    except AttributeError:
        #: :type: unqlite.Collection
        get_samples.gsic = SongInfoDatabase.db.collection(database.DB_GOOD_SONGS)
        #: :type: unqlite.Collection
        get_samples.bsic = SongInfoDatabase.db.collection(database.DB_BAD_SONGS)
        #: :type: unqlite.Collection
        get_samples.gssc = SongSamplesDatabase.db.collection(database.DB_GOOD_SONGS)
        #: :type: unqlite.Collection
        get_samples.bssc = SongSamplesDatabase.db.collection(database.DB_BAD_SONGS)
        get_samples.total_samples = len(get_samples.gsic) + len(get_samples.bsic)

    is_good_song = song_index < len(get_samples.gsic)

    song_info_collection = get_samples.gsic if is_good_song else get_samples.bsic
    if not is_good_song:
        song_index -= len(get_samples.gsic)
    song_sample_collection = get_samples.gssc if is_good_song else get_samples.bssc

    song_info_record = song_info_collection.fetch(song_index)
    song_record_id = SongInfoDatabase.SongInfoODBC.SONG_SAMPLES_UNQLITE_ID.get_value(song_info_record)
    if not is_good_song:
        song_record_id -= len(get_samples.gsic)
    song_sample_record = song_sample_collection.fetch(song_record_id)

    left_samples_sets, right_samples_sets = \
        SongSamplesDatabase.SongSamplesODBC.SONG_SAMPLES_LEFT.get_value(song_sample_record), \
        SongSamplesDatabase.SongSamplesODBC.SONG_SAMPLES_RIGHT.get_value(song_sample_record)

    if sample_index is None:
        if try_exclude_samples:
            unselected = set(range(len(left_samples_sets))) - try_exclude_samples
            if unselected:
                sample_index = np.random.choice(list(unselected))
        # Still none? Either we've used all samples, or we haven't tried any
        if sample_index is None:
            sample_index = random.randint(0, len(left_samples_sets) - 1)

    left_samples_set, right_samples_set = left_samples_sets[sample_index], right_samples_sets[sample_index]

    return left_samples_set, right_samples_set, song_info_record, sample_index


def generate_audio_sample(song_index, sample_index=None, file_or_path=None, delete_on_exit=True):
    left_samples_set, right_samples_set, song_info_record, sample_index = get_samples(song_index, sample_index)

    file_name = file_or_path
    if not file_or_path:
        file_or_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        file_name = file_or_path.name
        if delete_on_exit:
            atexit.register(delete_if_present, file_name)

    __logger__.debug("Generated [{}] containing sample index [{}] of [{}][{}]".format(
        file_name, sample_index, SongInfoDatabase.SongInfoODBC.SONG_PATH.get_value(song_info_record), song_index))
    _generate_audio_sample(left_samples_set, right_samples_set,
                           SongInfoDatabase.SongInfoODBC.SONG_SAMPLE_RATE.get_value(song_info_record), file_or_path)

    return file_name


def _generate_audio_sample(left_samples_set, right_samples_set, sample_rate, file_name):
    data = np.dstack((left_samples_set, right_samples_set))[0]
    scipy.io.wavfile.write(file_name, sample_rate, data)
