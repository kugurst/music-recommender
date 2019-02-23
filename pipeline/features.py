import atexit
import logging
import os
import random
import tempfile

import librosa
import numpy as np
import scipy
from scipy import io
from scipy.io import wavfile

from file_store import database
from file_store.database import *

__logger__ = logging.getLogger(__name__)

__all__ = ["generate_audio_sample", "compute_features"]

_audio_amplitude_max = 32767
hop_length = 2048


def compute_features(song_index, sample_index=None, song_file_or_path=None):
    left_samples_set, right_samples_set, song_info_record = get_samples(song_index, sample_index)
    # generate_audio_sample(song_index, sample_index, song_file_or_path, delete_on_exit=False)

    # Normalize the audio
    mono_mix = np.average([[left_samples_set, right_samples_set]], axis=1)[0]
    mono_mix *= _audio_amplitude_max / max(abs(mono_mix))
    mono_mix = mono_mix.astype(np.float32)

    sample_rate = SongInfoDatabase.SongInfoODBC.SONG_SAMPLE_RATE.get_value(song_info_record)

    # tempo
    flux = librosa.onset.onset_strength(y=mono_mix, sr=sample_rate, hop_length=hop_length)
    tempo = librosa.beat.tempo(onset_envelope=flux, sr=sample_rate, hop_length=hop_length)[0]

    # rolloff
    rolloff = librosa.feature.spectral_rolloff(y=mono_mix, sr=sample_rate, hop_length=hop_length)

    # Chroma
    n_fft = 2**12
    S = librosa.stft(y=mono_mix, n_fft=n_fft, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, fmin=31,
                                                 n_bands=9)

    mel = librosa.feature.melspectrogram(y=mono_mix, sr=sample_rate, S=S, n_fft=n_fft, hop_length=hop_length)

    # y_harmonic, y_percussive = librosa.effects.hpss(mono_mix)

    pass


def delete_if_present(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)


def get_samples(song_index, sample_index=None):
    try:
        get_samples.gsic
        get_samples.bsic
    except AttributeError:
        #: :type: unqlite.Collection
        get_samples.gsic = SongInfoDatabase.db.collection(database.DB_GOOD_SONGS)
        #: :type: unqlite.Collection
        get_samples.bsic = SongInfoDatabase.db.collection(database.DB_BAD_SONGS)

    song_sample_record = SongSamplesLVLDatabase.SongSamplesIndex.fetch(song_index)
    is_good_song = SongSamplesLVLDatabase.SongSamplesODBC.SONG_IS_GOOD.get_value(song_sample_record)
    song_info_collection = get_samples.gsic if is_good_song else get_samples.bsic
    song_info_record = song_info_collection.fetch(SongSamplesLVLDatabase.SongSamplesODBC.SONG_INFO_ID.get_value(
        song_sample_record))

    if sample_index is None:
        sample_index = random.randint(0, len(SongSamplesLVLDatabase.SongSamplesODBC.SONG_SAMPLES_LEFT.get_value(
            song_sample_record, True)) - 1)

    left_samples_set, right_samples_set = \
        SongSamplesLVLDatabase.SongSamplesODBC.SONG_SAMPLES_LEFT.get_value(song_sample_record)[sample_index], \
        SongSamplesLVLDatabase.SongSamplesODBC.SONG_SAMPLES_RIGHT.get_value(song_sample_record)[sample_index]

    return left_samples_set, right_samples_set, song_info_record


def generate_audio_sample(song_index, sample_index=None, file_or_path=None, delete_on_exit=True):
    left_samples_set, right_samples_set, song_info_record = get_samples(song_index, sample_index)

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
