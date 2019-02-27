import atexit
import logging
import math
import os
import pickle
import random
import tempfile
import warnings

import librosa
import numpy as np
import scipy
import sklearn
import transaction
from scipy import io
from scipy.io import wavfile

from file_store import database
from file_store.database import *
from pre_process.audio_representation import *

__logger__ = logging.getLogger(__name__)

__all__ = ["generate_audio_sample", "compute_features", "Feature", "TEMPO_SHAPE", "FLUX_SHAPE", "ROLLOFF_SHAPE",
           "MEL_SHAPE", "CONTRAST_SHAPE", "TONNETZ_SHAPE", "CHROMA_SHAPE", "HPSS_SHAPE", "RMS_SHAPE"]

_AUDIO_AMPLITUDE_MAX = 32767
_MAX_READS_BEFORE_ABORT = 30
HOP_LENGTH = 2**15
N_FFT = 2**12
N_MELS = 128


SAMPLE_RATE = 44100
TEMPO_SHAPE = (1,)
FLUX_SHAPE = (int(math.ceil(SECONDS_PER_RANDOM_SAMPLES * SAMPLE_RATE / HOP_LENGTH)),)
ROLLOFF_SHAPE = (int(math.ceil(SECONDS_PER_RANDOM_SAMPLES * SAMPLE_RATE / HOP_LENGTH)),)
MEL_SHAPE = (N_MELS, int(math.ceil(SECONDS_PER_RANDOM_SAMPLES * SAMPLE_RATE / HOP_LENGTH)),)
CONTRAST_SHAPE = (10, int(math.ceil(SECONDS_PER_RANDOM_SAMPLES * SAMPLE_RATE / HOP_LENGTH)),)
TONNETZ_SHAPE = (6, int(math.ceil(SECONDS_PER_RANDOM_SAMPLES * SAMPLE_RATE / HOP_LENGTH)),)
CHROMA_SHAPE = (12, int(math.ceil(SECONDS_PER_RANDOM_SAMPLES * SAMPLE_RATE / HOP_LENGTH)),)
HPSS_SHAPE = ((N_FFT >> 1) + 1, int(math.ceil(SECONDS_PER_RANDOM_SAMPLES * SAMPLE_RATE / HOP_LENGTH)), 2,)
RMS_SHAPE = (int(math.ceil(SECONDS_PER_RANDOM_SAMPLES * SAMPLE_RATE / HOP_LENGTH)),)
TEMPO_MAX = 160


class Feature(object):
    def __init__(self, sample_index, flux=None, tempo=None, rolloff=None, fft=None, fft_complex=None, fft_bins=None,
                 contrast=None, mel=None, mel_bins=None, y_harmonic=None, y_percussive=None, rms=None, chroma=None,
                 tonnetz=None):
        self.sample_index = sample_index
        self.flux = flux
        self.tempo = tempo
        self.rolloff = rolloff
        self.fft = fft
        self.fft_complex = fft_complex
        self.fft_bins = fft_bins
        self.contrast = contrast
        self.mel = mel
        self.mel_bins = mel_bins
        self.y_harmonic = y_harmonic
        self.y_percussive = y_percussive
        self.rms = rms
        self.chroma = chroma
        self.tonnetz = tonnetz

    def normalize_tempo(self):
        return self.tempo / TEMPO_MAX

    def normalize_flux(self):
        real_flux = np.abs(self.flux)
        max_flux = np.max(real_flux)
        if max_flux == 0:
            return real_flux
        return real_flux / max_flux

    def normalize_rolloff(self):
        return sklearn.preprocessing.normalize(self.rolloff, axis=1)[0]

    def normalize_mel(self):
        real_mel = np.abs(self.mel)
        max_mel = np.max(real_mel)
        if max_mel == 0:
            return real_mel
        return real_mel / max_mel

    def normalize_contrast(self):
        real_contrast = np.abs(self.contrast)
        max_contrast = np.max(real_contrast)
        if max_contrast == 0:
            return real_contrast
        return real_contrast / max_contrast

    def normalize_tonnetz(self):
        real_tonnetz = np.abs(self.tonnetz)
        max_tonnetz = np.max(real_tonnetz)
        if max_tonnetz == 0:
            return real_tonnetz
        return real_tonnetz / np.max(real_tonnetz)

    def normalize_chroma(self):
        real_chroma = np.abs(self.chroma)
        max_chroma = np.max(real_chroma)
        if max_chroma == 0:
            return real_chroma
        return real_chroma / np.max(self.chroma)

    def normalize_hpss(self):
        magnitude_h = np.abs(self.y_harmonic)
        magnitude_p = np.abs(self.y_percussive)

        max_h = np.max(magnitude_h)
        max_p = np.max(magnitude_p)

        if max_h == 0:
            res_h = magnitude_h
        else:
            res_h = magnitude_h / max_h

        if max_p == 0:
            res_p = magnitude_p
        else:
            res_p = magnitude_p / max_p

        return np.dstack((res_h, res_p))

    def normalize_rms(self):
        # return (self.rms / np.max(np.abs(self.rms)))[0]
        return sklearn.preprocessing.normalize(self.rms, axis=1)[0]

    def compute_fractional_rms_energy(self, lower_frequency=100, upper_frequency=300):
        below_freq = self.fft_bins <= lower_frequency
        above_freq = self.fft_bins >= upper_frequency

        below_freq = np.tile(below_freq, [self.fft_complex.shape[1], 1]).T
        above_freq = np.tile(above_freq, [self.fft_complex.shape[1], 1]).T

        fft_masked = np.ma.masked_array(self.fft_complex, mask=below_freq, fill_value=0).filled()
        fft_masked = np.ma.masked_array(fft_masked, mask=above_freq, fill_value=0).filled()
        rms_masked = librosa.feature.rms(S=fft_masked, hop_length=HOP_LENGTH, frame_length=HOP_LENGTH)

        if np.max(self.rms) == 0:
            return self.rms[0]

        with np.errstate(divide='ignore', invalid='ignore'):
            res = (rms_masked / self.rms)[0]
            res[~ np.isfinite(res)] = 0  # -inf inf NaN
        # if np.isnan(res).any() or np.isfinite(res).any():
        #     pass
        return res


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
        mono_mix *= _AUDIO_AMPLITUDE_MAX / max_sample
    mono_mix = mono_mix.astype(np.float32)

    sample_rate = SongInfoDatabase.SongInfoODBC.SONG_SAMPLE_RATE.get_value(song_info_record)

    # tempo
    flux = librosa.onset.onset_strength(y=mono_mix, sr=sample_rate, hop_length=HOP_LENGTH)
    tempo = librosa.beat.tempo(onset_envelope=flux, sr=sample_rate, hop_length=HOP_LENGTH)[0]

    # rolloff
    rolloff = librosa.feature.spectral_rolloff(y=mono_mix, sr=sample_rate, hop_length=HOP_LENGTH)

    # Chroma
    S = librosa.stft(y=mono_mix, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_bins = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
    D = np.abs(S) ** 2
    contrast = librosa.feature.spectral_contrast(S=S, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, fmin=31,
                                                 n_bands=9)

    mel = librosa.feature.melspectrogram(y=mono_mix, sr=sample_rate, S=D, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                         fmax=(HOP_LENGTH >> 1), n_mels=N_MELS, fmin=0)
    mel_bins = librosa.core.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=(HOP_LENGTH >> 1))

    y_harmonic, y_percussive = librosa.decompose.hpss(S=S)

    rms = librosa.feature.rms(S=S, hop_length=HOP_LENGTH, frame_length=HOP_LENGTH)

    chroma = librosa.feature.chroma_stft(S=S, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)

    tonnetz = librosa.feature.tonnetz(y=mono_mix, sr=sample_rate, chroma=chroma)

    feature = Feature(flux=flux, tempo=tempo, rolloff=rolloff, fft=np.abs(S), fft_complex=S, contrast=contrast, mel=mel,
                      mel_bins=mel_bins, y_harmonic=y_harmonic, y_percussive=y_percussive, rms=rms, chroma=chroma,
                      tonnetz=tonnetz, sample_index=sample_index, fft_bins=S_bins)
    return feature


def delete_if_present(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)


def get_samples_zodb(song_index, sample_index=None, try_exclude_samples=None):
    try:
        get_samples.gsic
        get_samples.bsic
        get_samples.songs

        get_samples.abort_count
    except AttributeError:
        #: :type: unqlite.Collection
        get_samples.gsic = SongInfoDatabase.db.collection(database.DB_GOOD_SONGS)
        #: :type: unqlite.Collection
        get_samples.bsic = SongInfoDatabase.db.collection(database.DB_BAD_SONGS)
        get_samples.songs, _ = SongSampleZODBDatabase.get_songs(True)

        get_samples.abort_count = 0

    #: :type: file_store.database.SongSamplesZODBPersist
    song_sample_record = get_samples.songs[song_index]

    song_info_collection = get_samples.gsic if song_sample_record.is_good_song else get_samples.bsic
    song_info_record = song_info_collection.fetch(song_sample_record.info_id)

    left_samples_sets, right_samples_sets = \
        song_sample_record.get_samples_left(), song_sample_record.get_samples_right()

    if get_samples.abort_count % _MAX_READS_BEFORE_ABORT == 0:
        transaction.abort()

    get_samples.abort_count += 1

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
