import aifc
import base64
import hashlib
import os

import numpy as np
import scipy
from pydub import AudioSegment
from scipy import signal
from scipy.fftpack import fft

from util import os_utils

__all__ = ["Song", "SongSpectrogram", "SongFFT"]


_FFMPEG_ENV = "FFMPEG"

try:
    AudioSegment.ffmpeg = os.environ[_FFMPEG_ENV]
except KeyError:
    raise RuntimeError("[{}] is not defined. Specify the path to the ffmpeg binary in this variable".format(
        _FFMPEG_ENV))


class Song(object):
    def __init__(self, path, use_audio_segment=False):
        self.__hash = None
        self.__path = path

        if os_utils.is_windows():
            path = "\\\\?\\" + path

        if use_audio_segment:
            self.song = AudioSegment.from_file(path)
        else:
            self.song = aifc.open(path, 'rb')

    @property
    def path(self):
        return self.__path

    def __hash__(self):
        if self.__hash is None:
            self.__hash = Song.compute_path_hash(self.__path.encode())

        return self.__hash

    @staticmethod
    def compute_path_hash(path):
        m = hashlib.sha512()
        m.update(path)
        return base64.urlsafe_b64encode(m.digest()).decode('ascii')


class SongSpectrogram(object):
    def __init__(self, frequency_series, time_series, spectrogram_series):
        self.frequency_series = frequency_series
        self.time_series = time_series
        self.spectrogram_series = spectrogram_series

    @staticmethod
    def compute_spectrogram(left_samples_sets, right_samples_sets, frame_rate):
        left_spectrograms, right_spectrograms = [], []
        for left_samples_set, right_samples_set in zip(left_samples_sets,right_samples_sets):
            f_right, t_right, Sxx_right = scipy.signal.spectrogram(right_samples_set, frame_rate, return_onesided=False)
            f_left, t_left, Sxx_left = scipy.signal.spectrogram(left_samples_set, frame_rate, return_onesided=False)

            left_spectrograms.append(SongSpectrogram(f_left, t_left, Sxx_left))
            right_spectrograms.append(SongSpectrogram(f_right, t_right, Sxx_right))
        return left_spectrograms, right_spectrograms


class SongFFT(object):
    def __init__(self, amplitude_series, frequency_bins):
        self.amplitude_series = amplitude_series
        self.frequency_bins = frequency_bins

    @staticmethod
    def compute_ffts(left_samples_sets, right_samples_sets, frame_rate):
        left_ffts, right_ffts = [], []
        for left_samples_set, right_samples_set in zip(left_samples_sets,right_samples_sets):
            # https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch04.html
            X_left = scipy.fftpack.fft(left_samples_set)
            X_right = scipy.fftpack.fft(right_samples_set)
            freqs_left = scipy.fftpack.fftfreq(len(left_samples_set))
            freqs_right = scipy.fftpack.fftfreq(len(right_samples_set))

            X_left = np.abs(X_left[:int(len(X_left) / 2)])
            X_right = np.abs(X_right[:int(len(X_right) / 2)])
            freqs_left = freqs_left[:int(len(freqs_left) / 2)]
            freqs_right = freqs_right[:int(len(freqs_right) / 2)]
            freqs_left = freqs_left * frame_rate
            freqs_right = freqs_right * frame_rate

            left_ffts.append(SongFFT(X_left, freqs_left))
            right_ffts.append(SongFFT(X_right, freqs_right))

        return left_ffts, right_ffts
