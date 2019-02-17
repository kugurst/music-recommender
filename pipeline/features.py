import numpy as np
import scipy

from scipy import io
from scipy.io import wavfile


def compute_features(left_samples_set, right_samples_set):
    pass


def generate_audio_sample(left_samples_set, right_samples_set, sample_rate, file_name):
    scipy.io.wavfile.write(file_name, sample_rate, np.ndarray([left_samples_set, right_samples_set], dtype=np.int16))
