import base64
import hashlib
import os

from pydub import AudioSegment

from util import os_utils

__all__ = ["Song"]


_FFMPEG_ENV = "FFMPEG"

try:
    AudioSegment.ffmpeg = os.environ[_FFMPEG_ENV]
except KeyError:
    raise RuntimeError("[{}] is not defined. Specify the path to the ffmpeg binary in this variable".format(
        _FFMPEG_ENV))


class Song(object):
    def __init__(self, path):
        self.__hash = None
        self.__path = path

        # self.song = aifc.open(path, 'rb')
        if os_utils.is_windows():
            self.song = AudioSegment.from_file("\\\\?\\" + path)
        else:
            self.song = AudioSegment.from_file(path)

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
