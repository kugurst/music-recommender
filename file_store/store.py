import os

from util.class_property import ClassProperty


__all__ = ["FileStore"]


class FileStore(object):
    @ClassProperty
    @classmethod
    def good_songs_dir(cls):
        return FileStore.__get_song_dir("GOOD_SONG_DIR")

    @ClassProperty
    @classmethod
    def bad_songs_dir(cls):
        return FileStore.__get_song_dir("BAD_SONG_DIR")

    @staticmethod
    def __get_song_dir(env_var):
        try:
            return os.environ[env_var]
        except KeyError:
            raise RuntimeError("[{}] is not defined. Specify the path to this song directory in the environment "
                               "variables".format(env_var))
