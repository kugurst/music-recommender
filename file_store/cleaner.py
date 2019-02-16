import logging
import os
import re
import shutil

from unidecode import unidecode

__logger__ = logging.getLogger(__name__)

_character_rename_map = {
    "'": "",
    '"': "",
    '?': "",
    "<": "",
    ">": "",
    "*": "",
    "|": "",
    "(": "",
    ")": ""
}

__all__ = ["clean_file_names"]


def clean_file_names(root_dir):
    for dir_name, subdir_list, file_list in os.walk(root_dir, topdown=False):
        for fn in file_list:
            song_path = os.path.join(root_dir, dir_name, fn)
            new_song_path = song_path

            new_song_path = unidecode(new_song_path)
            for character in _character_rename_map:
                new_song_path = new_song_path.replace(character, _character_rename_map[character])

            new_song_paths = re.split('/|\\\\', new_song_path)
            for idx, path in enumerate(new_song_paths):
                new_song_paths[idx] = path.strip().strip('...')
                if idx != 0:
                    new_song_paths[idx] = new_song_paths[idx].replace(':', "")
            new_song_path = os.path.sep.join(new_song_paths)

            if new_song_path != song_path:
                new_parent_dir = os.path.dirname(new_song_path)
                if not os.path.exists(new_parent_dir):
                    os.makedirs(new_parent_dir)
                if os.path.exists(new_song_path):
                    raise ValueError("Wanted to copy [{}] to [{}], but destination already exists".format(
                        song_path, new_song_path))
                try:
                    shutil.move(song_path, new_song_path)
                except FileNotFoundError:
                    raise
