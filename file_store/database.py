import atexit
import enum
import os
import traceback

from util.class_property import ClassProperty

from unqlite import UnQLite


_DATABASE_FN_ENV = "DATABASE_PATH"

DB_GOOD_SONG_PATHS = "good_song_indexes"
DB_BAD_SONG_PATHS = "bad_song_indexes"
DB_GOOD_SONG_REPRESENTATIONS = "good_song_representations"
DB_BAD_SONG_REPRESENTATIONS = "bad_song_representations"
DB_RECORD_FIELD = "__id"

__all__ = ["Database", "DB_GOOD_SONG_PATHS", "DB_BAD_SONG_PATHS", "DB_GOOD_SONG_REPRESENTATIONS",
           "DB_BAD_SONG_REPRESENTATIONS", "DB_RECORD_FIELD", "update_collection_with_field",
           "remove_field_from_collection"]


class Database(object):
    __database = None

    @ClassProperty
    @classmethod
    def db(cls):
        if Database.__database is not None:
            return Database.__database
        else:
            Database.__database = UnQLite(Database.database_file)
            atexit.register(Database.__database.close)
            return Database.__database

    @ClassProperty
    @classmethod
    def database_file(cls):
        try:
            db_fn = os.environ[_DATABASE_FN_ENV]
            try:
                if not os.path.exists(db_fn):
                    with open(db_fn, 'ab+'):
                        pass
            except IOError:
                raise RuntimeError("Unable to create database file! Error:\n{}".format(traceback.print_exc()))
            return db_fn
        except KeyError:
            raise EnvironmentError("[{}] is not specified. Specify the intended path to the database file (will be "
                                   "created if it does not exist)".format(_DATABASE_FN_ENV))

    class SongPathODBC(enum.Enum):
        SONG_HASH = "hash"
        SONG_PATH = "path"
        REPRESENTATION_BUILT = "representation_built"
        FFTS_BUILT = "ffts_built"

    class SongRepresentationODBC(enum.Enum):
        SONG_HASH = "hash"
        SONG_SPCS_LEFT = "spectrograms_left"
        SONG_SPCS_RIGHT = "spectrograms_right"
        SONG_SAMPLES_LEFT = "samples_left"
        SONG_SAMPLES_RIGHT = "samples_right"


def update_collection_with_field(collection, field, default, updates_between_commits=100, db=None):
    if not db:
        db = Database.db

    db.begin()
    for idx in range(len(collection)):
        if idx % updates_between_commits == updates_between_commits - 1:
            db.commit()
            db.begin()
        record = collection.fetch(idx)
        if field not in record:
            record[field] = default
            collection.update(record[DB_RECORD_FIELD], record)
    db.commit()


def remove_field_from_collection(collection, field, updates_between_commits=100, db=None):
    if not db:
        db = Database.db

    db.begin()
    for idx in range(len(collection)):
        if idx % updates_between_commits == updates_between_commits - 1:
            db.commit()
            db.begin()
        record = collection.fetch(idx)
        if field in record:
            del record[field]
            collection.update(record[DB_RECORD_FIELD], record)
    db.commit()
