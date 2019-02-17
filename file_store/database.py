import abc
import atexit
import base64
import enum
import os
import pickle
import six
import traceback

from util.class_property import ClassProperty

from unqlite import UnQLite


_SONG_INFO_DATABASE_FN_ENV = "SONG_INFO_DATABASE_PATH"
_SONG_SAMPLES_DATABASE_FN_ENV = "SONG_SAMPLES_DATABASE_PATH"

DB_GOOD_SONGS = "good_song_indexes"
DB_BAD_SONGS = "bad_song_indexes"
DB_RECORD_FIELD = "__id"

__all__ = ["SongInfoDatabase", "DB_GOOD_SONGS", "DB_BAD_SONGS", "DB_RECORD_FIELD", "update_collection_with_field",
           "remove_field_from_collection", "SongSamplesDatabase"]


class ODBCGetSetEnumMixin(object):
    def get_value(self, record, raw=False):
        if self.value[0] in record:
            value = record[self.value[0]]
            if raw:
                return value

            if self.value[1] is str:
                return value.decode('utf-8')
            elif self.value[1] is object:
                return ODBCGetSetEnumMixin.deserialize_object(value)
            else:
                return value
        else:
            return self.value[-1]

    def set_value(self, record, value):
        if self.value[1] is str:
            record[self.value[0]] = value.encode()
        elif self.value[1] is object:
            record[self.value[0]] = base64.urlsafe_b64encode(value)
        else:
            record[self.value[0]] = value

    def has_value(self, record):
        if self.value[0] in record:
            return True
        return False

    def remove_value(self, record):
        del record[self.value[0]]

    @staticmethod
    def serialize_object(value):
        return pickle.dumps(value, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def deserialize_object(value):
        return pickle.loads(base64.urlsafe_b64decode(value))


class ODBCInitEnumMixin(object):
    @classmethod
    def initialize_record(cls, *values):
        ret = dict()
        for idx, field in enumerate(cls):
            if idx < len(values):
                value = values[idx]
            else:
                continue
                # value = field.value[-1]
            field.set_value(ret, value)

        return ret


@six.add_metaclass(abc.ABCMeta)
class BaseDatabase(object):
    @ClassProperty
    @classmethod
    def db(cls):
        if cls._database is not None:
            return cls._database
        else:
            cls._database = UnQLite(cls.database_file)
            atexit.register(cls._database.close)
            return cls._database

    @ClassProperty
    @classmethod
    def database_file(cls):
        try:
            db_fn = os.environ[cls.database_fn_env()]
            try:
                if not os.path.exists(db_fn):
                    with open(db_fn, 'ab+'):
                        pass
            except IOError:
                raise RuntimeError("Unable to create database file! Error:\n{}".format(traceback.print_exc()))
            return db_fn
        except KeyError:
            raise EnvironmentError("[{}] is not specified. Specify the intended path to the database file (will be "
                                   "created if it does not exist)".format(cls.database_fn_env()))

    @abc.abstractmethod
    def database_fn_env(cls):
        pass


class SongInfoDatabase(BaseDatabase):
    _database = None

    @classmethod
    def database_fn_env(cls):
        return _SONG_INFO_DATABASE_FN_ENV

    class SongInfoODBC(ODBCInitEnumMixin, ODBCGetSetEnumMixin, enum.Enum):
        SONG_HASH = ["hash", str, None]
        SONG_PATH = ["path", str, None]
        SONG_SAMPLES_ID = ["samples_id", int, None]
        SONG_SAMPLE_RATE = ["sample_rate", int, None]


class SongSamplesDatabase(BaseDatabase):
    _database = None

    @classmethod
    def database_fn_env(cls):
        return _SONG_SAMPLES_DATABASE_FN_ENV

    class SongSamplesODBC(ODBCInitEnumMixin, ODBCGetSetEnumMixin, enum.Enum):
        SONG_HASH = ["hash", str, None]
        SONG_INFO_ID = ["info_id", int, None]
        SONG_SAMPLES_LEFT = ["samples_left", object, None]
        SONG_SAMPLES_RIGHT = ["samples_right", object, None]


def update_collection_with_field(collection, field, default, db, updates_between_commits=100):
    if not db:
        db = SongInfoDatabase.db

    db.begin()
    for idx in range(len(collection)):
        if idx % updates_between_commits == updates_between_commits - 1:
            db.commit()
            db.begin()

        record = collection.fetch(idx)
        if not field.has_value(record):
            field.set_value(record, default)
            collection.update(record[DB_RECORD_FIELD], record)
    db.commit()


def remove_field_from_collection(collection, field, db, updates_between_commits=100):
    if not db:
        db = SongInfoDatabase.db

    db.begin()
    for idx in range(len(collection)):
        if idx % updates_between_commits == updates_between_commits - 1:
            db.commit()
            db.begin()

        record = collection.fetch(idx)
        if field.has_value(record):
            field.remove_value(record)
            collection.update(record[DB_RECORD_FIELD], record)
    db.commit()
