import abc
import atexit
import base64
import enum
import os
import pickle
import six
import traceback

import leveldb

from util.class_property import ClassProperty

from unqlite import UnQLite


_SONG_INFO_DATABASE_FN_ENV = "SONG_INFO_DATABASE_PATH"
_SONG_SAMPLES_DATABASE_FN_ENV = "SONG_SAMPLES_DATABASE_PATH"

DB_GOOD_SONGS = "good_song_indexes"
DB_BAD_SONGS = "bad_song_indexes"
DB_RECORD_FIELD = "__id"

__all__ = ["SongInfoDatabase", "DB_GOOD_SONGS", "DB_BAD_SONGS", "DB_RECORD_FIELD", "update_collection_with_field",
           "remove_field_from_collection", "SongSamplesDatabase", "SongSamplesLVLDatabase", "SongSampleRecord"]


class SongSampleRecord(object):
    def __init__(self):
        #: :type: str
        self.hash = None
        #: :type: int
        self.info_id = None
        #: :type: bool
        self.is_good_song = None
        #: :type: list[numpy.ndarray]
        self.samples_left = None
        #: :type: list[numpy.ndarray]
        self.samples_right = None


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
        elif self.value[1] is bool:
            record[self.value[0]] = bytes(value)
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
        SONG_SAMPLES_UNQLITE_ID = ["unqlite_samples_id", int, None]
        SONG_SAMPLE_RATE = ["sample_rate", int, None]


class SongSamplesDatabase(BaseDatabase):
    _database = None

    @classmethod
    def database_fn_env(cls):
        return _SONG_SAMPLES_DATABASE_FN_ENV

    class SongSamplesODBC(ODBCInitEnumMixin, ODBCGetSetEnumMixin, enum.Enum):
        SONG_HASH = ["hash", str, None]
        SONG_INFO_ID = ["info_id", int, None]
        SONG_IS_GOOD = ["is_good_song", bool, None]
        SONG_SAMPLES_LEFT = ["samples_left", object, None]
        SONG_SAMPLES_RIGHT = ["samples_right", object, None]


class SongSamplesLVLDatabase(object):
    _database = None
    database_fn_env = "SONG_SAMPLES_LVL_DATABASE_PATH"

    @ClassProperty
    @classmethod
    def db(cls):
        if cls._database is not None:
            return cls._database
        else:
            cls._database = leveldb.LevelDB(cls.database_file)
            # atexit.register(cls._database)
            return cls._database

    @ClassProperty
    @classmethod
    def database_file(cls):
        try:
            db_fn = os.environ[cls.database_fn_env]
            try:
                if not os.path.exists(db_fn):
                    os.makedirs(db_fn)
            except IOError:
                raise RuntimeError("Unable to create database file! Error:\n{}".format(traceback.print_exc()))
            return db_fn
        except KeyError:
            raise EnvironmentError("[{}] is not specified. Specify the intended path to the database file (will be "
                                   "created if it does not exist)".format(cls.database_fn_env))

    class SongSamplesIndex(object):
        __index_key = "index".encode()
        ___index = None
        __batch = None

        @ClassProperty
        @classmethod
        def __index(cls):
            if not cls.___index:
                try:
                    cls.___index = SongSamplesLVLDatabase.db.Get(cls.__index_key)
                    cls.___index = cls.deserialize_object(cls.___index)
                except KeyError:
                    cls.___index = dict()
                    cls.begin()
                    SongSamplesLVLDatabase.db.Put(cls.__index_key, cls.serialize_object(cls.___index))
                    cls.commit()
            return cls.___index

        @classmethod
        def all(cls):
            return cls.__index

        @classmethod
        def store(cls, value):
            idx = len(cls.__index)
            hash = value["hash"]
            try:
                hash = hash.encode()
            except AttributeError:
                pass

            value_map = {hash + field.encode(): value[field]
                         for field in value}
            value_map = {field: cls.bytify(value_map[field]) for field in value_map}
            cls.__index[idx] = hash
            cls.__index[hash] = idx

            for field in value_map:
                SongSamplesLVLDatabase.db.Put(field, value_map[field])

            pass

        @classmethod
        def update(cls, idx, value):
            pass

        @classmethod
        def remove(cls, idx):
            pass

        @classmethod
        def fetch(cls, idx) -> SongSampleRecord:
            if isinstance(idx, int):
                hash = cls.__index[idx * 2]
            else:
                hash = idx.encode()

            value_map = dict()
            for field in SongSamplesLVLDatabase.SongSamplesODBC:
                db_field = hash + field.value[0].encode()
                try:
                    value = SongSamplesLVLDatabase.db.Get(db_field)
                except KeyError:
                    continue
                value_map[field.value[0]] = value

            value_map = SongSamplesLVLDatabase.SongSamplesODBC.load_from_bytes(value_map)
            value_map[DB_RECORD_FIELD] = cls.__index[hash] / 2
            return value_map

        @classmethod
        def begin(cls):
            cls.__batch = leveldb.WriteBatch()

        @classmethod
        def commit(cls):
            SongSamplesLVLDatabase.db.Put(cls.__index_key, cls.serialize_object(cls.__index))
            SongSamplesLVLDatabase.db.Write(cls.__batch)
            cls.__batch = None

        @classmethod
        def deserialize_object(cls, value):
            return pickle.loads(base64.urlsafe_b64decode(value))

        @classmethod
        def serialize_object(cls, value, should_pickle=True):
            if should_pickle:
                value = pickle.dumps(value, pickle.HIGHEST_PROTOCOL)
            return base64.urlsafe_b64encode(value)

        @classmethod
        def len(cls):
            return int(len(cls.__index) / 2)

        @classmethod
        def bytify(cls, value):
            if isinstance(value, six.string_types):
                return value.encode()
            elif isinstance(value, int):
                return str(value).encode()
            elif isinstance(value, bool):
                return bytes(value)
            elif isinstance(value, bytes):
                return value
            else:
                return base64.urlsafe_b64encode(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            pass

    class SongSamplesODBC(ODBCGetSetEnumMixin, enum.Enum):
        SONG_HASH = ["hash", str, None]
        SONG_INFO_ID = ["info_id", int, None]
        SONG_IS_GOOD = ["is_good_song", bool, None]
        SONG_SAMPLES_LEFT = ["samples_left", object, None]
        SONG_SAMPLES_RIGHT = ["samples_right", object, None]

        @classmethod
        def load_from_bytes(cls, record):
            new_record = dict()
            for enum_ in cls:
                field = enum_.value[0]
                if field not in record:
                    continue
                new_record[field] = enum_.convert_from_bytes(record[field])
            return new_record

        def convert_from_bytes(self, value):
            if self.value[1] is str:
                return value.decode('utf-8')
            elif self.value[1] is int:
                return int(value.decode('utf-8'))
            elif self.value[1] is bool:
                return bool(value)
            elif self.value[1] is object:
                return ODBCGetSetEnumMixin.deserialize_object(value)
            else:
                return value

        def get_value(self, record, raw=True):
            return super(self.__class__, self).get_value(record, raw)


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
