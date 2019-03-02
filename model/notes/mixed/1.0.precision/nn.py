import enum
import math
import multiprocessing
import os

import keras
import numpy as np
import sklearn
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint

from params import in_use_features
from pipeline.features import *

_NUM_GPUS_ENV = "NUM_GPUS"


@enum.unique
class InputNames(enum.Enum):
    TEMPO = "tempo"
    FLUX = "flux"
    ROLLOFF = "rolloff"
    MEL = "mel"
    CHROMA = "chroma"
    CONTRAST = "contrast"
    HPSS = "hpss"
    TONNETZ = "tonnetz"
    RMS_FRACTIONAL = "rms_fractional"

    def get_nn_input_name(self):
        return self.value + "_input"

    def get_layer_name(self):
        return self.value


def gen_model(tempo=in_use_features.USE_TEMPO, flux=in_use_features.USE_FLUX, rolloff=in_use_features.USE_ROLLOFF,
              mel=in_use_features.USE_MEL, contrast=in_use_features.USE_CONTRAST, tonnetz=in_use_features.USE_TONNETZ,
              chroma=in_use_features.USE_CHROMA, hpss=in_use_features.USE_HPSS,
              rms_fractional=in_use_features.USE_RMS_FRACTIONAL, use_flat=False):
    with tf.device('/cpu:0'):
        subsystems = []
        for subsystem, in_use in [(_tempo_model, tempo), (_flux_model, flux), (_rolloff_model, rolloff), (_mel_model, mel),
                                  (_contrast_model, contrast), (_tonnetz_model, tonnetz), (_chroma_model, chroma),
                                  (_hpss_model, hpss), (_rms_fractional_model, rms_fractional)]:
            if in_use:
                subsystems.append(subsystem(use_flat))

        model = keras.models.Sequential()
        model.add(keras.layers.Merge(subsystems, mode="concat"))
        model.add(keras.layers.Activation("relu"))

        model.add(keras.layers.Dense(8192, kernel_initializer="glorot_normal", bias_initializer="glorot_normal"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))

        if use_flat:
            model.add(keras.layers.Dense(4096, kernel_initializer="glorot_normal", bias_initializer="glorot_normal"))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation("relu"))

        # model.add(keras.layers.Dense(2048, kernel_initializer="glorot_normal", bias_initializer="glorot_normal"))
        # model.add(keras.layers.BatchNormalization())
        # model.add(keras.layers.Activation("relu"))

        if use_flat:
            model.add(keras.layers.Dense(1024, kernel_initializer="glorot_normal", bias_initializer="glorot_normal"))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation("relu"))

        model.add(keras.layers.Dense(512, kernel_initializer="glorot_normal", bias_initializer="glorot_normal"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))

        output_initializer = keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
        model.add(
            keras.layers.Dense(1, kernel_initializer=output_initializer, activation='sigmoid', name="classification"))

        num_gpus = int(os.environ.get(_NUM_GPUS_ENV, 1))
        if num_gpus > 1:
            model = keras.utils.multi_gpu_model(model, gpus=num_gpus, cpu_merge=False, cpu_relocation=False)
    return model


def _tempo_model(use_flat=False):
    model = keras.models.Sequential()

    if use_flat:
        model.add(keras.layers.ActivityRegularization(input_shape=TEMPO_SHAPE, name=InputNames.TEMPO.get_layer_name()))
        return model

    model.add(keras.layers.Dense(TEMPO_SHAPE[0] * 12, input_shape=TEMPO_SHAPE, kernel_initializer="glorot_normal",
                                 activation='relu', name=InputNames.TEMPO.get_layer_name()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(TEMPO_SHAPE[0] * 24, kernel_initializer="glorot_normal",
                                 kernel_regularizer=keras.layers.regularizers.l2(1e-8)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))

    return model


def _flux_model(use_flat=False):
    model = keras.models.Sequential()

    if use_flat:
        model.add(keras.layers.ActivityRegularization(input_shape=FLUX_SHAPE, name=InputNames.FLUX.get_layer_name()))
        return model

    model.add(keras.layers.Dense(FLUX_SHAPE[0] * 12, input_shape=FLUX_SHAPE, kernel_initializer="glorot_normal",
                                 activation='relu', name=InputNames.FLUX.get_layer_name()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(FLUX_SHAPE[0] * 24, kernel_initializer="glorot_normal",
                                 kernel_regularizer=keras.layers.regularizers.l2(1e-8)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))

    return model


def _rolloff_model(use_flat=False):
    model = keras.models.Sequential()

    if use_flat:
        model.add(keras.layers.ActivityRegularization(input_shape=ROLLOFF_SHAPE,
                                                      name=InputNames.ROLLOFF.get_layer_name()))
        return model

    model.add(keras.layers.Dense(ROLLOFF_SHAPE[0] * 12, input_shape=ROLLOFF_SHAPE, kernel_initializer="glorot_normal",
                                 activation='relu', name=InputNames.ROLLOFF.get_layer_name()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(ROLLOFF_SHAPE[0] * 24, kernel_initializer="glorot_normal",
                                 kernel_regularizer=keras.layers.regularizers.l2(1e-8)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))

    return model


def _mel_model(use_flat=False):
    keep_probability = 0.25

    model = keras.models.Sequential()

    if use_flat:
        model.add(keras.layers.Flatten(input_shape=MEL_SHAPE, name=InputNames.MEL.get_layer_name()))
        return model

    # Convolutional input
    model.add(keras.layers.Conv1D(32, kernel_size=8, activation='relu', input_shape=MEL_SHAPE,
                                  name=InputNames.MEL.get_layer_name()))
    # Convolutional hidden 1
    model.add(keras.layers.Conv1D(64, kernel_size=4, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, padding="same"))
    model.add(keras.layers.Conv1D(128, kernel_size=2, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, padding="same"))
    model.add(keras.layers.Conv1D(256, kernel_size=1, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, padding="same"))

    model.add(keras.layers.Flatten())

    # Fully connected 1
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    # Dropout layer
    model.add(keras.layers.Dropout(keep_probability))
    # # Fully connected 2
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))

    return model


def _contrast_model(use_flat=False):
    keep_probability = 0.25

    model = keras.models.Sequential()

    if use_flat:
        model.add(keras.layers.Flatten(input_shape=CONTRAST_SHAPE, name=InputNames.CONTRAST.get_layer_name()))
        return model

    # Convolutional input
    model.add(keras.layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=CONTRAST_SHAPE,
                                  name=InputNames.CONTRAST.get_layer_name()))
    # Convolutional hidden 1
    model.add(keras.layers.Conv1D(64, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, padding="same"))

    model.add(keras.layers.Flatten())

    # Fully connected 1
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    # Dropout layer
    model.add(keras.layers.Dropout(keep_probability))
    # # Fully connected 2
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))

    return model


def _tonnetz_model(use_flat=False):
    keep_probability = 0.25

    model = keras.models.Sequential()

    if use_flat:
        model.add(keras.layers.Flatten(input_shape=TONNETZ_SHAPE, name=InputNames.TONNETZ.get_layer_name()))
        return model

    # Convolutional input
    model.add(keras.layers.Conv1D(32, kernel_size=4, strides=2, activation='relu', input_shape=TONNETZ_SHAPE,
                                  name=InputNames.TONNETZ.get_layer_name()))
    # Convolutional hidden 1
    model.add(keras.layers.Conv1D(64, kernel_size=2, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, padding="same"))
    model.add(keras.layers.Conv1D(128, kernel_size=1, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, padding="same"))

    model.add(keras.layers.Flatten())

    # Fully connected 1
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    # Dropout layer
    model.add(keras.layers.Dropout(keep_probability))
    # # Fully connected 2
    # model.add(Dense(512))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))

    return model


def _chroma_model(use_flat=False):
    keep_probability = 0.25

    model = keras.models.Sequential()

    if use_flat:
        model.add(keras.layers.Flatten(input_shape=CHROMA_SHAPE, name=InputNames.CHROMA.get_layer_name()))
        return model

    # Convolutional input
    model.add(keras.layers.Conv1D(32, kernel_size=6, activation='relu', input_shape=CHROMA_SHAPE,
                                  name=InputNames.CHROMA.get_layer_name()))
    # Convolutional hidden 1
    model.add(keras.layers.Conv1D(64, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, padding="same"))
    model.add(keras.layers.Conv1D(128, kernel_size=1, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, padding="same"))

    model.add(keras.layers.Flatten())

    # Fully connected 1
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    # Dropout layer
    model.add(keras.layers.Dropout(keep_probability))
    # # Fully connected 2
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))

    return model


def _hpss_model(use_flat=False):
    keep_probability = 0.25

    model = keras.models.Sequential()

    if use_flat:
        model.add(keras.layers.Flatten(input_shape=HPSS_SHAPE, name=InputNames.HPSS.get_layer_name()))
        return model

    # Convolutional input
    model.add(keras.layers.Conv2D(32, kernel_size=4, strides=2, activation='relu', input_shape=HPSS_SHAPE,
                                  name=InputNames.HPSS.get_layer_name()))
    # Convolutional hidden 1
    model.add(keras.layers.Conv2D(64, kernel_size=2, activation='relu'))
    model.add(keras.layers.MaxPooling2D(2, padding="same"))
    model.add(keras.layers.Conv2D(128, kernel_size=1, activation='relu'))
    model.add(keras.layers.MaxPooling2D(2, padding="same"))

    model.add(keras.layers.Flatten())

    # Fully connected 1
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    # Dropout layer
    model.add(keras.layers.Dropout(keep_probability))
    # # Fully connected 2
    # model.add(Dense(512))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))

    return model


def _rms_fractional_model(use_flat=False):
    model = keras.models.Sequential()

    if use_flat:
        model.add(keras.layers.ActivityRegularization(input_shape=RMS_SHAPE,
                                                      name=InputNames.RMS_FRACTIONAL.get_layer_name()))
        return model

    model.add(keras.layers.Dense(RMS_SHAPE[0] * 12, input_shape=RMS_SHAPE, kernel_initializer="glorot_normal",
                                 activation='relu', name=InputNames.RMS_FRACTIONAL.get_layer_name()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(RMS_SHAPE[0] * 24, kernel_initializer="glorot_normal",
                                 kernel_regularizer=keras.layers.regularizers.l2(1e-8)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))

    return model


def false_positive_rate(**kwargs):
    def metric(labels, predictions):
        # any tensorflow metric
        value, update_op = tf.metrics.false_positives(labels, predictions, **kwargs)

        # find all variables created for this metric
        metric_vars = [i for i in tf.local_variables() if 'false_positives' in i.name.split('/')[2]]

        # Add metric variables to GLOBAL_VARIABLES collection.
        # They will be initialized for new session.
        for v in metric_vars:
            tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

        # force to update metric values
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
            return value

    return metric


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


class BestPrecisionSaver(keras.callbacks.Callback):
    def __init__(self, filepath, validate_set, validate_target, batch_size, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.filepath = filepath
        self.validate_set = validate_set
        self.validate_target = validate_target
        self.batch_size = batch_size
        self.best_precision = 0

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validate_set, batch_size=self.batch_size, verbose=2)
        y_pred = y_pred.flatten()
        y_pred_label = np.round(y_pred)
        # tn, fp, fn, tp = sklearn.metrics.confusion_matrix(self.validate_target, y_pred_label)

        val_precision, val_recall, val_f1, _ = sklearn.metrics.precision_recall_fscore_support(
            self.validate_target, y_pred_label, beta=0.5, labels=[0, 1], average="binary")
        print ("— val_f1: % f — val_precision: % f — val_recall % f" % (val_f1, val_precision, val_recall))

        if val_precision > self.best_precision:
            print("Validation precision improved from [{}] to [{}]. Saving model to [{}]".format(
                self.best_precision, val_precision, self.filepath
            ))
            self.best_precision = val_precision
            self.model.save_weights(self.filepath, True)


def compile_model(model):
    model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy', precision, recall])


def train_model(model, sequencer, epochs=120, batch_size=64):
    # metrics = Metrics()
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', verbose=1,
                                   save_best_only=True, save_weights_only=True)

    train_steps = int(math.ceil(len(sequencer.train_sequence) / batch_size))
    validate_steps = int(math.ceil(len(sequencer.validate_sequence) / batch_size))

    try:
        model.fit_generator(generator=sequencer.train_sequence, validation_data=sequencer.validate_sequence,
                            epochs=epochs, callbacks=[checkpointer], verbose=2, class_weight={0: 0.25, 1: 1},
                            max_queue_size=multiprocessing.cpu_count() ** 3, workers=multiprocessing.cpu_count(),
                            steps_per_epoch=train_steps, validation_steps=validate_steps, shuffle=False,
                            use_multiprocessing=True)
        # model.fit_generator(generator=sequencer.train_sequence(), validation_data=sequencer.validate_sequence(),
        #                     steps_per_epoch=math.ceil(len(sequencer.train_set) / sequencer.batch_size),
        #                     validation_steps=math.ceil(len(sequencer.validate_set) / sequencer.batch_size),
        #                     class_weight={0: 0.25, 1: 1}, epochs=epochs,
        #                     callbacks=[checkpointer], verbose=2,
        #                     max_queue_size=multiprocessing.cpu_count() ** 2)
    except:
        model.save_weights('saved_models/weights.emergency.from_scratch.hdf5')
        raise
    # finally:
    #     # Stop generators
    #     for processes in [sequencer.processes_train, sequencer.processes_validate]:
    #         #: :type: multiprocessing.Process
    #         for process in processes:  # type: multiprocessing.Process
    #             process.terminate()
    #
    #     sequencer.done_queue_train.put(0)
    #     sequencer.done_queue_validate.put(0)


def train_model_flat(model, train_set, train_target, validate_set, validate_target, epochs=500, batch_size_base=128):
    num_gpus = int(os.environ.get(_NUM_GPUS_ENV, 1))
    batch_size = num_gpus * batch_size_base

    best_precision_checkpointer = BestPrecisionSaver('saved_models/weights.best_precision.from_scratch.hdf5',
                                                    validate_set, validate_target, batch_size)
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best_loss.from_scratch.hdf5', verbose=1,
                                   save_best_only=True, save_weights_only=True)

    try:
        model.load_weights('saved_models/weights.best_precision.from_scratch.hdf5')
        model.fit(
            x=train_set, y=train_target, validation_data=(validate_set, validate_target),
            shuffle=True, class_weight={0: 0.125, 1: 1},
            batch_size=batch_size, epochs=epochs, verbose=2, callbacks=[checkpointer, best_precision_checkpointer],
            initial_epoch=92
        )
    except:
        model.save_weights('saved_models/weights.emergency.from_scratch.hdf5')
        raise
