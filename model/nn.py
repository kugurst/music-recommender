import enum
import math
import multiprocessing
import os

import keras
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
              rms_fractional=in_use_features.USE_RMS_FRACTIONAL):
    subsystems = []
    for subsystem, in_use in [(_tempo_model, tempo), (_flux_model, flux), (_rolloff_model, rolloff), (_mel_model, mel),
                              (_contrast_model, contrast), (_tonnetz_model, tonnetz), (_chroma_model, chroma),
                              (_hpss_model, hpss), (_rms_fractional_model, rms_fractional)]:
        if in_use:
            subsystems.append(subsystem())

    model = keras.models.Sequential()
    model.add(keras.layers.Merge(subsystems, mode="concat"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(1024,
                                 kernel_initializer="glorot_normal", activation='relu'))
    output_initializer = keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
    model.add(keras.layers.Dense(1, kernel_initializer=output_initializer, activation='sigmoid', name="classification"))

    num_gpus = int(os.environ.get(_NUM_GPUS_ENV, 1))
    if num_gpus > 1:
        model = keras.utils.multi_gpu_model(model, gpus=num_gpus, cpu_merge=True, cpu_relocation=False)
    return model


def _tempo_model():
    model = keras.models.Sequential()

    model.add(keras.layers.Dense(TEMPO_SHAPE[0] * 12, input_shape=TEMPO_SHAPE, kernel_initializer="glorot_normal",
                                 activation='relu', name=InputNames.TEMPO.get_layer_name()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(TEMPO_SHAPE[0] * 24, kernel_initializer="glorot_normal",
                                 kernel_regularizer=keras.layers.regularizers.l2(1e-8)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))

    return model


def _flux_model():
    model = keras.models.Sequential()

    model.add(keras.layers.Dense(FLUX_SHAPE[0] * 12, input_shape=FLUX_SHAPE, kernel_initializer="glorot_normal",
                                 activation='relu', name=InputNames.FLUX.get_layer_name()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(FLUX_SHAPE[0] * 24, kernel_initializer="glorot_normal",
                                 kernel_regularizer=keras.layers.regularizers.l2(1e-8)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))

    return model


def _rolloff_model():
    model = keras.models.Sequential()

    model.add(keras.layers.Dense(ROLLOFF_SHAPE[0] * 12, input_shape=ROLLOFF_SHAPE, kernel_initializer="glorot_normal",
                                 activation='relu', name=InputNames.ROLLOFF.get_layer_name()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(ROLLOFF_SHAPE[0] * 24, kernel_initializer="glorot_normal",
                                 kernel_regularizer=keras.layers.regularizers.l2(1e-8)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))

    return model


def _mel_model():
    keep_probability = 0.75

    model = keras.models.Sequential()

    # Convolutional input
    model.add(keras.layers.Conv1D(32, kernel_size=4, strides=2, activation='relu', input_shape=MEL_SHAPE,
                                  name=InputNames.MEL.get_layer_name()))
    # Convolutional hidden 1
    model.add(keras.layers.Conv1D(64, kernel_size=2, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, padding="same"))
    model.add(keras.layers.Conv1D(128, kernel_size=2, activation='relu'))
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


def _contrast_model():
    keep_probability = 0.75

    model = keras.models.Sequential()

    # Convolutional input
    model.add(keras.layers.Conv1D(32, kernel_size=4, strides=2, activation='relu', input_shape=CONTRAST_SHAPE,
                                  name=InputNames.CONTRAST.get_layer_name()))
    # Convolutional hidden 1
    model.add(keras.layers.Conv1D(64, kernel_size=2, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, padding="same"))
    model.add(keras.layers.Conv1D(128, kernel_size=2, activation='relu'))
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


def _tonnetz_model():
    keep_probability = 0.75

    model = keras.models.Sequential()

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


def _chroma_model():
    keep_probability = 0.75

    model = keras.models.Sequential()

    # Convolutional input
    model.add(keras.layers.Conv1D(32, kernel_size=4, strides=2, activation='relu', input_shape=CHROMA_SHAPE,
                                  name=InputNames.CHROMA.get_layer_name()))
    # Convolutional hidden 1
    model.add(keras.layers.Conv1D(64, kernel_size=2, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, padding="same"))
    model.add(keras.layers.Conv1D(128, kernel_size=2, activation='relu'))
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


def _hpss_model():
    keep_probability = 0.75

    model = keras.models.Sequential()

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


def _rms_fractional_model():
    model = keras.models.Sequential()

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


def true_positive_rate(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    # return K.mean(y_pred)

    return true_positives / possible_positives


def compile_model(model):
    model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy', true_positive_rate])


def train_model(model, sequencer, epochs=1):
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', verbose=1,
                                   save_best_only=True, save_weights_only=True)

    model.fit_generator(generator=sequencer.train_sequence(), validation_data=sequencer.validate_sequence(),
                        steps_per_epoch=math.ceil(len(sequencer.train_set) / sequencer.batch_size),
                        validation_steps=math.ceil(len(sequencer.validate_set) / sequencer.batch_size),
                        class_weight={0: 0.25, 1: 1}, epochs=epochs,
                        callbacks=[checkpointer], verbose=2,
                        max_queue_size=multiprocessing.cpu_count() ** 2)

    # Stop generators
    for processes in [sequencer.processes_train, sequencer.processes_validate]:
        #: :type: multiprocessing.Process
        for process in processes:  # type: multiprocessing.Process
            process.terminate()

    sequencer.done_queue_train.put(0)
    sequencer.done_queue_validate.put(0)
