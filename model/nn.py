import math
import multiprocessing
import threading

import keras

from keras.callbacks import ModelCheckpoint


def gen_model():
    model = keras.models.Sequential()
    # input/hidden layer 1
    model.add(keras.layers.Dense(9 * 3 * 4, input_shape=(1,), name="actor_in",
                                 kernel_regularizer=keras.layers.regularizers.l2(1e-8)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    # hidden layer 2
    model.add(keras.layers.Dense(9 * 3 * 3, kernel_initializer="glorot_normal",
                                 kernel_regularizer=keras.layers.regularizers.l2(1e-8)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    # # hidden layer 3
    # model.add(keras.layers.Dense(self.state_dim * 3 * 4, kernel_initializer="glorot_normal"))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Activation("relu"))
    # output layer
    output_initializer = keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
    model.add(keras.layers.Dense(1, kernel_initializer=output_initializer, activation='sigmoid', name="output"))

    model = keras.utils.multi_gpu_model(model, gpus=2, cpu_merge=False, cpu_relocation=False)
    return model


def compile_model(model):
    model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy'])


def train_model(model, sequencer, epochs=1):
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', verbose=1,
                                   save_best_only=True, save_weights_only=True)

    model.fit_generator(generator=sequencer.train_sequence(), validation_data=sequencer.validate_sequence(),
                        steps_per_epoch=math.ceil(len(sequencer.train_set) / sequencer.batch_size),
                        validation_steps=math.ceil(len(sequencer.validate_set) / sequencer.batch_size),
                        epochs=epochs,
                        callbacks=[checkpointer], verbose=2,
                        max_queue_size=multiprocessing.cpu_count() ** 2)

    # Stop generators
    for processes in [sequencer.processes_train, sequencer.processes_validate]:
        #: :type: multiprocessing.Process
        for process in processes:  # type: multiprocessing.Process
            process.terminate()

    sequencer.done_queue_train.put(0)
    sequencer.done_queue_validate.put(0)
