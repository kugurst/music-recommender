import logging

from model import nn
from pipeline.sequencer import *


logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s]%(processName)s:%(name)s:%(levelname)s - %(message)s')


def main():
    model = nn.gen_model()
    sequencer = Sequencer()

    sequencer.initialize_dataset()
    nn.compile_model(model)

    nn.train_model(model, sequencer)


if __name__ == "__main__":
    main()
