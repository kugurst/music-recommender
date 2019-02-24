import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]%(processName)s:%(name)s:%(levelname)s - %(message)s')

from model import nn
from pipeline.sequencer import *


def main():
    model = nn.gen_model()
    sequencer = Sequencer()

    sequencer.initialize_dataset()
    nn.compile_model(model)

    nn.train_model(model, sequencer)


if __name__ == "__main__":
    main()
