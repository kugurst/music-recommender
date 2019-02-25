import logging

from numpy import ComplexWarning

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]%(processName)s:%(name)s:%(levelname)s - %(message)s')
import warnings

from model import nn
from pipeline.sequencer import *


def main():
    warnings.filterwarnings("ignore", category=ComplexWarning)

    model = nn.gen_model()
    print(model.summary())
    sequencer = Sequencer()

    sequencer.initialize_dataset()
    nn.compile_model(model)

    nn.train_model(model, sequencer)


if __name__ == "__main__":
    main()
