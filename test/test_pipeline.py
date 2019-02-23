from pipeline.sequencer import *


def test_create_pipeline():
    pipeline = Sequencer()
    pipeline.initialize_dataset()
