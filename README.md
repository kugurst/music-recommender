# Machine Learning Engineer Nanodegree
## Specializations
## Project: Capstone Proposal and Capstone Project

### Requirements

The following packages are required to run the code in this repository:

- numpy
- scipy
- sklearn
- keras==2.1.6
- tensorflow-gpu
- pydub
- pytest
- pyarrow
- regex
- zodb
- unqlite
- unidecode
- librosa
- psutil

### Support Files

https://drive.google.com/open?id=1U9H2o7AQ_1jsd-qV3mi4QHUs_2Ta44Gk

The link above contains intermediate files for model training and feature investigation. The following are descriptions of the files:

- `song_info.db`: The list of all songs being trained; including whether or not they are considered good
- `nn_input`: A folder containing the model-ready input to use in keras training.
- `song_zodb_samples_small_features.tar.bz2`: An archive containing the `pyarrow` serialized feature samples for all 30 samples for all songs.

### Source Code Tour

As a general note, the environment variables mentioned in each section are cumulative to the next section

#### I Want to Train a Model

- Define the environment variable `NN_INPUT_PYARROW_DIR` to point towards the folder where you downloaded `nn_input`
- Run `main.py` to train the neural network, or `main_decision_tree.py` to grid search for a good decision tree classifier.

#### I Want to Create a Database with My Own Music

- Define the environment variable `FFMPEG` to point towards the `ffmpeg` executable
- Define the environment variable `SONG_SAMPLES_ZODB_DATABASE_PATH` to point towards where you want the database of song samples to be created.
- Define the environment variable `SONG_INFO_DATABASE_PATH` to point towards where you want the database of song information to be created.
- Define the environment variable `GOOD_SONG_DIR` to point towards the the folder where songs considered good are contained.
- Defint the environment variable `BAD_SONG_DIR` to point towards the folder where songs considered bad are contained.

- Run `test_build_song_indexes()` in `test/test_audio_representation.py`
- Run `test_build_song_zodb_samples()` in the same file above

#### I Want to Extract the Features from My Own Datbaase

- Create the directory `/capstone/tmp/sample_records` (or make an alias to that past. Recommend using a RAM disk).
- Run `test_load_samples` in `test/test_features.py`

#### I Want to Prepare my Features for Training a Model

- Define the environment variable `SONG_FEATURES_PYARROW_DATABASE` to point towards the directory where samples (the `.pyarrow` files) were dumped.
- Then run `main.py`, and it will take care of the rest

#### I've Trained My Model, I Want to Compute ROC Statistics

- For the neural net, uncomment these lines at the bottom of `main.py`:

```
    # with open("validation_prediction.arrow", 'wb+') as f:
    #     f.write(pyarrow.serialize(y_pred).to_buffer())
```

    - Then run `nn_metrics.py`

- For the decision tree, uncomment these lines at the bottom of `main_decision_tree.py`:

```
    # with open("tree.pickle", "wb+") as f:
    #     pickle.dump(best_tree, f, protocol=pickle.HIGHEST_PROTOCOL)
```

    - Then run `tree_metrics.py`

- Precision, recall, F1, and combined metrics are computed as part of the training process.

### Miscalleaneous

- The model directory contains some logs of various runs of different stages of training neural networks. The hardware I was running on was not fully stable, thus the logs are not complete.
- `test_remove_similar_euclidean.py` can provide an interesting analysis of compute feature vectors. It will try to determine how similar a feature vector is to another vector by computing the euclidaen distance of each feature, normalizing them, and then summing them together.
    - This is a very memory expensive operation, so it's use case is mostly for explanitory.
