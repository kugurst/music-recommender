# Feb 9, 2019 #

- Created the bad playlist
    - Songs that weren't in the bad playlist, either by filename or by title
    - Removed songs that had a playcount of greater than 6
    - TODO: Investigate if I should remove "TV Size" editions of songs
- Created the good playlist
    - Songs with a play count of greater than 5
- Converting to:
    - AIFF (python support seems really good)
    - 16-bit depth
    - 44.1 khz sample rate
    - Stereo


# Feb 10, 2019 #

- Cleaning up songs that are good (i.e. are in the good playlist, but have a different case, slightly different album
 name, different language, etc)
- Removed "TV Edit songs"
- Removed full soundtrack files
- Removed songs over 1 GB in size
- Didn't remove "vocal" and "instrumental" edition of tracks; this might provide the algorithm some crucial insight 
(or it may confuse it)


# Feb 11, 2019 #

- Started writing music retrieval strategies
    - Made a database to store links to songs
    - Started writing the preprocessing logic
    
# Feb 13, 2019 #

- Completed work on the preprocessing logic
- Decided to store both forms of representation: the waveforms and the spectrum

# Feb 14, 2019 #

- Made the database storage multi-process
- Debugged some ffmpeg issues and discovered that they were file system issues
    - Wrote a script to cleanup file names
- Debugged some long path issues, which I resolved at the file loading level for ffmpeg

# Feb 15, 2019 #

- Added a function to update the records of an arbitrary collection with a default value for a given field
- Added an "FFT_BUILT" section to the song database
- Discovered that unqlite is not properly retrieving/storing pickled data. Will need to change backend

# Feb 16, 2019 #

- Found a way to fix the DB issue by encoding the binary data in base64 before writing to the database
- It takes about 57 minutes to process the entire music library, written to my fastest SSD
- Benchmarked and realized it only takes about a second to compute the FFTs and spectrograms each of a given song for
 15 samples, which are each 5 seconds long.
    - Savings in somputation space don't seem commiserate with disk space usage and eventual pipelining 
    implementation, abandoning storing the data like this
    - It only takes 36 minutes in this case

# Feb 17, 2019 #

- Adding the sample rate of songs to the database
- Switched over to using leveldb for the song samples database
    - It appears to build and fetch faster than unqlite

# Feb 18, 2019 #

- Wrote a routine to take a song id and sample number, and generate a wavfile from it

# Feb 19, 2019 #

- Installed delegator for cleaner subprocess control

# Feb 20, 2019 #

- Found that jAudio's command line interface sucks, and is slow. Not usable

- Reviewing marsys filters:
    - Chroma uses fixed chords
        - It measures around piano frequencies, which is possibly where I want to be, but it doesn't go as low as 200 Hz
        - https://musicinformationretrieval.com/chroma.html
    - The centroid uses the output of the FFT to determine where the spectral mean of the audio is
        - https://en.wikipedia.org/wiki/Spectral_centroid

# Feb 21, 2019 #

- Started using librosa for MIR, instead of the marsys
- Started collecting and benchmarking some features to use
    - They all run significantly faster than their marsys equivalent

# Feb 22, 2019 #

- Fleshed out the selection of features to use
- Created the skeleton of the NN pipeline
