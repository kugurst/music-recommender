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
