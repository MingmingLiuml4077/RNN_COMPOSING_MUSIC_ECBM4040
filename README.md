# RNN_COMPOSING_MUSIC_ECBM4040
This is a course project of ECBM4040, Columbia University.  
This code implements a recurrent neural network trained to generate classical music. The model, which uses LSTM layers and draws inspiration from convolutional neural networks, learns to predict which notes will be played at each time step of a musical piece.
## Requirement
The code is written in python, and used TensorFlow deep learning frame work.  
To load the data miod package is required, and to download midi files from internet bs4 package is also required.
```
pip install mido
pip install bs4
pip install lxml
```

## Use it
To train the model run _main.py_, or _main.ipynb_. You could change the arguments in it to train a new model or restore a pre-trained model. While training the model will generate new songs, which will be saved to 'NewSong/', you could also modified the arguments to change the frequency of composing a new song. If you don't have the data 
needed to train or predict, the main.py will automatically download from http://www.piano-midi.de/, and save them to 'midis/'. While training model data will be saved to the 'model/'.
  
The predict.py is used to predict new songs. It will restore a trained model, and predict new songs base on that. You need to specific the name 
of the trained model, and make sure the model data is save to 'model/'.

model.py is for the biaxial model class. There are twe arguments for it, 'cache name' and 'model name'

data.py is used to clean data for the model.  

midi_scraper.py is for scrape midi files.  

midi_to_statematrix.py is to read the midi files and translate it to not state matrix.

operations.py is for truning a python function to tensorflow operation.

cache.py is used to save the input format of statematrix, which enables us train faster whithout translating data every time.

In the folder _best_samples_, there are samples of midi files the moel generated.

