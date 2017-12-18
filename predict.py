import model
import data

if __name__ == '__main__':
    
    # load midi files as statematrix, if no file will download from "http://www.piano-midi.de"
    pieces = data.getpices(mode='all') 
    
    # Building model
    print('Building model')
    music_model = model.biaxial_model(t_layer_sizes=[300,300], n_layer_sizes=[100,50])
    
    print('Start predicting')
    music_model.predict(pieces,'biaxial_rnn_1513531465',step=320,conservativity=1,n=20)
