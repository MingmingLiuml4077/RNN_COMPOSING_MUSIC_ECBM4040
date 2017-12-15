import model
import data

if __name__ == '__main__':
    
    # load midi files as statematrix, if no file will download from "http://www.piano-midi.de"
    pieces = data.getpices(mode='partial') 
    
    # Building model
    print('Building model')
    music_model = model.biaxial_model(t_layer_sizes=[300,300], n_layer_sizes=[100,50])
    
    print('Start training')
    music_model.train(pieces,batch_size=5,max_epoch=1000,predict_freq=100)