import model_tb as model
import data

import os
import pickle
import sys


if __name__ == '__main__':

    # Check for the existence of previous cache and models
    cache_name = sys.argv[-2]
    model_name = sys.argv[-1]

    if not os.path.exists(cache_name):
        composers = input("Enter composers separated by spaces: ").split()
        all_pieces = {}
        for c in composers:
            all_pieces.update(data.getpices(path="../midis", composer=c))

        cache = data.initialize_cache(all_pieces, save_loc=cache_name)
    else:
        with open(cache_name, 'rb') as f:
            cache = pickle.load(f)

    # Build and load the pre-existing model if it exists
    print('Building model')
    music_model = model.biaxial_model(t_layer_sizes=[300,300], n_layer_sizes=[100,50])

    print('Start training')
    music_model.train(cache, batch_size=5, max_epoch=10000, predict_freq=100, pre_trained_model=model_name)
