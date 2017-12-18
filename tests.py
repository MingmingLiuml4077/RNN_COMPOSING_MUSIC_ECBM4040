import data
import numpy as np
import os
import time

from cache import Cache


def test_cache(composers):
    all_pieces = dict()
    for composer in composers:
        start = time.time()
        pieces = data.getpices(path="../midis", composer=composer)
        print("Loading {} {} pieces took {} seconds".format(
            composer, len(pieces), time.time() - start
        ))
        all_pieces.update(pieces)

    midi_cache = data.initialize_cache(all_pieces)
    gen = data.generate_batch(midi_cache, batch_size=10)

    for i in range(5):
        pre_start = time.time()
        batch_in, batch_out = next(gen)

        print("One batch took {}".format(time.time() - pre_start))
        print("Batch iput is size: ({}, {}, {})".format(
            len(batch_in), len(batch_in[0]), len(batch_in[0][0])))
        print("Batch output is size: ({}, {}, {})".format(
            len(batch_out), len(batch_out[0]), len(batch_out[0][0])))


def test_translate():
    # Simulate a piece matrix of length 100 timesteps and test if translate
    # shifts the matrix down and up correctly across the note axis
    rand_matrix = np.random.rand(100*78*2).reshape(100, 78, 2)
    translated_matrix = data.translate(rand_matrix, direction="up")
    assert translated_matrix[0][1:] == rand_matrix[0][:-1]

    translated_matrix_d = data.translate(rand_matrix, direction="down")
    assert translated_matrix_d[0][:-1] == rand_matrix[0][1:]

def test_translate_np():
    # Simulate a piece matrix of length 100 timesteps and test if translate
    # shifts the matrix down and up correctly across the note axis
    rand_matrix = np.random.rand(100*78*2).reshape(100, 78, 2)
    translated_matrix = data.translate(rand_matrix, direction="up")
    assert translated_matrix[0][1:] == rand_matrix[0][:-1]

    translated_matrix_d = data.translate(rand_matrix, direction="down")
    assert translated_matrix_d[0][:-1] == rand_matrix[0][1:]
