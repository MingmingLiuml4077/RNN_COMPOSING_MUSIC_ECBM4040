# For data preprocessing, we just used the codes provided by Daniel Johnson here: https://github.com/hexahedria/biaxial-rnn-music-composition/blob/master/data.py

import itertools
from midi_to_statematrix import *
import os
import random

def startSentinel():
    def noteSentinel(note):
        position = note
        part_position = [position]

        pitchclass = (note + lowerBound) % 12
        part_pitchclass = [int(i == pitchclass) for i in range(12)]

        return part_position + part_pitchclass + [0] * 66 + [1]

    return [noteSentinel(note) for note in range(upperBound - lowerBound)]


def getOrDefault(l, i, d):
    try:
        return l[i]
    except IndexError:
        return d


def buildContext(state):
    context = [0] * 12
    for note, notestate in enumerate(state):
        if notestate[0] == 1:
            pitchclass = (note + lowerBound) % 12
            context[pitchclass] += 1
    return context


def buildBeat(time):
    return [2 * x - 1 for x in [time % 2, (time // 2) % 2, (time // 4) % 2, (time // 8) % 2]]


def noteInputForm(note, state, context, beat):
    position = note
    part_position = [position]

    pitchclass = (note + lowerBound) % 12
    part_pitchclass = [int(i == pitchclass) for i in range(12)]
    part_prev_vicinity = list(
        itertools.chain.from_iterable((getOrDefault(state, note + i, [0, 0]) for i in range(-12, 13))))

    part_context = context[pitchclass:] + context[:pitchclass]

    return part_position + part_pitchclass + part_prev_vicinity + part_context + beat + [0]


def noteStateSingleToInputForm(state, time):
    beat = buildBeat(time)
    context = buildContext(state)
    return [noteInputForm(note, state, context, beat) for note in range(len(state))]


def noteStateMatrixToInputForm(statematrix):
    inputform = [noteStateSingleToInputForm(state, time) for time, state in enumerate(statematrix)]
    return inputform

def getpices(path='midis', midi_len=128, mode='all',composer=None):
    
    pieces = {}
    if not os.path.exists(path):
        # Download midi files
        import midi_scraper
    song_count = 0

    for composer_name in os.listdir(path):
        if composer is not None and composer_name not in composer: continue
        for fname in os.listdir(path+'/'+composer_name):
            if fname[-4:] not in ('.mid','.MID'):
                continue

            name = fname[:-4]
        
            outMatrix = midiToNoteStateMatrix(os.path.join(path, composer_name, fname))
            if len(outMatrix) < midi_len:
                continue
        
            pieces[name] = outMatrix
            song_count += 1
            print ("Loaded {}-{}".format(composer_name, fname))
            if mode != 'all':
                if song_count >= 10: 
                    print ("{} songs are loaded".format(song_count))
                    return pieces
    print ("{} songs are loaded".format(song_count))
    return pieces


    
def getPieceSegment(pieces, piece_length=128, measure_len=16):
    # puece_length means the number of ticks in a training sample, measure_len means number of ticks in a measure
    piece_output = random.choice(list(pieces.values()))
    
    # We just need a segment of a piece as train data, and we want the start of a sample is the start of a measure
    start = random.randrange(0,len(piece_output)-piece_length,measure_len)

    seg_out = piece_output[start:start+piece_length]
    seg_in = noteStateMatrixToInputForm(seg_out)

    return seg_in, seg_out

def generate_batch(pieces, batch_size, piece_length=128):
    while True:
        i,o = zip(*[getPieceSegment(pieces,piece_length) for _ in range(batch_size)])
        yield(i,o)

