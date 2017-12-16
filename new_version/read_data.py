from midi_to_statematrix import midiToNoteStateMatrix
from data_preprocessing import noteStateMatrixToInputForm

def read_data(music_list):
    data1 = []
    data = []
    for music in music_list:
        y_tmp = midiToNoteStateMatrix(music)
        X_tmp = noteStateMatrixToInputForm(y_tmp)
        data1.extend(y_tmp)
        data.extend(X_tmp)
    return [data,data1]


