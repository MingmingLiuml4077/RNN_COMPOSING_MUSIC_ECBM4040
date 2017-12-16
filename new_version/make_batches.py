def make_batches1(data, data1, batch_size=5):
    batch_num = int(len(data) / batch_size)
    X_train = []
    y_train = []
    for i in range(batch_num):
        X_train.append(data[i * batch_size:(i + 1) * batch_size])
        y_train.append(data1[i * batch_size:(i + 1) * batch_size])
    return [X_train, y_train]

def make_batches2(data, data1, batch_size=5):
    return None