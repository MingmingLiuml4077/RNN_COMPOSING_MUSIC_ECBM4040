class Cache:
    """
    Originally, transforming each piece into a state matrix for input and output
    took nearly 2 seconds for each batch. Since our training data and model do not take
    that much space, we can instead simply store the transformed inputs and outputs
    in memory.
    """

    def __init__(self):
        self.input_storage = {}
        self.output_storage = {}
        self.keys_and_lengths = []
        self.size = 0

    def cache(self, in_matrix, out_matrix, piece_name):
        self.output_storage[piece_name] = out_matrix
        self.input_storage[piece_name] = in_matrix
        self.keys_and_lengths.append((piece_name, len(out_matrix)))
        self.size = len(self.keys_and_lengths)

    def get(self, piece_name, start, end):
        in_matrix = self.input_storage.get(piece_name)[start:end]
        out_matrix = self.output_storage.get(piece_name)[start:end]
        return in_matrix, out_matrix

    def save(self, fname="cache.pkl"):
        import pickle

        with open(fname, "wb") as f:
            pickle.dump(f, self)
