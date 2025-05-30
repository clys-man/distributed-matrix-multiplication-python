import pickle

import numpy as np


def serialize(data):
    return pickle.dumps(data)


def deserialize(data):
    return pickle.loads(data)


def divide_matrix(A, num_parts):
    return np.array_split(A, num_parts, axis=0)
