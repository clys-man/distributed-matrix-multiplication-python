import pickle

import numpy as np

CHUNK_SIZE = 4096


def send_chunks(sock, data_bytes):
    total_len = len(data_bytes)
    sock.sendall(total_len.to_bytes(8, "big"))
    for i in range(0, total_len, CHUNK_SIZE):
        sock.sendall(data_bytes[i : i + CHUNK_SIZE])


def recv_chunks(sock):
    total_len = int.from_bytes(sock.recv(8), "big")
    received = bytearray()
    while len(received) < total_len:
        part = sock.recv(min(CHUNK_SIZE, total_len - len(received)))
        if not part:
            raise ConnectionError("Conexão perdida durante recebimento")
        received.extend(part)
    return bytes(received)


def serialize(data):
    return pickle.dumps(data)


def deserialize(data):
    return pickle.loads(data)


def divide_matrix(A, num_parts):
    return np.array_split(A, num_parts, axis=0)
