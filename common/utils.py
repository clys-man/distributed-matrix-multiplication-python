import pickle

import numpy as np

CHUNK_SIZE = 4096  # Tamanho do bloco para envio e recebimento de dados via socket


def send_chunks(sock, data_bytes):
    """
    Envia os dados serializados em blocos (chunks) via socket.
    Primeiro envia o tamanho total dos dados (8 bytes),
    depois envia os dados em partes de CHUNK_SIZE.
    """
    total_len = len(data_bytes)
    sock.sendall(
        total_len.to_bytes(8, "big")
    )  # Envia o tamanho dos dados como um inteiro de 8 bytes
    for i in range(0, total_len, CHUNK_SIZE):
        sock.sendall(data_bytes[i : i + CHUNK_SIZE])  # Envia os dados em blocos


def recv_chunks(sock):
    """
    Recebe dados via socket em chunks, reconstruindo o conteúdo original.
    Primeiro lê o tamanho total esperado dos dados,
    depois recebe os dados em partes até completar o total.
    """
    total_len = int.from_bytes(
        sock.recv(8), "big"
    )  # Lê o tamanho total dos dados (8 bytes)
    received = bytearray()
    while len(received) < total_len:
        part = sock.recv(
            min(CHUNK_SIZE, total_len - len(received))
        )  # Recebe um bloco de dados
        if not part:
            raise ConnectionError("Conexão perdida durante recebimento")
        received.extend(part)  # Adiciona o bloco ao total recebido
    return bytes(received)  # Retorna os dados completos como bytes


def serialize(data):
    return pickle.dumps(data)


def deserialize(data):
    return pickle.loads(data)


def divide_matrix(A, num_parts):
    return np.array_split(A, num_parts, axis=0)
