import socket
import threading
import time

import numpy as np

from common import deserialize, divide_matrix, serialize

N, M, P = 8, 6, 5  # A: 8x6, B: 6x5
HOST = "127.0.0.1"
PORT = 65432
SERVERS = [
    ("localhost", 5000),
    ("localhost", 5001),
    ("localhost", 5002),
    ("localhost", 5003),
]


def send_submatrix(index, sub_A, B, results):
    host, port = SERVERS[index]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall(serialize((sub_A, B)))

            data = deserialize(s.recv(8192))

            print(data)
            results[index] = data
            print(f"[CLIENTE] Submatriz {index} processada.")
    except Exception as e:
        print(f"[ERRO] Servidor {host}:{port} – {e}")


def main():
    A = np.random.randint(0, 10, (N, M))
    B = np.random.randint(0, 10, (M, P))
    sub_matrices = divide_matrix(A, len(SERVERS))

    print("[CLIENTE] Matriz A:")
    print(A)
    print("[CLIENTE] Matriz B:")
    print(B)

    threads = []
    results = [None] * len(SERVERS)

    start = time.time()
    for i in range(len(SERVERS)):
        t = threading.Thread(
            target=send_submatrix, args=(i, sub_matrices[i], B, results)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    C = np.vstack(results)
    end = time.time()

    print("[CLIENTE] Matriz Resultante C = A × B:")
    print(C)
    print(f"[TEMPO] Execução total: {end - start:.2f} segundos")


if __name__ == "__main__":
    main()
