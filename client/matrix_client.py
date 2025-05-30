import socket
import threading
import time

import numpy as np

from common.utils import deserialize, divide_matrix, serialize


class MatrixClient:
    def __init__(self, servers):
        self.servers = servers
        self.results = [None] * len(servers)

    def send_submatrix(self, index, sub_A, B):
        host, port = self.servers[index]
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))
                s.sendall(serialize((sub_A, B)))
                data = deserialize(s.recv(8192))
                self.results[index] = data
                print(f"[CLIENTE] Submatriz {index} processada.")
        except Exception as e:
            print(f"[ERRO] Servidor {host}:{port} – {e}")

    def compute(self, A, B):
        assert A.shape[1] == B.shape[0], (
            "Número de colunas de A deve ser igual ao número de linhas de B"
        )
        sub_matrices = divide_matrix(A, len(self.servers))

        print("[CLIENTE] Matriz A:")
        print(A)
        print("[CLIENTE] Matriz B:")
        print(B)

        print("[CLIENTE] Resultado esperado (A @ B):")
        print(A @ B)

        threads = []
        start = time.time()

        for i in range(len(self.servers)):
            t = threading.Thread(
                target=self.send_submatrix, args=(i, sub_matrices[i], B)
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        C = np.vstack(self.results)
        end = time.time()

        print("[CLIENTE] Matriz Resultante C = A × B:")
        print(C)
        print(f"[TEMPO] Execução total: {end - start:.2f} segundos")
        return C
