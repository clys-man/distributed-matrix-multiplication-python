import socket
import threading

import numpy as np

from common.utils import deserialize, divide_matrix, recv_chunks, send_chunks, serialize


class MatrixClient:
    def __init__(self, servers):
        self.servers = servers
        self.results = [None] * len(servers)
        self.sync_barrier = threading.Barrier(len(servers))
        self.sync_event = threading.Event()

    def send_submatrix(self, index, sub_A, B):
        host, port = self.servers[index]
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))

                # Envia dados da submatriz
                send_chunks(s, serialize((sub_A, B)))

                # Aguarda confirmação de recebimento
                ack = deserialize(recv_chunks(s))

                if ack != "ACK":
                    raise Exception("Falha na confirmação de recebimento")

                print(f"[CLIENTE] Servidor {index} pronto, aguardando sincronização...")

                # Aguarda todos os servidores estarem prontos
                self.sync_barrier.wait()

                # Envia sinal de início da computação
                s.sendall(serialize("START"))
                print(f"[CLIENTE] Sinal de início enviado para servidor {index}")

                # Recebe resultado
                data = deserialize(recv_chunks(s))

                self.results[index] = data
                print(f"[CLIENTE] Submatriz {index} processada.")

        except Exception as e:
            print(f"[ERRO] Servidor {host}:{port} – {e}")
            # Em caso de erro, libera a barreira para evitar deadlock
            try:
                self.sync_barrier.wait()
            except:
                pass

    def compute(self, A, B):
        assert A.shape[1] == B.shape[0], (
            "Número de colunas de A deve ser igual ao número de linhas de B"
        )

        sub_matrices = divide_matrix(A, len(self.servers))

        threads = []

        for i in range(len(self.servers)):
            t = threading.Thread(
                target=self.send_submatrix, args=(i, sub_matrices[i], B)
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        C = np.vstack(self.results)

        return C
