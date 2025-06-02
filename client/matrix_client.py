import socket
import threading

import numpy as np

from common.utils import deserialize, divide_matrix, recv_chunks, send_chunks, serialize


class MatrixClient:
    def __init__(self, servers):
        self.servers = servers  # Lista de servidores (host, port) disponíveis
        self.results = [None] * len(servers)  # Armazena os resultados de cada submatriz
        self.sync_barrier = threading.Barrier(
            len(servers)
        )  # Barreira para sincronizar threads
        self.sync_event = (
            threading.Event()
        )  # Evento opcional de sincronização (não usado aqui)

    def send_submatrix(self, index, sub_A, B):
        host, port = self.servers[index]
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))

                # Envia a submatriz de A e a matriz B ao servidor
                send_chunks(s, serialize((sub_A, B)))

                # Aguarda confirmação de recebimento dos dados
                ack = deserialize(recv_chunks(s))
                if ack != "ACK":
                    raise Exception("Falha na confirmação de recebimento")

                print(f"[CLIENTE] Servidor {index} pronto, aguardando sincronização...")

                # Espera todos os servidores estarem prontos antes de iniciar o cálculo
                self.sync_barrier.wait()

                # Envia sinal de início da computação para o servidor
                s.sendall(serialize("START"))
                print(f"[CLIENTE] Sinal de início enviado para servidor {index}")

                # Recebe o resultado da multiplicação parcial do servidor
                data = deserialize(recv_chunks(s))
                self.results[index] = data
                print(f"[CLIENTE] Submatriz {index} processada.")

        except Exception as e:
            print(f"[ERRO] Servidor {host}:{port} – {e}")
            # Libera a barreira em caso de erro para evitar deadlock das outras threads
            try:
                self.sync_barrier.wait()
            except:
                pass

    def compute(self, A, B):
        # Garante que as dimensões das matrizes são compatíveis para multiplicação
        assert A.shape[1] == B.shape[0], (
            "Número de colunas de A deve ser igual ao número de linhas de B"
        )

        # Divide a matriz A em submatrizes para distribuir entre os servidores
        sub_matrices = divide_matrix(A, len(self.servers))
        threads = []

        # Cria uma thread para cada servidor, enviando uma parte da matriz A
        for i in range(len(self.servers)):
            t = threading.Thread(
                target=self.send_submatrix, args=(i, sub_matrices[i], B)
            )
            threads.append(t)
            t.start()

        # Aguarda todas as threads finalizarem
        for t in threads:
            t.join()

        # Junta os resultados recebidos em uma única matriz C
        C = np.vstack(self.results)
        return C
