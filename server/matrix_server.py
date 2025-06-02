import socket
import threading
import time

import numpy as np

from common.utils import deserialize, recv_chunks, send_chunks, serialize


class MatrixServer:
    def __init__(self, host="localhost", port=5001):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )  # Permite reuso da porta
        self.running = False

    def start(self):
        """
        Inicia o servidor, escutando conexões de clientes.
        Cada cliente é tratado em uma thread separada.
        """
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            print(f"[SERVIDOR] Servidor iniciado em {self.host}:{self.port}")
            print("[SERVIDOR] Aguardando conexões...")

            while self.running:
                conn, addr = self.server_socket.accept()
                # Cria uma nova thread para tratar o cliente conectado
                client_thread = threading.Thread(
                    target=self.handle_client, args=(conn, addr)
                )
                client_thread.daemon = True
                client_thread.start()

        except KeyboardInterrupt:
            print("\n[SERVIDOR] Servidor interrompido pelo usuário")
        except Exception as e:
            print(f"[ERRO] Erro no servidor: {e}")
        finally:
            self.shutdown()

    def handle_client(self, conn, addr):
        """
        Gerencia o ciclo de vida de um cliente:
        - Recebe os dados (submatriz de A e matriz B)
        - Aguarda o sinal 'START' para iniciar a computação
        - Realiza a multiplicação e envia o resultado
        """
        try:
            print(f"[SERVIDOR] Conexão estabelecida com {addr}")

            # Recebe os dados do cliente: submatriz de A e matriz B
            data = recv_chunks(conn)
            sub_A, B = deserialize(data)

            print(f"[SERVIDOR] Dados recebidos de {addr}, enviando confirmação...")

            # Envia confirmação de recebimento
            send_chunks(conn, serialize("ACK"))

            print("[SERVIDOR] Aguardando sinal de início da computação...")
            start_signal = conn.recv(4096)
            signal = deserialize(start_signal)

            if signal == "START":
                print("[SERVIDOR] Sinal de início recebido! Iniciando computação...")
                start_time = time.time()

                # Usa múltiplas threads para acelerar o cálculo da submatriz
                from concurrent.futures import ThreadPoolExecutor

                def compute_row_chunk(args):
                    row_chunk, B = args
                    return np.dot(row_chunk, B)  # Multiplicação da parte da submatriz

                num_threads = 4  # Número de threads internas para o cálculo local
                chunks = np.array_split(sub_A, num_threads, axis=0)

                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    results = list(
                        executor.map(
                            compute_row_chunk, [(chunk, B) for chunk in chunks]
                        )
                    )

                # Junta os resultados parciais em uma única matriz
                result = np.vstack(results)

                end_time = time.time()
                print(
                    f"[SERVIDOR] Computação concluída em {end_time - start_time:.4f} segundos"
                )
                print("[SERVIDOR] Resultado:")
                print(result)

                # Envia o resultado final ao cliente
                send_chunks(conn, serialize(result))
                print(f"[SERVIDOR] Resultado enviado para {addr}")

            else:
                print(f"[SERVIDOR] Sinal inválido recebido: {signal}")

        except Exception as e:
            print(f"[ERRO] Erro ao lidar com o cliente {addr}: {e}")
        finally:
            conn.close()
            print(f"[SERVIDOR] Conexão com {addr} encerrada")

    def shutdown(self):
        """
        Encerra o servidor e libera recursos.
        """
        self.running = False
        try:
            self.server_socket.close()
        except:
            pass
        print("[SERVIDOR] Servidor finalizado")


def main(host="localhost", port=5001):
    server = MatrixServer(host, port)
    server.start()


if __name__ == "__main__":
    import sys

    # Permite iniciar o servidor com argumentos de linha de comando
    if len(sys.argv) == 3:
        main(sys.argv[1], int(sys.argv[2]))
    elif len(sys.argv) == 2:
        main(port=int(sys.argv[1]))
    else:
        main()
