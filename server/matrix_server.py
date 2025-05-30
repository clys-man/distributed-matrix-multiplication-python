import socket
import threading

import numpy as np

from common.utils import deserialize, serialize


class MatrixServer:
    def __init__(self, host="localhost", port=5001):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.running = False

    def start(self):
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True

            print(f"[SERVIDOR] Servidor iniciado em {self.host}:{self.port}")
            print("[SERVIDOR] Aguardando conexões...")

            while self.running:
                conn, addr = self.server_socket.accept()
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
        try:
            print(f"[SERVIDOR] Conexão estabelecida com {addr}")
            data = conn.recv(8192)
            sub_A, B = deserialize(data)
            result = np.dot(sub_A, B)
            print(result)
            conn.sendall(serialize(result))
        except Exception as e:
            print(f"[ERRO] Erro ao lidar com o cliente {addr}: {e}")
        finally:
            conn.close()

    def shutdown(self):
        self.running = False
        self.server_socket.close()
        print("[SERVIDOR] Servidor finalizado")


def main(host="localhost", port=5001):
    server = MatrixServer(host, port)
    server.start()


if __name__ == "__main__":
    main()
