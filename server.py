import socket
import threading

import numpy as np

from common import deserialize, serialize


def handle_client(conn, addr):
    try:
        print(f"[SERVIDOR] Conexão estabelecida com {addr}")

        data = conn.recv(8192)
        sub_A, B = deserialize(data)
        result = np.dot(sub_A, B)
        print(result)

        conn.sendall(serialize(result))

    except Exception as e:
        raise e

    finally:
        conn.close()


def start_server(host, port):
    """
    Inicia o servidor e aguarda conexões
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen(5)

            print(f"[SERVIDOR] Servidor iniciado em {host}:{port}")
            print("[SERVIDOR] Aguardando conexões...")

            while True:
                conn, addr = s.accept()

                client_thread = threading.Thread(
                    target=handle_client, args=(conn, addr)
                )
                client_thread.daemon = True
                client_thread.start()

    except KeyboardInterrupt:
        print("\n[SERVIDOR] Servidor interrompido pelo usuário")
    except Exception as e:
        print(f"[ERRO] Erro no servidor: {e}")
    finally:
        socket.close()


def main():
    start_server("localhost", 5001)


if __name__ == "__main__":
    main()
