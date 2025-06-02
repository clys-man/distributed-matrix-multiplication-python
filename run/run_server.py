from multiprocessing import Process

from server.matrix_server import main

TOTAL_SERVERS = 10  # Número total de servidores que serão iniciados


def run_all():
    # Define a lista de servidores com host "localhost" e portas consecutivas
    SERVERS = [("localhost", 5000 + i) for i in range(TOTAL_SERVERS)]

    processes = []

    # Cria e inicia um processo separado para cada servidor
    for host, port in SERVERS:
        p = Process(
            target=main, args=(host, port)
        )  # Cada processo roda o servidor em uma porta específica
        p.start()
        processes.append(p)

    print("[INFO] Todos os servidores foram iniciados.")

    # Aguarda todos os processos terminarem (bloqueia o encerramento do script)
    for p in processes:
        p.join()


if __name__ == "__main__":
    run_all()
