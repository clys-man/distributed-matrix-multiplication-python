from multiprocessing import Process

from server.matrix_server import main

TOTAL_SERVERS = 10


def run_all():
    SERVERS = [("localhost", 5000 + i) for i in range(TOTAL_SERVERS)]

    processes = []
    for host, port in SERVERS:
        p = Process(target=main, args=(host, port))
        p.start()
        processes.append(p)

    print("[INFO] Todos os servidores foram iniciados.")

    for p in processes:
        p.join()


if __name__ == "__main__":
    run_all()
