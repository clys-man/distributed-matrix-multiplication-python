from multiprocessing import Process

from server import start_server

SERVERS = [
    ("localhost", 5000),
    ("localhost", 5001),
    ("localhost", 5002),
    ("localhost", 5003),
]


def run_all():
    processes = []
    for host, port in SERVERS:
        p = Process(target=start_server, args=(host, port))
        p.start()
        processes.append(p)

    print("[INFO] Todos os servidores foram iniciados.")

    for p in processes:
        p.join()


if __name__ == "__main__":
    run_all()
