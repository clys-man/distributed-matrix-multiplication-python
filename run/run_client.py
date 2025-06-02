import numpy as np

from client.matrix_client import MatrixClient


def main():
    SERVERS = [
        ("localhost", 5000),
        ("localhost", 5001),
        ("localhost", 5002),
        ("localhost", 5003),
    ]
    N, M, P = 2000, 2000, 2000
    A = np.random.randint(0, 10, (N, M))
    B = np.random.randint(0, 10, (M, P))

    client = MatrixClient(SERVERS)
    client.compute(A, B)


if __name__ == "__main__":
    main()
