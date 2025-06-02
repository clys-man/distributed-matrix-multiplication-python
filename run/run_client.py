import time

import numpy as np

from client.matrix_client import MatrixClient

TOTAL_SERVERS = 10


def main():
    SERVERS = [("localhost", 5000 + i) for i in range(TOTAL_SERVERS)]

    N, M, P = 2000, 2000, 2000
    A = np.random.randint(0, 10, (N, M))
    B = np.random.randint(0, 10, (M, P))

    print("[BENCHMARK] Iniciando multiplicação local...")
    start_local = time.time()
    C_local = A @ B
    end_local = time.time()
    temp_local = end_local - start_local
    print(f"[BENCHMARK] Tempo da multiplicação local: {temp_local:.2f} segundos")

    print("\n[DISTRIBUÍDO] Iniciando cliente distribuído...")
    start_dist = time.time()
    client = MatrixClient(SERVERS)
    C_dist = client.compute(A, B)
    end_dist = time.time()
    temp_disp = end_dist - start_dist
    print(f"[BENCHMARK] Tempo da multiplicação distribuída: {temp_disp:.2f} segundos")

    iguais = np.array_equal(C_local, C_dist)
    print(f"\n[COMPARAÇÃO] Resultados são iguais? {'✅ SIM' if iguais else '❌ NÃO'}")
    print(
        f"[COMPARAÇÃO] Diferença máxima: {np.max(np.abs(C_local - C_dist)) if not iguais else 0}"
    )

    print("\n[RESUMO]")
    print(f"- Tempo local:       {temp_local:.2f} segundos")
    print(f"- Tempo distribuído: {temp_disp:.2f} segundos")


if __name__ == "__main__":
    main()
