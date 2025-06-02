import time

import numpy as np

from client.matrix_client import MatrixClient

TOTAL_SERVERS = 10  # Número total de servidores a serem utilizados


def main():
    # Define a lista de servidores como localhost em portas consecutivas
    SERVERS = [("localhost", 5000 + i) for i in range(TOTAL_SERVERS)]

    # Define as dimensões das matrizes A (N x M) e B (M x P)
    N, M, P = 100, 100, 100
    A = np.random.randint(0, 10, (N, M))  # Matriz A com valores aleatórios de 0 a 9
    B = np.random.randint(0, 10, (M, P))  # Matriz B com valores aleatórios de 0 a 9

    # Multiplicação local (referência para benchmarking)
    print("[BENCHMARK] Iniciando multiplicação local...")
    start_local = time.time()
    C_local = A @ B  # Multiplicação usando numpy (otimizada localmente)
    end_local = time.time()
    temp_local = end_local - start_local
    print(f"[BENCHMARK] Tempo da multiplicação local: {temp_local:.2f} segundos")

    # Multiplicação distribuída usando os servidores definidos
    print("\n[DISTRIBUÍDO] Iniciando cliente distribuído...")
    start_dist = time.time()
    client = MatrixClient(SERVERS)
    C_dist = client.compute(A, B)  # Computação paralela distribuída
    end_dist = time.time()
    temp_disp = end_dist - start_dist
    print(f"[BENCHMARK] Tempo da multiplicação distribuída: {temp_disp:.2f} segundos")

    # Verifica se o resultado distribuído é idêntico ao local
    iguais = np.array_equal(C_local, C_dist)
    print(f"\n[COMPARAÇÃO] Resultados são iguais? {'✅ SIM' if iguais else '❌ NÃO'}")
    print(
        f"[COMPARAÇÃO] Diferença máxima: {np.max(np.abs(C_local - C_dist)) if not iguais else 0}"
    )

    # Resumo dos tempos
    print("\n[RESUMO]")
    print(f"- Tempo local:       {temp_local:.2f} segundos")
    print(f"- Tempo distribuído: {temp_disp:.2f} segundos")


if __name__ == "__main__":
    main()
