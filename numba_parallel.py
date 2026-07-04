import numpy as np
import time
from numba import njit, prange, set_num_threads, get_num_threads


@njit(parallel=True)
def matrix_multiply_numba(A):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.int64)

    for i in prange(n):          # parallel di baris
        for j in range(n):
            total = 0

            for k in range(n):   # dot product
                total += A[i, k] * A[k, j]

            C[i, j] = total

    return C


@njit(parallel=True)
def find_max_each_row_numba(matrix):
    n = matrix.shape[0]
    max_values = np.zeros(n, dtype=np.int64)
    max_columns = np.zeros(n, dtype=np.int64)

    for i in prange(n):
        max_value = matrix[i, 0]
        max_col = 0

        for j in range(1, n):
            if matrix[i, j] > max_value:
                max_value = matrix[i, j]
                max_col = j

        max_values[i] = max_value
        max_columns[i] = max_col

    return max_values, max_columns


def main():
    print("=" * 45)
    print(" MATRIX MULTIPLICATION - NUMBA PARALLEL")
    print("=" * 45)

    ukuran = int(input("Masukkan ukuran matriks: "))
    jumlah_thread = int(input("Masukkan jumlah thread: "))

    set_num_threads(jumlah_thread)

    nama_file = f"similarity_matrix_{ukuran}.csv"

    print("\nMembaca matriks...")

    matrix = np.loadtxt(
        nama_file,
        delimiter=",",
        dtype=np.int64
    )

    print("Warm-up JIT compiler...")

    # warm-up supaya waktu compile Numba tidak ikut dihitung
    dummy = matrix[:2, :2]
    _ = matrix_multiply_numba(dummy)

    print("Perkalian matriks dimulai...\n")

    start = time.perf_counter()

    result = matrix_multiply_numba(matrix)
    max_values, max_columns = find_max_each_row_numba(result)

    finish = time.perf_counter()

    execution_time = finish - start

    print("=" * 45)
    print("HASIL")
    print("=" * 45)
    print(f"Ukuran Matriks : {ukuran} x {ukuran}")
    print(f"Jumlah Thread  : {get_num_threads()}")
    print(f"Waktu Eksekusi : {execution_time:.6f} detik")

    print("\n10 Baris Pertama Hasil Maksimum\n")

    for i in range(min(10, ukuran)):
        print(
            f"User {i+1:4} -> "
            f"Max = {max_values[i]:8} "
            f"Kolom = {max_columns[i]+1}"
        )


if __name__ == "__main__":
    main()