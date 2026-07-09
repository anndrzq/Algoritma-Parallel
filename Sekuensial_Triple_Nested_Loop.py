import numpy as np
import time
from pathlib import Path


def matrix_multiply(A):
    n = len(A)

    C = np.zeros((n, n), dtype=np.int64)

    for i in range(n):
        for j in range(n):
            total = 0

            for k in range(n):
                total += A[i][k] * A[k][j]

            C[i][j] = total

    return C


def find_max_each_row(matrix):
    n = matrix.shape[0]
    max_values = np.zeros(n, dtype=np.int64)
    max_columns = np.zeros(n, dtype=np.int64)

    for i in range(n):
        # Inisialisasi dengan kolom pertama yang bukan diagonal (i)
        if i == 0:
            max_value = matrix[i, 1]
            max_col = 1
            start_j = 2
        else:
            max_value = matrix[i, 0]
            max_col = 0
            start_j = 1

        for j in range(start_j, n):
            if j == i:
                continue
            if matrix[i, j] > max_value:
                max_value = matrix[i, j]
                max_col = j

        max_values[i] = max_value
        max_columns[i] = max_col

    return max_values, max_columns


def main():

    print("=" * 45)
    print(" MATRIX MULTIPLICATION - SEQUENTIAL")
    print("=" * 45)

    ukuran = int(input("Masukkan ukuran matriks: "))

    nama_file = Path(__file__).with_name(f"similarity_matrix_{ukuran}.csv")

    if not nama_file.exists():
        raise FileNotFoundError(
            f"{nama_file.name} tidak ditemukan. "
            f"Jalankan dulu: py Matrix_Similarity.py lalu masukkan ukuran {ukuran}."
        )

    print("\nMembaca matriks...")

    matrix = np.loadtxt(
        nama_file,
        delimiter=",",
        dtype=np.int16
    )

    print("Perkalian matriks dimulai...\n")

    start = time.perf_counter()

    result = matrix_multiply(matrix)

    max_values, max_columns = find_max_each_row(result)

    finish = time.perf_counter()

    execution_time = finish - start

    print("=" * 45)
    print("HASIL")
    print("=" * 45)

    print(f"Ukuran Matriks : {ukuran} x {ukuran}")
    print(f"Waktu Eksekusi : {execution_time:.6f} detik")

    print("\n10 Baris Pertama Hasil Maksimum\n")

    for i in range(min(10, ukuran)):
        print(
            f"User {i+1:4} -> "
            f"User {max_columns[i]+1:4} "
            f"dengan nilai kemiripan kolaboratif Max = {max_values[i]:8}"
        )

if __name__ == "__main__":
    main()
