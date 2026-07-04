import numpy as np
import time


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
    max_values = []
    max_columns = []

    for i in range(len(matrix)):
        max_value = matrix[i][0]
        max_col = 0

        for j in range(1, len(matrix[i])):
            if matrix[i][j] > max_value:
                max_value = matrix[i][j]
                max_col = j

        max_values.append(max_value)
        max_columns.append(max_col)

    return max_values, max_columns


def main():

    print("=" * 45)
    print(" MATRIX MULTIPLICATION - SEQUENTIAL")
    print("=" * 45)

    ukuran = int(input("Masukkan ukuran matriks: "))

    nama_file = f"similarity_matrix_{ukuran}.csv"

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
            f"Max = {max_values[i]:8} "
            f"Kolom = {max_columns[i]+1}"
        )


if __name__ == "__main__":
    main()