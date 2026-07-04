from mpi4py import MPI
import numpy as np
import time


def matrix_multiply_rows(A_rows, B):
    local_rows = A_rows.shape[0]
    n = B.shape[0]

    C_rows = np.zeros((local_rows, n), dtype=np.int64)

    for i in range(local_rows):
        for j in range(n):
            total = 0
            for k in range(n):
                total += A_rows[i, k] * B[k, j]
            C_rows[i, j] = total

    return C_rows


def find_max_each_row(matrix):
    n = matrix.shape[0]

    max_values = np.zeros(n, dtype=np.int64)
    max_columns = np.zeros(n, dtype=np.int64)

    for i in range(n):
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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    A = None
    n = None
    rows_per_process = None
    sendcounts = None
    displs = None

    if rank == 0:
        print("=" * 45)
        print(" MATRIX MULTIPLICATION - MPI4PY")
        print("=" * 45)

        ukuran = int(input("Masukkan ukuran matriks: "))
        nama_file = f"similarity_matrix_{ukuran}.csv"

        print("\nMembaca matriks...")

        A = np.loadtxt(
            nama_file,
            delimiter=",",
            dtype=np.int64
        )

        n = A.shape[0]

        if A.shape[0] != A.shape[1]:
            raise ValueError("Matriks harus berbentuk persegi.")

        if ukuran != n:
            raise ValueError("Ukuran input tidak sesuai dengan ukuran file matriks.")

        rows_per_process = [n // size] * size

        for i in range(n % size):
            rows_per_process[i] += 1

        sendcounts = np.array(
            [rows * n for rows in rows_per_process],
            dtype=np.int32
        )

        displs = np.zeros(size, dtype=np.int32)

        for i in range(1, size):
            displs[i] = displs[i - 1] + sendcounts[i - 1]

    n = comm.bcast(n, root=0)

    if rank != 0:
        A = np.empty((n, n), dtype=np.int64)

    comm.Bcast(A, root=0)

    rows_per_process = comm.bcast(rows_per_process, root=0)
    sendcounts = comm.bcast(sendcounts, root=0)
    displs = comm.bcast(displs, root=0)

    local_rows = rows_per_process[rank]
    local_A = np.empty((local_rows, n), dtype=np.int64)

    comm.Scatterv(
        [A, sendcounts, displs, MPI.INT64_T],
        local_A,
        root=0
    )

    comm.Barrier()

    if rank == 0:
        print("Perkalian matriks dimulai...\n")
        start = time.perf_counter()

    local_C = matrix_multiply_rows(local_A, A)

    recv_C = None

    if rank == 0:
        recv_C = np.empty((n, n), dtype=np.int64)

    comm.Gatherv(
        local_C,
        [recv_C, sendcounts, displs, MPI.INT64_T],
        root=0
    )

    comm.Barrier()

    if rank == 0:
        max_values, max_columns = find_max_each_row(recv_C)

        finish = time.perf_counter()
        execution_time = finish - start

        print("=" * 45)
        print("HASIL")
        print("=" * 45)
        print(f"Ukuran Matriks : {n} x {n}")
        print(f"Jumlah Proses  : {size}")
        print(f"Waktu Eksekusi : {execution_time:.6f} detik")

        print("\n10 Baris Pertama Hasil Maksimum\n")

        for i in range(min(10, n)):
            print(
                f"User {i+1:4} -> "
                f"Max = {max_values[i]:8} "
                f"Kolom = {max_columns[i]+1}"
            )


if __name__ == "__main__":
    main()