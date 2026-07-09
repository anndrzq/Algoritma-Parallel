import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from numba import get_num_threads, set_num_threads

from Sekuensial_Triple_Nested_Loop import find_max_each_row, matrix_multiply
from numba_parallel import (
    find_max_each_row_numba,
    matrix_multiply_numba,
)


def load_matrix(ukuran):
    nama_file = Path(__file__).with_name(f"similarity_matrix_{ukuran}.csv")

    if not nama_file.exists():
        raise FileNotFoundError(
            f"{nama_file.name} tidak ditemukan. "
            f"Jalankan dulu: py Matrix_Similarity.py lalu masukkan ukuran {ukuran}."
        )

    matrix = np.loadtxt(nama_file, delimiter=",", dtype=np.int64)

    if matrix.shape != (ukuran, ukuran):
        raise ValueError(
            f"Ukuran input {ukuran} tidak sesuai dengan ukuran file {matrix.shape}."
        )

    return matrix


def tampilkan_maksimum(max_values, max_columns, ukuran):
    print("\n10 Baris Pertama Hasil Maksimum\n")

    for i in range(min(10, ukuran)):
        print(
            f"User {i+1:4} -> "
            f"User {max_columns[i]+1:4} "
            f"dengan nilai kemiripan kolaboratif Max = {max_values[i]:8}"
        )


def jalankan_sekuensial(matrix):
    print("\n" + "=" * 55)
    print("1. MATRIX MULTIPLICATION - SEQUENTIAL")
    print("=" * 55)
    print("Perkalian matriks dimulai...")

    start = time.perf_counter()
    result = matrix_multiply(matrix)
    max_values, max_columns = find_max_each_row(result)
    execution_time = time.perf_counter() - start

    print(f"Waktu Eksekusi : {execution_time:.6f} detik")
    tampilkan_maksimum(max_values, max_columns, matrix.shape[0])

    return {
        "nama": "Sequential",
        "waktu": execution_time,
        "max_values": max_values,
        "max_columns": max_columns,
    }


def jalankan_numba(matrix, jumlah_thread):
    print("\n" + "=" * 55)
    print("2. MATRIX MULTIPLICATION - NUMBA PARALLEL")
    print("=" * 55)

    set_num_threads(jumlah_thread)

    print("Warm-up JIT compiler...")
    dummy = matrix[:2, :2]
    _ = matrix_multiply_numba(dummy)
    _ = find_max_each_row_numba(dummy)

    print("Perkalian matriks dimulai...")

    start = time.perf_counter()
    result = matrix_multiply_numba(matrix)
    max_values, max_columns = find_max_each_row_numba(result)
    execution_time = time.perf_counter() - start

    print(f"Jumlah Thread  : {get_num_threads()}")
    print(f"Waktu Eksekusi : {execution_time:.6f} detik")
    tampilkan_maksimum(max_values, max_columns, matrix.shape[0])

    return {
        "nama": f"Numba ({get_num_threads()} thread)",
        "waktu": execution_time,
        "max_values": max_values,
        "max_columns": max_columns,
    }


def parse_waktu_mpi(output):
    cocok = re.search(r"Waktu Eksekusi\s*:\s*([0-9.]+)\s*detik", output)
    if not cocok:
        return None

    return float(cocok.group(1))


def jalankan_mpi(ukuran, jumlah_proses):
    print("\n" + "=" * 55)
    print("3. MATRIX MULTIPLICATION - MPI4PY")
    print("=" * 55)
    print(f"Menjalankan MPI dengan {jumlah_proses} proses...")

    script_mpi = Path(__file__).with_name("mpi4py_parallel.py")
    perintah = [
        "mpiexec",
        "-n",
        str(jumlah_proses),
        sys.executable,
        str(script_mpi),
    ]

    try:
        proses = subprocess.run(
            perintah,
            input=f"{ukuran}\n",
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        print(
            "MPI tidak dijalankan karena perintah 'mpiexec' tidak ditemukan. "
            "Pastikan MPI sudah ter-install dan tersedia di PATH."
        )
        return {
            "nama": f"MPI4Py ({jumlah_proses} proses)",
            "waktu": None,
            "error": "mpiexec tidak ditemukan",
        }

    if proses.stdout:
        print(proses.stdout.strip())

    if proses.returncode != 0:
        if proses.stderr:
            print("\nError MPI:")
            print(proses.stderr.strip())
        return {
            "nama": f"MPI4Py ({jumlah_proses} proses)",
            "waktu": None,
            "error": "MPI gagal dijalankan",
        }

    waktu = parse_waktu_mpi(proses.stdout)

    return {
        "nama": f"MPI4Py ({jumlah_proses} proses)",
        "waktu": waktu,
    }


def tampilkan_ringkasan(hasil):
    print("\n" + "=" * 55)
    print("RINGKASAN WAKTU EKSEKUSI")
    print("=" * 55)

    for item in hasil:
        if item["waktu"] is None:
            print(f"{item['nama']:<24}: gagal / tidak tersedia")
        else:
            print(f"{item['nama']:<24}: {item['waktu']:.6f} detik")


def buat_grafik(hasil):
    data_valid = [item for item in hasil if item["waktu"] is not None]

    if not data_valid:
        print("\nGrafik tidak dibuat karena tidak ada data waktu yang valid.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib belum ter-install, grafik batang teks ditampilkan:")
        waktu_terbesar = max(item["waktu"] for item in data_valid)
        for item in data_valid:
            panjang = int((item["waktu"] / waktu_terbesar) * 40)
            print(f"{item['nama']:<24} | {'#' * panjang} {item['waktu']:.6f}s")
        print("\nInstall matplotlib jika ingin grafik PNG: pip install matplotlib")
        return

    nama = [item["nama"] for item in data_valid]
    waktu = [item["waktu"] for item in data_valid]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(nama, waktu, color=["#4C78A8", "#F58518", "#54A24B"][: len(nama)])
    plt.title("Perbandingan Waktu Eksekusi Matrix Multiplication")
    plt.xlabel("Metode")
    plt.ylabel("Waktu Eksekusi (detik)")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, waktu):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output = Path(__file__).with_name("grafik_perbandingan_eksekusi.png")
    plt.savefig(output, dpi=150)
    print(f"\nGrafik disimpan ke: {output}")
    plt.show()


def main():
    print("=" * 55)
    print(" GABUNGAN EKSEKUSI: SEQUENTIAL, NUMBA, DAN MPI4PY")
    print("=" * 55)

    ukuran = int(input("Masukkan ukuran matriks: "))
    jumlah_thread = int(input("Masukkan jumlah thread Numba: "))
    jumlah_proses = int(input("Masukkan jumlah proses MPI4Py: "))

    print("\nMembaca matriks...")
    matrix = load_matrix(ukuran)
    print(f"Ukuran Matriks : {ukuran} x {ukuran}")

    hasil = []
    hasil.append(jalankan_sekuensial(matrix))
    hasil.append(jalankan_numba(matrix, jumlah_thread))
    hasil.append(jalankan_mpi(ukuran, jumlah_proses))

    tampilkan_ringkasan(hasil)
    buat_grafik(hasil)


if __name__ == "__main__":
    main()
