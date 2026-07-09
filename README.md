# Sistem Rekomendasi Berbasis Similarity Matrix

Proyek ini berisi simulasi sistem rekomendasi sederhana untuk mencari user yang paling relevan berdasarkan kemiripan profil. Alur utamanya adalah membuat dataset user, menghitung nilai kemiripan antar user, lalu memproses matrix tersebut dengan perkalian matriks untuk mendapatkan kandidat rekomendasi terkuat.

## Alur Kerja Aplikasi

```text
generate_data_csv.py
        |
        v
users.csv
        |
        v
Matrix_Similarity.py
        |
        v
similarity_matrix_<ukuran>.csv
        |
        +------------------------------+
        |                              |
        v                              v
Sekuensial_Triple_Nested_Loop.py   numba_parallel.py / mpi4py_parallel.py
        |                              |
        v                              v
Hasil rekomendasi user paling mirip per baris
```

## 1. Generate Dataset User

File `generate_data_csv.py` digunakan untuk membuat dataset awal bernama `users.csv`.

Data yang dibuat berisi:

- `UserID`: ID user, misalnya `U0001`
- `Umur`: umur user
- `Gender`: `L` atau `P`
- `Daerah`: asal daerah user
- `Kategori`: minat utama user

Script ini menghasilkan 4096 data user secara acak.

Jalankan:

```bash
py generate_data_csv.py
```

Output:

```text
users.csv
```

## 2. Membuat Similarity Matrix

File `Matrix_Similarity.py` membaca data dari `users.csv`, lalu menghitung skor kemiripan setiap pasangan user.

Aturan skor similarity:

- Selisih umur <= 5: tambah 3
- Selisih umur <= 10: tambah 2
- Selisih umur <= 20: tambah 1
- Gender sama: tambah 2
- Daerah sama: tambah 3
- Kategori sama: tambah 2

Nilai maksimum similarity antar dua user adalah 10.

Jalankan:

```bash
py Matrix_Similarity.py
```

Masukkan ukuran matrix yang ingin dibuat, misalnya:

```text
512
```

Output:

```text
similarity_matrix_512.csv
```

Ukuran yang didukung mengikuti jumlah data yang tersedia, misalnya `512`, `1024`, `2048`, atau `4096`.

## 3. Memproses Matrix untuk Rekomendasi

Setelah similarity matrix dibuat, matrix tersebut diproses dengan perkalian matriks:

```text
C = A x A
```

Keterangan:

- `A` adalah similarity matrix.
- `C` adalah hasil perkalian matrix.
- Nilai `C[i][j]` menunjukkan kekuatan hubungan tidak langsung antara user `i` dan user `j` berdasarkan pola kemiripan terhadap user lain.

Setelah perkalian selesai, program mencari nilai terbesar pada setiap baris matrix `C`.

Artinya:

```text
User i direkomendasikan ke user j jika kolom j memiliki nilai terbesar pada baris i.
```

## 4. Mode Eksekusi

Proyek ini menyediakan tiga pendekatan eksekusi untuk membandingkan performa.

### Sequential

File `Sekuensial_Triple_Nested_Loop.py` menjalankan perkalian matrix secara berurutan menggunakan tiga nested loop.

Jalankan:

```bash
py Sekuensial_Triple_Nested_Loop.py
```

Gunakan mode ini sebagai baseline pembanding performa.

### Numba Parallel

File `numba_parallel.py` menjalankan proses dengan bantuan Numba dan parallel loop (`prange`).

Jalankan:

```bash
py numba_parallel.py
```

Program akan meminta:

- ukuran matrix
- jumlah thread

Mode ini cocok untuk memanfaatkan banyak core CPU pada satu komputer.

### MPI4Py Parallel

File `mpi4py_parallel.py` membagi baris matrix ke beberapa proses menggunakan MPI.

Contoh menjalankan dengan 4 proses:

```bash
mpiexec -n 4 python mpi4py_parallel.py
```

Mode ini cocok untuk pemrosesan paralel berbasis proses, termasuk skenario komputasi terdistribusi jika environment MPI mendukung.

## 5. Interpretasi Output

Contoh output:

```text
User    1 -> Max =     9850 Kolom = 245
```

Artinya user ke-1 memiliki skor hubungan/rekomendasi tertinggi dengan user pada kolom ke-245 berdasarkan hasil perkalian similarity matrix.

Kolom tersebut dapat dianggap sebagai kandidat rekomendasi paling kuat untuk user terkait.

## Urutan Menjalankan Program

Jalankan dari awal dengan urutan berikut:

```bash
py generate_data_csv.py
py Matrix_Similarity.py
py Sekuensial_Triple_Nested_Loop.py
```

Untuk versi paralel, langkah ketiga bisa diganti dengan:

```bash
py numba_parallel.py
```

atau:

```bash
mpiexec -n 4 python mpi4py_parallel.py
```

## Dependensi

Install dependensi utama:

```bash
pip install numpy numba mpi4py
```

Jika hanya menjalankan versi sequential dan generator data, cukup install:

```bash
pip install numpy
```

## Ringkasan

Sistem rekomendasi ini bekerja dengan konsep berikut:

1. Membuat data profil user.
2. Menghitung similarity antar semua pasangan user.
3. Menyimpan similarity dalam bentuk matrix.
4. Mengalikan similarity matrix untuk memperkuat hubungan berbasis pola kemiripan.
5. Mengambil nilai maksimum setiap baris sebagai kandidat rekomendasi user terbaik.

