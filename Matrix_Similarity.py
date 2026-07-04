import csv
import numpy as np


def read_users(filename, limit):
    users = []

    with open(filename, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            users.append({
                "UserID": row["UserID"],
                "Umur": int(row["Umur"]),
                "Gender": row["Gender"],
                "Daerah": row["Daerah"],
                "Kategori": row["Kategori"]
            })

            if len(users) == limit:
                break

    if len(users) < limit:
        raise ValueError(
            f"Jumlah user pada users.csv hanya {len(users)} data."
        )

    return users


def similarity_score(a, b):
    score = 0

    # Umur
    selisih = abs(a["Umur"] - b["Umur"])

    if selisih <= 5:
        score += 3
    elif selisih <= 10:
        score += 2
    elif selisih <= 20:
        score += 1

    # Gender
    if a["Gender"] == b["Gender"]:
        score += 2

    # Daerah
    if a["Daerah"] == b["Daerah"]:
        score += 3

    # Kategori
    if a["Kategori"] == b["Kategori"]:
        score += 2

    return score


def main():

    print("=" * 45)
    print(" GENERATE SIMILARITY MATRIX")
    print("=" * 45)

    ukuran = int(input("Masukkan ukuran matriks (512/1024/2048/4096): "))

    users = read_users("users.csv", ukuran)

    matrix = np.zeros((ukuran, ukuran), dtype=np.int16)

    print("\nMembangun Similarity Matrix...")

    for i in range(ukuran):
        for j in range(ukuran):
            matrix[i][j] = similarity_score(users[i], users[j])

    output = f"similarity_matrix_{ukuran}.csv"

    np.savetxt(output, matrix, fmt="%d", delimiter=",")

    print("\n===================================")
    print("Similarity Matrix berhasil dibuat")
    print("===================================")
    print(f"Ukuran Matriks : {ukuran} x {ukuran}")
    print(f"Output         : {output}")


if __name__ == "__main__":
    main()