import csv
import random

JUMLAH_USER = 4096

daerah = [
    "Jakarta", "Bandung", "Surabaya", "Semarang", "Yogyakarta",
    "Medan", "Makassar", "Palembang", "Denpasar", "Malang",
    "Bogor", "Depok", "Bekasi", "Tangerang", "Batam",
    "Balikpapan", "Samarinda", "Pontianak", "Manado", "Padang"
]

kategori = [
    "Game",
    "Film",
    "Musik",
    "Olahraga",
    "Teknologi",
    "Kuliner",
    "Travel",
    "Fashion",
    "Edukasi",
    "Otomotif"
]

with open("users.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # Header
    writer.writerow([
        "UserID",
        "Umur",
        "Gender",
        "Daerah",
        "Kategori"
    ])

    for i in range(1, JUMLAH_USER + 1):
        user_id = f"U{i:04d}"
        umur = random.randint(16, 80)
        gender = random.choice(["L", "P"])
        daerah_user = random.choice(daerah)
        kategori_user = random.choice(kategori)

        writer.writerow([
            user_id,
            umur,
            gender,
            daerah_user,
            kategori_user
        ])

print("=" * 40)
print("Generator Dataset SnapGram")
print("=" * 40)
print(f"Jumlah User : {JUMLAH_USER}")
print("Output      : users.csv")
print("Selesai.")