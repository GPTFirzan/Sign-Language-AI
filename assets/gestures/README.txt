CARA 1 — OTOMATIS (disarankan)
================================
Kalau kamu sudah punya foto hasil collect_imgs.py di training/data/<index>/,
tinggal jalankan dari folder training/:

    python buat_gambar_kamus.py

Script ini otomatis:
- Mencari foto terbaik di tiap folder training/data/<kelas_index>/
  (yang jumlah tangannya sesuai field "jumlah_tangan" & skor deteksi tertinggi)
- Crop rapi di sekitar tangan
- Simpan ke folder ini dengan nama sesuai field "gambar" di gesture_dictionary.json

Tidak perlu pilih/copy-paste foto manual lagi. Aman dijalankan ulang kapan saja.


CARA 2 — MANUAL (kalau mau pilih foto sendiri)
================================================
Taruh gambar gesture di folder ini secara manual.

Aturan penamaan:
- Nama file HARUS sama persis dengan field "gambar" di gesture_dictionary.json
- Contoh isi gesture_dictionary.json:
    "Halo": { "gambar": "Halo.png", ... }
  -> maka file gambarnya harus: Halo.png (di folder ini)

Format yang didukung: .png, .jpg, .jpeg (apapun yang bisa dibuka PIL/Pillow)
Ukuran bebas, akan otomatis di-resize jadi 160x160 di popup kamus.

Kalau gambar belum ada / nama file salah, aplikasi tidak akan crash —
akan muncul kotak placeholder abu-abu bertanda "?" sebagai penanda.


MENAMBAH KATA BARU KE KAMUS
============================
1. Tambahkan entry baru di gesture_dictionary.json (copy format yang sudah ada),
   isi juga "kelas_index" (folder training/data/<index> untuk gesture itu)
   dan "jumlah_tangan" (1 atau 2)
2. Jalankan ulang buat_gambar_kamus.py (Cara 1), ATAU taruh gambar manual (Cara 2)
3. Tidak perlu edit app.py sama sekali
